# training.py
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from metaflow import FlowSpec, step, Parameter
import joblib 

OUTPUT_DIR = "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.info(xgb.__version__)
logging.info(xgb.build_info())
# Check if GPU is available for XGBoost
try:
    test_data = np.random.rand(10, 2)
    test_labels = np.random.rand(10)
    test_dmatrix = xgb.DMatrix(test_data, label=test_labels)
    test_param = {"device": "cuda:0", "tree_method": "hist"}
    test_bst = xgb.train(test_param, test_dmatrix, num_boost_round=1)
    logging.info("🚀 GPU is available for XGBoost")
    USE_GPU = True
except:
    logging.warning("No GPU found or CUDA not available. Falling back to CPU.")
    USE_GPU = False


class CreditRiskTrainingFlow(FlowSpec):
    @step
    def start(self):
        """Download and preprocess the dataset"""
        url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
        excel_path = "data/default of credit card clients.xls"

        if not os.path.exists(excel_path):
            import zipfile, requests, io

            r = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall("data")

        df = pd.read_excel(excel_path, header=1)
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)

        self.X = df.drop(columns=["default payment next month"])
        self.y = df["default payment next month"]

        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Create advanced features for credit risk prediction"""
        logging.info("Creating advanced features...")
        df = self.X
        # Bill and Payment Dynamics
        bill_cols = [
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
        ]
        pay_amt_cols = [
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]

        df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
        df["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)
        df["TOTAL_BILL_AMT"] = df[bill_cols].sum(axis=1)
        df["TOTAL_PAY_AMT"] = df[pay_amt_cols].sum(axis=1)

        # Utilization and Efficiency Metrics
        df["UTILIZATION_RATIO"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"]
        df["REMAINING_BALANCE"] = df["TOTAL_BILL_AMT"] - df["TOTAL_PAY_AMT"]

        # Volatility Features
        df["BILL_VOLATILITY"] = df[bill_cols].std(axis=1)
        df["PAY_VOLATILITY"] = df[pay_amt_cols].std(axis=1)

        # Update X with new features
        self.X = df
        logging.info(f"Created {len(df.columns) - len(self.X.columns)} new features")
        logging.info(f"Final feature count: {len(df.columns)}")

        self.next(self.prepare)

    @step
    def prepare(self):
        """Split and scale the dataset"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        scaler = StandardScaler()
        self.scaler = scaler 
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

        # Calculate class weights
        counts = np.bincount(self.y_train.astype(int))
        self.scale_pos_weight = (
            counts[0] / counts[1]
        )  # ratio of negative to positive samples

        self.next(self.cross_validate)

    @step
    def cross_validate(self):
        """Perform 5-fold cross-validation and log metrics."""
        import xgboost as xgb

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        device = "cuda" if USE_GPU else "cpu"
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "scale_pos_weight": self.scale_pos_weight,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": device,
        }
        import logging

        logging.info("Starting 5-fold cross-validation with XGBoost...")
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            metrics=["auc", "logloss"],
            early_stopping_rounds=10,
            stratified=True,
            seed=42,
            verbose_eval=False,
        )
        mean_auc = cv_results["test-auc-mean"].iloc[-1]
        std_auc = cv_results["test-auc-std"].iloc[-1]
        mean_logloss = cv_results["test-logloss-mean"].iloc[-1]
        std_logloss = cv_results["test-logloss-std"].iloc[-1]
        logging.info(
            f"5-Fold CV Results: AUC = {mean_auc:.4f} ± {std_auc:.4f}, LogLoss = {mean_logloss:.4f} ± {std_logloss:.4f}"
        )
        self.cv_metrics = {
            "cv_auc_mean": float(mean_auc),
            "cv_auc_std": float(std_auc),
            "cv_logloss_mean": float(mean_logloss),
            "cv_logloss_std": float(std_logloss),
        }
        self.next(self.train)

    @step
    def train(self):
        """Train XGBoost model with GPU support if available"""
        # Convert data to DMatrix format with feature names
        feature_names = list(self.X.columns)
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train, feature_names=feature_names)
        dval = xgb.DMatrix(self.X_test, label=self.y_test, feature_names=feature_names)

        device = "cuda" if USE_GPU else "cpu"
        # Set XGBoost parameters
        logging.info(f"Training on: {device}")
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "scale_pos_weight": self.scale_pos_weight,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": device,
        }

        # Train model with early stopping on full train/test split
        logging.info("Training XGBoost model...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=10,
        )

        # Save the model
        model_path = os.path.join(OUTPUT_DIR, "best_model.json")
        self.model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")

        # Make predictions
        y_proba = self.model.predict(dval)
        y_pred = (y_proba >= 0.5).astype(int)

        # Calculate metrics
        self.metrics = {
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "auc": roc_auc_score(self.y_test, y_proba),
        }

        # --- Plot and save ROC Curve ---
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        logging.info(f"ROC curve saved to {roc_path}")

        # --- Plot and save Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        pr_path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
        plt.savefig(pr_path)
        plt.close()
        logging.info(f"Precision-Recall curve saved to {pr_path}")

        # --- Plot and save Feature Importance ---
        importance = self.model.get_score(importance_type="weight")
        
        if importance:
            # Define feature name mappings
            name_mapping = {
                "LIMIT_BAL": "Credit Limit",
                "SEX": "Gender",
                "EDUCATION": "Education Level",
                "MARRIAGE": "Marital Status",
                "AGE": "Age",
                "PAY_0": "Last Month Payment Status",
                "PAY_2": "2 Months Ago Payment Status",
                "PAY_3": "3 Months Ago Payment Status",
                "PAY_4": "4 Months Ago Payment Status",
                "PAY_5": "5 Months Ago Payment Status",
                "PAY_6": "6 Months Ago Payment Status",
                "BILL_AMT1": "Last Month Bill",
                "BILL_AMT2": "2 Months Ago Bill",
                "BILL_AMT3": "3 Months Ago Bill",
                "BILL_AMT4": "4 Months Ago Bill",
                "BILL_AMT5": "5 Months Ago Bill",
                "BILL_AMT6": "6 Months Ago Bill",
                "PAY_AMT1": "Last Month Payment",
                "PAY_AMT2": "2 Months Ago Payment",
                "PAY_AMT3": "3 Months Ago Payment",
                "PAY_AMT4": "4 Months Ago Payment",
                "PAY_AMT5": "5 Months Ago Payment",
                "PAY_AMT6": "6 Months Ago Payment",
                "PAY_MEAN": "Average Payment Delay",
                "PAY_TREND": "Payment Trend",
                "MAX_PAY_DELAY": "Maximum Payment Delay",
                "PAY_DELAY_VOLATILITY": "Payment Delay Volatility",
                "AVG_BILL_AMT": "Average Bill Amount",
                "AVG_PAY_AMT": "Average Payment Amount",
                "TOTAL_BILL_AMT": "Total Bill Amount",
                "TOTAL_PAY_AMT": "Total Payment Amount",
                "UTILIZATION_RATIO": "Credit Utilization",
                "REMAINING_BALANCE": "Remaining Balance",
                "BILL_VOLATILITY": "Bill Amount Volatility",
                "PAY_VOLATILITY": "Payment Amount Volatility"
            }
            
            # Map technical names to display names and sort by importance
            importance_dict = {name_mapping.get(k, k): v for k, v in importance.items()}
            sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            # Take only top 10 features
            sorted_items = sorted_items[:10]
            keys, values = zip(*sorted_items)

            # Create the plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(keys)), values)
            plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
            plt.title("Top 10 Most Important Features", pad=20)
            plt.xlabel("Features")
            plt.ylabel("Importance Score")

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom"
                )

            plt.tight_layout()
            fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
            plt.savefig(fi_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Feature importance plot saved to {fi_path}")
        else:
            logging.info("No feature importance data available to plot.")

        logging.info(f"Metrics: {self.metrics}")
        logging.info(f"CV Metrics: {self.cv_metrics}")
        self.next(self.output)

    @step
    def output(self):
        """Export final metrics to output/metrics.json"""
        metrics = {
            "precision": round(self.metrics["precision"], 4),
            "recall": round(self.metrics["recall"], 4),
            "f1_score": round(self.metrics["f1"], 4),
            "roc_auc": round(self.metrics["auc"], 4),
        }

        output_path = os.path.join(OUTPUT_DIR, "metrics.json")
        params_path = os.path.join(OUTPUT_DIR, "params.json")
        cv_metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics.json")

        # Get the best iteration from early stopping
        params = {
            "best_iteration": self.model.best_iteration,
            "scale_pos_weight": self.scale_pos_weight,
            "tree_method": "hist",
            "roc_auc": round(self.metrics["auc"], 4),
        }

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        with open(cv_metrics_path, "w") as f:
            json.dump(self.cv_metrics, f, indent=2)

        logging.info(f"Training complete. Metrics written to {output_path}")
        # Save the fitted scaler
        scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
        self.next(self.end)

    @step
    def end(self):
        logging.info("Pipeline finished.")


if __name__ == "__main__":
    CreditRiskTrainingFlow()

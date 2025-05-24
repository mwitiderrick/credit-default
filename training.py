# training.py
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from metaflow import FlowSpec, step, Parameter

OUTPUT_DIR = "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  

# Check if GPU is available for XGBoost
try:
    test_data = np.random.rand(10, 2)
    test_labels = np.random.rand(10)
    test_dmatrix = xgb.DMatrix(test_data, label=test_labels)
    test_param = {'device': 'cuda:0', 'tree_method': 'hist'}
    test_bst = xgb.train(test_param, test_dmatrix, num_boost_round=1)
    logging.info("🚀 GPU is available for XGBoost")
    USE_GPU = True
except:
    logging.warning("⚠️ No GPU found or CUDA not available. Falling back to CPU.")
    USE_GPU = False

class CreditRiskTrainingFlow(FlowSpec):
    auc_threshold = Parameter("auc-threshold", default=0.75)

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

        self.next(self.prepare)

    @step
    def prepare(self):
        """Split and scale the dataset"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

        # Calculate class weights
        counts = np.bincount(self.y_train.astype(int))
        self.scale_pos_weight = counts[0] / counts[1]  # ratio of negative to positive samples
        
        self.next(self.train)

    @step
    def train(self):
        """Train XGBoost model with GPU support if available"""
        # Convert data to DMatrix format
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_test, label=self.y_test)

        # Set XGBoost parameters
        device = 'cuda' if USE_GPU else 'cpu'
        logging.info(f"Training on: {device}")
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'scale_pos_weight': self.scale_pos_weight,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'device': device
        }

        # Train model with early stopping
        logging.info("🚀 Training XGBoost model...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=10
        )

        # Save the model
        model_path = os.path.join(OUTPUT_DIR, 'best_model.json')
        self.model.save_model(model_path)
        logging.info(f"✅ Model saved to {model_path}")

        # Make predictions
        y_proba = self.model.predict(dval)
        y_pred = (y_proba >= 0.5).astype(int)

        # Calculate metrics
        self.metrics = {
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_proba)
        }
        
        logging.info(f"📊 Metrics: {self.metrics}")
        self.next(self.output)

    @step
    def output(self):
        """Export final metrics to output/metrics.json"""
        metrics = {
            "precision": round(self.metrics['precision'], 4),
            "recall": round(self.metrics['recall'], 4),
            "f1_score": round(self.metrics['f1'], 4),
            "roc_auc": round(self.metrics['auc'], 4),
            "threshold": self.auc_threshold,
        }

        output_path = os.path.join(OUTPUT_DIR, "metrics.json")
        params_path = os.path.join(OUTPUT_DIR, "params.json")
        
        # Get the best iteration from early stopping
        params = {
            "best_iteration": self.model.best_iteration,
            "scale_pos_weight": self.scale_pos_weight,
            "tree_method": "hist" if USE_GPU else "cpu",
            "roc_auc": round(self.metrics['auc'], 4),
        }
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        logging.info(f"✅ Training complete. Metrics written to {output_path}")
        self.next(self.end)

    @step
    def end(self):
        logging.info("✅ Pipeline finished.")

if __name__ == "__main__":
    CreditRiskTrainingFlow()

import os
import json
import numpy as np
import xgboost as xgb
import pandas as pd
import joblib

MODEL_DIR = os.getenv("MODEL_DIR", "/output")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(os.getenv("MODEL_DIR", "output"), "best_model.json")

# Raw input features expected from user
RAW_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# For compatibility with tests
FEATURE_NAMES = RAW_FEATURES

def load_model():
    """Load the saved XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

def apply_feature_engineering(df):
    """Create the same engineered features as used during training."""

    # Bill and Payment Dynamics
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

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

    return df

def predict_default(model, input_data):
    """
    Predict the probability of default.
    Args:
        model: loaded XGBoost model
        input_data: 2D list or numpy array of raw input features
    Returns:
        List of dictionaries containing prediction results
    """
    # Convert input to DataFrame with raw feature names
    df = pd.DataFrame(input_data, columns=RAW_FEATURES)

    # Feature engineering
    df = apply_feature_engineering(df)

    # Scale using standard scaler (fitted during training - ideally, load it)
    scaler = joblib.load(SCALER_PATH)

    input_scaled = scaler.transform(df)  # In production, load the fitted scaler

    # Use same column names as training
    dtest = xgb.DMatrix(input_scaled, feature_names=df.columns.tolist())

    # Predict
    probabilities = model.predict(dtest)

    # Format output
    results = []
    for prob in probabilities:
        results.append({
            "probability": round(float(prob), 4),
            "prediction": "DEFAULT" if prob >= 0.5 else "NO DEFAULT",
            "risk_level": (
                "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
            ),
        })
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    model = load_model()
    predictions = predict_default(model, input_data)
    print(json.dumps({"predictions": predictions}, indent=2))

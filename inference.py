# inference.py
import os
import json
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd

MODEL_PATH = os.path.join(os.getenv("MODEL_DIR", "/output"), "best_model.json")
FEATURE_NAMES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def load_model():
    """Load the saved XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model


def predict_default(model, input_data):
    """
    Predict the probability of default.
    Args:
        model: loaded XGBoost model
        input_data: 2D list or numpy array of features
    Returns:
        List of dictionaries containing prediction results
    """
    # Convert input to DataFrame with correct feature names
    input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)

    # Scale the features (using the same logic as in training)
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)

    # Convert to DMatrix
    dtest = xgb.DMatrix(input_scaled)

    # Get predictions
    probabilities = model.predict(dtest)

    # Format results
    results = []
    for prob in probabilities:
        results.append(
            {
                "probability": round(float(prob), 4),
                "prediction": "DEFAULT" if prob >= 0.5 else "NO DEFAULT",
                "risk_level": (
                    "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
                ),
            }
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JSON file"
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f)

    model = load_model()
    predictions = predict_default(model, input_data)
    print(json.dumps({"predictions": predictions}, indent=2))

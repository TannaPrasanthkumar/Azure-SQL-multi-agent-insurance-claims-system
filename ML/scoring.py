"""
Final Production-Grade Scoring Script for Azure ML Deployment
Fraud Detection Model – With Preprocessing, Encoders, Scaler & Metadata
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Global variables for Azure
model = None
label_encoders = {}
scaler = None
model_metadata = {}


def init():
    """
    Initialize the model, scaler, encoders, and metadata.
    Called once when Azure ML spins up the container.
    """
    global model, label_encoders, scaler, model_metadata

    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
        model_dir = os.path.join(model_dir, "models")

        # Load model
        model_path = os.path.join(model_dir, "balanced_random_forest_fraud_detector_0.pkl")
        model = joblib.load(model_path)
        print(f"[INIT] Model loaded: {model_path}")

        # Load label encoders
        enc_path = os.path.join(model_dir, "label_encoders_0.pkl")
        label_encoders = joblib.load(enc_path)
        print(f"[INIT] Label encoders loaded: {enc_path}")

        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler_0.pkl")
        scaler = joblib.load(scaler_path)
        print(f"[INIT] Scaler loaded: {scaler_path}")

        # Load metadata
        meta_path = os.path.join(model_dir, "model_metadata_0.pkl")
        model_metadata = joblib.load(meta_path)
        print(f"[INIT] Metadata loaded: {meta_path}")

        print("[INIT] Initialization completed successfully.")

    except Exception as e:
        print(f"[INIT-ERROR] {str(e)}")
        raise e


def run(raw_data):
    """
    Azure ML entry point for each request.
    raw_data is always a JSON-formatted string.
    """
    try:
        # Parse JSON
        data = json.loads(raw_data)

        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return json.dumps({"error": "Invalid JSON input format."})

        # Validate input
        missing = validate_input(df)
        if missing:
            return json.dumps({
                "error": "Missing required fields.",
                "missing_fields": missing
            })

        # Preprocess
        X = preprocess(df)

        # Predict labels
        y_pred = model.predict(X)

        # Predict probabilities (fallback safe)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # Format response
        output = []
        for p, prob in zip(y_pred, y_proba):
            output.append({
                "fraud_prediction": int(p),
                "fraud_probability": float(prob),
                "fraud_risk": get_risk_level(prob)
            })

        return json.dumps({"predictions": output})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ------------------ Helper Functions ------------------

def validate_input(df):
    """Check for missing required fields."""
    required = model_metadata.get("features", model_metadata.get("feature_columns", []))
    missing = [col for col in required if col not in df.columns]
    return missing


def safe_transform(encoder, series):
    """
    Apply encoder safely.
    Unseen categories → replaced with first known class.
    """
    series = series.astype(str)
    known = set(encoder.classes_)

    # Replace unseen labels
    series = series.apply(lambda x: x if x in known else encoder.classes_[0])
    return encoder.transform(series)


def preprocess(df):
    """Full preprocessing pipeline identical to training."""

    df_proc = df.copy()

    feature_columns = model_metadata.get("features", model_metadata.get("feature_columns", []))
    categorical_cols = model_metadata.get("categorical_features", model_metadata.get("categorical_columns", []))
    default_fill = model_metadata.get("default_fill_value", 0)

    # Fill missing values
    df_proc.fillna(default_fill, inplace=True)

    # Encode categorical variables safely
    for col in categorical_cols:
        if col in df_proc.columns and col in label_encoders:
            df_proc[col] = safe_transform(label_encoders[col], df_proc[col])

    # Ensure every required feature exists
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = default_fill

    # Select required features (correct order)
    df_proc = df_proc[feature_columns]

    # Scale - keep as DataFrame to preserve feature names
    df_scaled = pd.DataFrame(
        scaler.transform(df_proc),
        columns=feature_columns,
        index=df_proc.index
    )

    return df_scaled


def get_risk_level(prob):
    """Map probability to fraud risk."""
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.5:
        return "Medium Risk"
    elif prob >= 0.3:
        return "Low Risk"
    else:
        return "Very Low Risk"


# For local testing
if __name__ == "__main__":
    os.environ["AZUREML_MODEL_DIR"] = os.path.dirname(__file__)
    print("Running local test...")
    init()

    # Sample test data - using actual features from trained model
    sample = {
        "DriverRating": 3,
        "Age": 35,
        "PoliceReportFiled": "Yes",
        "WeekOfMonthClaimed": 2,
        "PolicyType": "Sport - Liability",
        "WeekOfMonth": 3,
        "AccidentArea": "Urban",
        "Sex": "Male",
        "Deductible": 500
    }

    print("\n" + "="*60)
    print("LOCAL TEST RESULT:")
    print("="*60)
    result = run(json.dumps(sample))
    print(json.dumps(json.loads(result), indent=2))
    print("="*60)


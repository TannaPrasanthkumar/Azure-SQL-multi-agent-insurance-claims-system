"""
Test Scoring Script with Optimal Threshold (0.7)
Resilient init() for Azure ML deployments — searches for required artifacts
in nested folders (e.g., "1/", "New folder/", "models/") and fails gracefully
so liveness/readiness probes don't repeatedly crash the container.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Global variables
model = None
label_encoders = {}
scaler = None
model_metadata = {}
_initialized = False


def _find_file(base_dir: Path, filename: str):
    """
    Recursively search base_dir for filename. Return Path or None.
    """
    # check exact path first
    candidate = base_dir / filename
    if candidate.exists():
        return candidate

    # recursive search
    for p in base_dir.rglob(filename):
        if p.is_file():
            return p
    return None


def _find_model_and_artifacts(base_dir: Path):
    """
    Locate model metadata, model file, label encoders, and scaler.
    Returns dict of paths (Path objects) or None if not found.
    """
    artifacts = {}

    # metadata
    meta = _find_file(base_dir, "model_metadata_0.pkl")
    if meta:
        artifacts["metadata"] = meta
    else:
        # sometimes metadata is directly inside nested folder names
        # allow fallback to any file named model_metadata*.pkl
        for p in base_dir.rglob("model_metadata*.pkl"):
            artifacts["metadata"] = p
            break

    # label encoders & scaler
    le = _find_file(base_dir, "label_encoders_0.pkl")
    if le:
        artifacts["label_encoders"] = le
    else:
        for p in base_dir.rglob("label_encoders*.pkl"):
            artifacts["label_encoders"] = p
            break

    sc = _find_file(base_dir, "scaler_0.pkl")
    if sc:
        artifacts["scaler"] = sc
    else:
        for p in base_dir.rglob("scaler*.pkl"):
            artifacts["scaler"] = p
            break

    # model file: prefer name in metadata if present, otherwise search for common patterns
    artifacts["model_file_candidate"] = None
    if "metadata" in artifacts:
        try:
            mdata = joblib.load(artifacts["metadata"])
            model_filename = mdata.get("model_filename")
            if model_filename:
                mf = _find_file(base_dir, model_filename)
                if mf:
                    artifacts["model_file_candidate"] = mf
                else:
                    # try relative to metadata location
                    mf2 = artifacts["metadata"].parent / model_filename
                    if mf2.exists():
                        artifacts["model_file_candidate"] = mf2
            # store metadata content for later
            artifacts["metadata_obj"] = mdata
        except Exception:
            # we'll still try to search for model file below
            artifacts["metadata_obj"] = None

    # fallback: search for model files with expected names
    if not artifacts.get("model_file_candidate"):
        patterns = [
            "balanced_random_forest_fraud_detector_0.pkl",
            "model.pkl",
            "*.pkl"
        ]
        for pat in patterns:
            for p in base_dir.rglob(pat):
                # Avoid picking metadata/scaler/label encoder files
                if p.name in ("model_metadata_0.pkl", "label_encoders_0.pkl", "scaler_0.pkl"):
                    continue
                artifacts["model_file_candidate"] = p
                break
            if artifacts.get("model_file_candidate"):
                break

    return artifacts


def init():
    """
    Initialize the model, scaler, encoders, and metadata.
    This function is robust to different model folder layouts.
    It logs problems but does not raise to avoid container crash on missing files.
    """
    global model, label_encoders, scaler, model_metadata, _initialized

    try:
        # Use AZUREML_MODEL_DIR if set, otherwise current dir
        azure_model_dir = os.getenv("AZUREML_MODEL_DIR")
        if azure_model_dir:
            base_dir = Path(azure_model_dir)
        else:
            base_dir = Path(".").resolve()

        # If base_dir contains a numbered subfolder (common in Azure ML), prefer it
        # e.g. /var/azureml-app/azureml-models/<model-name>/1
        # but we still search recursively so it's resilient.
        print(f"[INIT] Starting initialization. Base directory: {base_dir}")

        artifacts = _find_model_and_artifacts(base_dir)

        # metadata
        meta_path = artifacts.get("metadata")
        if meta_path and meta_path.exists():
            try:
                model_metadata = joblib.load(meta_path)
                print(f"[INIT] Loaded metadata from: {meta_path}")
            except Exception as e:
                model_metadata = {}
                print(f"[INIT-WARN] Failed to load metadata {meta_path}: {e}")
        else:
            model_metadata = {}
            print("[INIT-WARN] model_metadata_0.pkl not found. Proceeding with defaults.")

        # model
        model_path = artifacts.get("model_file_candidate")
        if model_path and model_path.exists():
            try:
                model = joblib.load(model_path)
                print(f"[INIT] Model loaded from: {model_path}")
            except Exception as e:
                model = None
                print(f"[INIT-ERROR] Failed to load model at {model_path}: {e}")
        else:
            print("[INIT-WARN] Model file not found in model directory.")

        # label encoders
        enc_path = artifacts.get("label_encoders")
        if enc_path and enc_path.exists():
            try:
                label_encoders = joblib.load(enc_path)
                print(f"[INIT] Label encoders loaded from: {enc_path}")
            except Exception as e:
                label_encoders = {}
                print(f"[INIT-WARN] Failed to load label encoders {enc_path}: {e}")
        else:
            label_encoders = {}
            print("[INIT-WARN] label_encoders_0.pkl not found.")

        # scaler
        sc_path = artifacts.get("scaler")
        if sc_path and sc_path.exists():
            try:
                scaler = joblib.load(sc_path)
                print(f"[INIT] Scaler loaded from: {sc_path}")
            except Exception as e:
                scaler = None
                print(f"[INIT-WARN] Failed to load scaler {sc_path}: {e}")
        else:
            scaler = None
            print("[INIT-WARN] scaler_0.pkl not found.")

        # If metadata provided an optimal threshold, show it
        optimal_threshold = model_metadata.get("optimal_threshold", 0.5)
        print(f"[INIT] Optimal threshold (from metadata or default): {optimal_threshold}")

        # Mark as initialized only if model and scaler present
        if model is not None and scaler is not None:
            _initialized = True
            print("[INIT] Initialization completed successfully.")
        else:
            _initialized = False
            missing = []
            if model is None:
                missing.append("model")
            if scaler is None:
                missing.append("scaler")
            print(f"[INIT-WARN] Initialization incomplete. Missing artifacts: {missing}")
            # don't raise: allow container to stay up and respond with an informative error

    except Exception as e:
        # Log everything and avoid raising to prevent repeated probe failures
        _initialized = False
        print(f"[INIT-ERROR] Unexpected init error: {e}")


def run(raw_data):
    """
    Entry point for each request with optimal threshold.
    Returns JSON string.
    """
    try:
        if not _initialized:
            # Be explicit: return error JSON instead of crashing container
            msg = "Model not initialized. Check logs for missing artifacts (model, scaler, metadata)."
            print(f"[RUN-ERROR] {msg}")
            return json.dumps({"error": msg})

        # Parse input JSON
        try:
            data = json.loads(raw_data)
        except Exception:
            return json.dumps({"error": "Invalid JSON input format."})

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

        # Get optimal threshold from metadata
        optimal_threshold = model_metadata.get("optimal_threshold", 0.5)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            # fallback: if model only predicts labels, treat as deterministic
            y_proba = model.predict(X).astype(float)

        # Apply optimal threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)

        # Format response
        output = []
        for p, prob in zip(y_pred, y_proba):
            output.append({
                "fraud_prediction": int(p),
                "fraud_probability": float(prob),
                "fraud_risk": get_risk_level(float(prob), optimal_threshold),
                "threshold_used": optimal_threshold
            })

        return json.dumps({"predictions": output})

    except Exception as e:
        # unexpected runtime error
        print(f"[RUN-EXCEPTION] {e}")
        return json.dumps({"error": str(e)})


# ------------------ Helper Functions ------------------

def validate_input(df):
    """Check for missing required fields."""
    required = model_metadata.get("features", model_metadata.get("feature_columns", []))
    # if metadata empty, try picking all columns present in scaler mean/var if available
    if not required and scaler is not None:
        try:
            # attempt to infer feature names if scaler is a DataFrame/has feature names
            if hasattr(scaler, "feature_names_in_"):
                required = list(getattr(scaler, "feature_names_in_"))
        except Exception:
            required = []
    missing = [col for col in required if col not in df.columns]
    return missing


def safe_transform(encoder, series):
    """
    Apply encoder safely.
    Unseen categories → replaced with first known class.
    """
    series = series.astype(str)
    known = set(getattr(encoder, "classes_", []))

    if not known:
        # no-op if encoder not populated
        return encoder.transform(series)

    # Replace unseen labels with the first known class
    series = series.apply(lambda x: x if x in known else list(known)[0])
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
            try:
                df_proc[col] = safe_transform(label_encoders[col], df_proc[col])
            except Exception as e:
                print(f"[PREPROC-WARN] Failed to transform column {col}: {e}")

    # Ensure every required feature exists
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = default_fill

    # Select required features (correct order)
    if feature_columns:
        df_proc = df_proc[feature_columns]

    # Scale - ensure scaler available
    if scaler is None:
        raise RuntimeError("Scaler not loaded during init; cannot preprocess inputs.")
    df_scaled = pd.DataFrame(
        scaler.transform(df_proc),
        columns=feature_columns if feature_columns else df_proc.columns,
        index=df_proc.index
    )

    return df_scaled


def get_risk_level(prob, threshold=0.5):
    """Map probability to fraud risk based on threshold."""
    if prob >= threshold:
        return "High Risk (Fraud Detected)"
    elif prob >= threshold - 0.2:
        return "Medium Risk"
    elif prob >= threshold - 0.4:
        return "Low Risk"
    else:
        return "Very Low Risk"


# For local testing
if __name__ == "__main__":
    # local base - helpful when running file directly
    # set AZUREML_MODEL_DIR to current folder for local testing
    os.environ.setdefault("AZUREML_MODEL_DIR", os.path.dirname(__file__))
    print("Running local test with resilient init()...")
    init()

    # Sample test data - using actual features from trained model
    test_samples = [
        {
            "DriverRating": 3,
            "Age": 35,
            "PoliceReportFiled": "Yes",
            "WeekOfMonthClaimed": 2,
            "PolicyType": "Sport - Liability",
            "WeekOfMonth": 3,
            "AccidentArea": "Urban",
            "Sex": "Male",
            "Deductible": 500
        },
        {
            "DriverRating": 1,
            "Age": 25,
            "PoliceReportFiled": "No",
            "WeekOfMonthClaimed": 4,
            "PolicyType": "Sport - Collision",
            "WeekOfMonth": 4,
            "AccidentArea": "Rural",
            "Sex": "Male",
            "Deductible": 700
        },
        {
            "DriverRating": 4,
            "Age": 45,
            "PoliceReportFiled": "Yes",
            "WeekOfMonthClaimed": 1,
            "PolicyType": "Sedan - All Perils",
            "WeekOfMonth": 1,
            "AccidentArea": "Urban",
            "Sex": "Female",
            "Deductible": 400
        }
    ]

    print("\n" + "="*60)
    print("LOCAL TEST RESULTS WITH OPTIMAL THRESHOLD:")
    print("="*60)

    for i, sample in enumerate(test_samples, 1):
        print(f"\nTest Sample {i}:")
        print(f"  Input: {sample}")
        result = run(json.dumps(sample))
        try:
            result_dict = json.loads(result)
        except Exception:
            result_dict = {"error": "invalid json returned"}
        print(f"  Output: {json.dumps(result_dict, indent=4)}")

    print("="*60)

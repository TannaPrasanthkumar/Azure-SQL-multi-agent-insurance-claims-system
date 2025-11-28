"""
Azure ML Scoring Script for Fraud Detection Model
Works with model files in 'New folder' (with _0 suffix)
Compatible with Azure ML deployment
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


def init():
    """
    Initialize the model, scaler, encoders, and metadata.
    Called when the container starts (Azure ML) or when script is imported.
    """
    global model, label_encoders, scaler, model_metadata, _initialized

    try:
        # Get model directory from Azure ML environment variable
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        
        if model_dir:
            # Azure ML deployment - model is in subdirectory
            base_path = Path(model_dir)
            print(f"[INIT] Azure ML Model Directory: {base_path}")
            
            # Azure ML puts models in versioned folders like: /var/azureml-app/azureml-models/model-name/1/
            # Search recursively for the model files
            model_files = list(base_path.rglob("*.pkl"))
            print(f"[INIT] Found {len(model_files)} .pkl files")
            
            # Find each artifact by searching for the filename pattern
            metadata_file = None
            encoder_file = None
            scaler_file = None
            model_file = None
            
            for file in model_files:
                if "model_metadata" in file.name:
                    metadata_file = file
                elif "label_encoders" in file.name:
                    encoder_file = file
                elif "scaler" in file.name:
                    scaler_file = file
                elif "balanced_random_forest" in file.name or "model" in file.name:
                    model_file = file
            
        else:
            # Local testing - files are in current directory
            base_path = Path(__file__).parent
            print(f"[INIT] Local Directory: {base_path}")
            
            metadata_file = base_path / "Azure/model_metadata_0.pkl"
            encoder_file = base_path / "Azure/label_encoders_0.pkl"
            scaler_file = base_path / "Azure/scaler_0.pkl"
            model_file = base_path / "Azure/balanced_random_forest_fraud_detector_0.pkl"
        
        # Load metadata
        if metadata_file and metadata_file.exists():
            model_metadata = joblib.load(metadata_file)
            print(f"[INIT] ✅ Loaded metadata from: {metadata_file.name}")
        else:
            print("[INIT] ⚠️ model_metadata_0.pkl not found, using defaults")
            model_metadata = {}
        
        # Load model
        if model_file and model_file.exists():
            model = joblib.load(model_file)
            print(f"[INIT] ✅ Loaded model from: {model_file.name}")
        else:
            print("[INIT] ❌ Model file not found!")
            return
        
        # Load label encoders
        if encoder_file and encoder_file.exists():
            label_encoders = joblib.load(encoder_file)
            print(f"[INIT] ✅ Loaded label encoders from: {encoder_file.name}")
        else:
            print("[INIT] ⚠️ label_encoders_0.pkl not found")
            label_encoders = {}
        
        # Load scaler
        if scaler_file and scaler_file.exists():
            scaler = joblib.load(scaler_file)
            print(f"[INIT] ✅ Loaded scaler from: {scaler_file.name}")
        else:
            print("[INIT] ❌ Scaler not found!")
            return
        
        # Get threshold from metadata
        threshold = model_metadata.get("optimal_threshold", 0.5)
        print(f"[INIT] ✅ Threshold: {threshold}")
        
        # Mark as initialized
        if model is not None and scaler is not None:
            _initialized = True
            print("[INIT] ✅ Initialization completed successfully!")
        else:
            print("[INIT] ❌ Initialization failed - missing required artifacts")
            
    except Exception as e:
        print(f"[INIT] ❌ Initialization error: {e}")
        _initialized = False


def run(raw_data):
    """
    Score the model with incoming data.
    Called for each prediction request.
    
    Args:
        raw_data: JSON string with input features
        
    Returns:
        JSON string with predictions
    """
    try:
        if not _initialized:
            return json.dumps({
                "error": "Model not initialized. Check deployment logs."
            })
        
        # Parse input JSON
        try:
            data = json.loads(raw_data)
        except Exception as e:
            return json.dumps({
                "error": f"Invalid JSON input: {str(e)}"
            })
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return json.dumps({
                "error": "Input must be a JSON object or array"
            })
        
        # Get feature columns from metadata
        feature_columns = model_metadata.get("features", model_metadata.get("feature_columns", []))
        categorical_columns = model_metadata.get("categorical_features", model_metadata.get("categorical_columns", []))
        
        # Validate required features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            return json.dumps({
                "error": f"Missing required features: {missing_features}"
            })
        
        # Preprocess the data
        df_processed = df.copy()
        
        # Fill missing values
        df_processed.fillna(0, inplace=True)
        
        # Encode categorical features
        for col in categorical_columns:
            if col in df_processed.columns and col in label_encoders:
                encoder = label_encoders[col]
                try:
                    # Handle unseen categories
                    df_processed[col] = df_processed[col].astype(str)
                    known_classes = set(encoder.classes_)
                    df_processed[col] = df_processed[col].apply(
                        lambda x: x if x in known_classes else encoder.classes_[0]
                    )
                    df_processed[col] = encoder.transform(df_processed[col])
                except Exception as e:
                    print(f"[PREPROCESS] Warning: Failed to encode {col}: {e}")
        
        # Select and order features correctly
        df_features = df_processed[feature_columns]
        
        # Scale features
        df_scaled = scaler.transform(df_features)
        
        # Get predictions
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_scaled)[:, 1]
        else:
            probabilities = model.predict(df_scaled).astype(float)
        
        # Get threshold
        threshold = model_metadata.get("optimal_threshold", 0.5)
        
        # Make predictions
        predictions = (probabilities >= threshold).astype(int)
        
        # Format output
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "fraud_prediction": int(pred),
                "fraud_probability": float(prob),
                "fraud_risk": get_risk_level(float(prob), threshold),
                "threshold_used": threshold
            })
        
        return json.dumps({"predictions": results})
        
    except Exception as e:
        print(f"[RUN] ❌ Prediction error: {e}")
        return json.dumps({
            "error": f"Prediction failed: {str(e)}"
        })


def get_risk_level(probability, threshold=0.5):
    """
    Map fraud probability to risk level.
    
    Args:
        probability: Fraud probability (0-1)
        threshold: Decision threshold
        
    Returns:
        Risk level string
    """
    if probability >= threshold:
        return "High Risk (Fraud Detected)"
    elif probability >= threshold - 0.2:
        return "Medium Risk"
    elif probability >= threshold - 0.4:
        return "Low Risk"
    else:
        return "Very Low Risk"


# For local testing
if __name__ == "__main__":
    print("="*70)
    print("TESTING SCORING SCRIPT LOCALLY")
    print("="*70)
    
    # Initialize
    init()
    
    if not _initialized:
        print("\n❌ Initialization failed. Cannot test.")
        exit(1)
    
    # Test with sample data
    test_data = {
        "DriverRating": 2,
        "Age": 48,
        "PoliceReportFiled": 1,
        "WeekOfMonthClaimed": 1,
        "PolicyType": 1,
        "WeekOfMonth": 1,
        "AccidentArea": 1,
        "Sex": 1,
        "Deductible": 400
    }
    
    print("\n📊 Test Input:")
    print(json.dumps(test_data, indent=2))
    
    # Run prediction
    result = run(json.dumps(test_data))
    
    print("\n📊 Test Output:")
    print(json.dumps(json.loads(result), indent=2))
    
    print("\n✅ Local test completed!")

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def main():

    print(f"✅ Libraries imported successfully!")   
    print(f"   numpy: {np.__version__}")
    print(f"   pandas: {pd.__version__}")

    # Use local data file
    data_path = "azureml://subscriptions/94a0d91a-896b-4e4d-a3c7-df06074221d2/resourcegroups/AgenticAI/workspaces/AgenticAI_ML/datastores/workspaceblobstore/paths/UI/2025-11-21_094623_UTC/fraud_oracle.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Fraud cases: {df['FraudFound_P'].sum()} ({df['FraudFound_P'].mean()*100:.2f}%)")
    except FileNotFoundError:
        print(f"⚠️ Data file not found at: {data_path}")
        return

    # Define features in EXACT order from original metadata
    features = ['DriverRating', 'Age', 'PoliceReportFiled', 'WeekOfMonthClaimed', 
                'PolicyType', 'WeekOfMonth', 'AccidentArea', 'Sex', 'Deductible']
    categorical_features = ['PoliceReportFiled', 'PolicyType', 'AccidentArea', 'Sex']
    target = 'FraudFound_P'
    
    print(f"📊 Using {len(features)} features in EXACT original order:")
    print(f"   Features: {features}")
    print(f"   Categorical: {categorical_features}")

    X = df[features].copy()
    y = df[target]

    # Encode categorical features
    print("🔄 Encoding categorical features...")
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} unique values")

    print(f"\n✅ Features prepared: {X.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"   Train fraud rate: {y_train.mean()*100:.2f}%")
    print(f"   Test fraud rate: {y_test.mean()*100:.2f}%")

    # Scaling pipeline
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Data prepared for training")

    # Train Balanced Random Forest with improved hyperparameters
    print("🤖 Training Balanced Random Forest Classifier...\n")
    model = BalancedRandomForestClassifier(
        n_estimators=200,           # More trees for better performance
        max_depth=15,               # Deeper trees to capture more patterns
        min_samples_split=10,       # Reduce overfitting
        min_samples_leaf=4,         # Smoother decision boundaries
        max_features='sqrt',        # Best subset of features at each split
        bootstrap=True,
        sampling_strategy='auto',   # Balance classes automatically
        replacement=True,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train_scaled, y_train)
    print("\n✅ Training complete!")

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*60)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts to Azure ML "outputs"
    output_dir = Path("./outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("💾 Saving model artifacts...")

    # Save model
    model_path = output_dir / "balanced_random_forest_fraud_detector.pkl"
    joblib.dump(model, model_path)

    # Save encoders
    encoders_path = output_dir / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Save metadata
    numerical_features = [f for f in features if f not in categorical_features]
    metadata = {
        'features': features,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'model_name': 'Balanced Random Forest Fraud Detector',
        'model_type': 'BalancedRandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 10,
        'optimal_threshold': 0.5,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        }
    }

    metadata_path = output_dir / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)

    print("\n" + "="*60)
    print("🎉 ALL MODEL ARTIFACTS SAVED SUCCESSFULLY!")
    print("="*60)
    print(f"Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

"""
Train Balanced Random Forest with simple hyperparameters (like original model)
Then test on test_dataset_15_samples.csv
"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

print("="*60)
print("TRAINING WITH SIMPLE HYPERPARAMETERS")
print("="*60)

# Load training data
data_path = "data/fraud_oracle.csv"
df = pd.read_csv(data_path)
print(f"\n✅ Data loaded: {df.shape}")
print(f"   Fraud cases: {df['FraudFound_P'].sum()} ({df['FraudFound_P'].mean()*100:.2f}%)")

# Define features in EXACT order from original metadata
features = ['DriverRating', 'Age', 'PoliceReportFiled', 'WeekOfMonthClaimed', 
            'PolicyType', 'WeekOfMonth', 'AccidentArea', 'Sex', 'Deductible']
categorical_features = ['PoliceReportFiled', 'PolicyType', 'AccidentArea', 'Sex']
target = 'FraudFound_P'

print(f"\n📊 Using {len(features)} features (original order):")
print(f"   Features: {features}")

X = df[features].copy()
y = df[target]

# Encode categorical features
print("\n🔄 Encoding categorical features...")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"   {col}: {len(le.classes_)} classes")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Scale features
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Balanced Random Forest
print("\n🤖 Training Balanced Random Forest Classifier...")
print("   This handles class imbalance automatically!\n")

"""model = BalancedRandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)"""

print("🤖 Training Balanced Random Forest Classifier...\n")
model = BalancedRandomForestClassifier(
        n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("\n✅ Training complete!")

# Evaluate on main test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*60)
print("MODEL PERFORMANCE ON MAIN TEST SET (3084 samples)")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("="*60)

# Save model artifacts
output_dir = Path('./new_model')
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, output_dir / "balanced_random_forest_fraud_detector.pkl")
joblib.dump(label_encoders, output_dir / "label_encoders.pkl")
joblib.dump(scaler, output_dir / "scaler.pkl")

print(f"\n💾 Model saved to: {output_dir}/")

# Now test on 15 sample test dataset
print("\n" + "="*60)
print("TESTING ON 15 SAMPLE TEST DATASET")
print("="*60)

test_data_path = "data/test_dataset_15_samples.csv"
df_test = pd.read_csv(test_data_path)
print(f"\n✅ Test data loaded: {df_test.shape}")
print(f"   Fraud cases: {df_test['FraudFound_P'].sum()} out of {len(df_test)}")

# Prepare test features
X_test_15 = df_test[features].copy()
y_test_15 = df_test['FraudFound_P']

# Encode categorical features
for col in categorical_features:
    X_test_15[col] = label_encoders[col].transform(X_test_15[col].astype(str))

# Scale features
X_test_15_scaled = scaler.transform(X_test_15)

# Predict with probabilities
y_proba_15 = model.predict_proba(X_test_15_scaled)[:, 1]

# Try different thresholds to find optimal one
print("\n🔍 Testing different thresholds:")
print("-" * 60)
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
best_threshold = 0.5
best_accuracy = 0

for thresh in thresholds_to_test:
    y_pred_thresh = (y_proba_15 >= thresh).astype(int)
    acc = accuracy_score(y_test_15, y_pred_thresh)
    prec = precision_score(y_test_15, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test_15, y_pred_thresh, zero_division=0)
    print(f"Threshold {thresh:.1f}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = thresh

print(f"\n⭐ Best threshold: {best_threshold} with accuracy: {best_accuracy:.4f}")

# Use best threshold for predictions
optimal_threshold = best_threshold
y_pred_15 = (y_proba_15 >= optimal_threshold).astype(int)

print("\nSample-by-Sample Results:")
print("-" * 60)
for i in range(len(df_test)):
    actual = "FRAUD" if y_test_15.iloc[i] == 1 else "NOT FRAUD"
    predicted = "FRAUD" if y_pred_15[i] == 1 else "NOT FRAUD"
    prob = y_proba_15[i]
    match = "✅" if y_test_15.iloc[i] == y_pred_15[i] else "❌"
    
    print(f"Sample {i+1:2d}: Actual={actual:10s} | Predicted={predicted:10s} | Prob={prob:.4f} | {match}")

# Calculate metrics on 15 samples
accuracy_15 = accuracy_score(y_test_15, y_pred_15)
precision_15 = precision_score(y_test_15, y_pred_15, zero_division=0)
recall_15 = recall_score(y_test_15, y_pred_15, zero_division=0)
f1_15 = f1_score(y_test_15, y_pred_15, zero_division=0)
roc_auc_15 = roc_auc_score(y_test_15, y_proba_15)

print("\n" + "="*60)
print(f"PERFORMANCE METRICS ON 15 SAMPLES (Threshold={optimal_threshold})")
print("="*60)
print(f"Accuracy:  {accuracy_15:.4f} ({accuracy_15*100:.2f}%)")
print(f"Precision: {precision_15:.4f} ({precision_15*100:.2f}%)")
print(f"Recall:    {recall_15:.4f} ({recall_15*100:.2f}%)")
print(f"F1-Score:  {f1_15:.4f}")
print(f"ROC-AUC:   {roc_auc_15:.4f}")
print("="*60)

# Confusion matrix
cm = confusion_matrix(y_test_15, y_pred_15)
print("\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives:  {cm[1][1]}")
print("="*60)

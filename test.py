"""
Test Azure ML Deployed Endpoint with test_dataset_15_samples.csv
Enhanced version: stability fixes, retry logic, clean formatting, safer parsing
"""

import os
import pandas as pd
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("           TESTING AZURE ML REAL-TIME ENDPOINT")
print("="*70)

# Azure ML Environment Variables
scoring_uri = os.getenv("AZURL_ML_ENDPOINT")  # Note: typo in .env, should be AZURE_ML_ENDPOINT
api_key = os.getenv("AZURE_ML_API_KEY")

# Validate credentials
if not scoring_uri or not api_key:
    print("❌ Missing Azure ML endpoint credentials in .env file")
    print("   Required: AZURL_ML_ENDPOINT and AZURE_ML_API_KEY")
    exit(1)

print(f"\n✅ Using endpoint from .env")
print(f"   Scoring URI: {scoring_uri}")
endpoint_name = "fraud-detection-endpoint"  # For display purposes

# Load test data
print("\n📊 Loading test dataset...")
test_data_path = "ML/data/test_dataset_15_samples.csv"

try:
    df_test = pd.read_csv(test_data_path)
except Exception as e:
    print(f"❌ Could not load test data: {e}")
    exit(1)

print(f"✅ Loaded dataset with shape: {df_test.shape}")
print(f"   Fraud cases: {df_test['FraudFound_P'].sum()} of {len(df_test)}")

# Feature list
features = [
    'DriverRating', 'Age', 'PoliceReportFiled', 'WeekOfMonthClaimed',
    'PolicyType', 'WeekOfMonth', 'AccidentArea', 'Sex', 'Deductible'
]

X_test = df_test[features].copy()
y_test = df_test['FraudFound_P']

# Setup request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

print("\n" + "="*70)
print("                 RUNNING 15 SINGLE-PREDICTION CALLS")
print("="*70)

correct_predictions = 0
predictions_list = []

# Helper — retry wrapper
def call_with_retry(url, data, headers, retries=3, backoff=1.5):
    for attempt in range(1, retries + 1):
        try:
            return requests.post(url, data=data, headers=headers, timeout=15)
        except requests.exceptions.RequestException:
            if attempt == retries:
                return None
            time.sleep(backoff * attempt)

# MAIN LOOP
for i in range(len(df_test)):
    sample = X_test.iloc[i].to_dict()
    request_body = json.dumps(sample)

    response = call_with_retry(scoring_uri, request_body, headers)

    if not response:
        print(f"\n❌ Sample {i+1}: Request failed after retries.")
        continue

    if response.status_code != 200:
        print(f"\n❌ Sample {i+1}: Endpoint Error {response.status_code}")
        print(f"   Response: {response.text}")
        continue

    # Parse the response safely
    try:
        result = response.json()
        
        # Handle double-encoded JSON (string containing JSON)
        if isinstance(result, str):
            result = json.loads(result)

        if "predictions" in result:
            pred = result["predictions"][0]
            predicted = pred["fraud_prediction"]
            probability = pred["fraud_probability"]
            risk = pred.get("fraud_risk", "N/A")
            threshold = pred.get("threshold_used", "N/A")
        else:
            # fallback
            predicted = result.get("fraud_prediction", 0)
            probability = result.get("fraud_probability", 0.0)
            risk = result.get("fraud_risk", "N/A")
            threshold = result.get("threshold_used", "N/A")

    except Exception as e:
        print(f"\n❌ Sample {i+1}: Failed to parse JSON: {e}")
        print(f"   Response: {response.text}")
        continue

    actual = y_test.iloc[i]
    is_correct = (predicted == actual)
    if is_correct:
        correct_predictions += 1

    predictions_list.append({
        "sample": i+1,
        "actual": actual,
        "predicted": predicted,
        "probability": probability,
        "correct": is_correct
    })

    # Pretty output
    match_symbol = "✅" if is_correct else "❌"
    actual_label = "FRAUD" if actual == 1 else "NOT FRAUD"
    pred_label = "FRAUD" if predicted == 1 else "NOT FRAUD"

    print(f"\nSample {i+1:02d}: {match_symbol}")
    print(f"  Actual:      {actual_label}")
    print(f"  Predicted:   {pred_label}")
    print(f"  Probability: {probability:.4f}")
    print(f"  Risk Level:  {risk}")
    if i == 0:
        print(f"  Threshold:   {threshold}")

# METRICS
accuracy = correct_predictions / len(df_test)

tp = sum(1 for p in predictions_list if p["actual"] == 1 and p["predicted"] == 1)
fp = sum(1 for p in predictions_list if p["actual"] == 0 and p["predicted"] == 1)
tn = sum(1 for p in predictions_list if p["actual"] == 0 and p["predicted"] == 0)
fn = sum(1 for p in predictions_list if p["actual"] == 1 and p["predicted"] == 0)

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print("\n" + "="*70)
print("                 ENDPOINT PERFORMANCE SUMMARY")
print("="*70)
print(f"Endpoint Name:      {endpoint_name}")
print(f"Total Samples:      {len(df_test)}")
print(f"Correct Predictions:{correct_predictions}")
print(f"Accuracy:           {accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(f"  True Negatives:   {tn}")
print(f"  False Positives:  {fp}")
print(f"  False Negatives:  {fn}")
print(f"  True Positives:   {tp}")

print("\nMetrics:")
print(f"  Precision:        {precision*100:.2f}%")
print(f"  Recall:           {recall*100:.2f}%")
print(f"  F1 Score:         {f1:.4f}")

print("\n✅ Endpoint test completed successfully.")
"""
Test Azure ML endpoint with all 15 samples from test dataset
"""
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Get Azure ML credentials
endpoint = os.getenv("AZURE_ML_ENDPOINT")
api_key = os.getenv("AZURE_ML_API_KEY")

print("="*80)
print("TESTING AZURE ML ENDPOINT WITH 15 SAMPLES")
print("="*80)
print(f"\nEndpoint: {endpoint}")
print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT FOUND")

if not endpoint or not api_key:
    print("\n❌ ERROR: Missing Azure ML credentials")
    exit(1)

# Load test dataset
df = pd.read_csv('ML/data/test_dataset_15_samples.csv')
print(f"\n📊 Loaded {len(df)} samples from test dataset")

# Prepare headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

correct = 0
total = len(df)
results = []

print("\n" + "="*80)
print("TESTING EACH SAMPLE")
print("="*80)

for idx, row in df.iterrows():
    # Prepare input data - send categorical values as STRINGS, not encoded numbers
    # Azure ML's scoring.py will encode them using the label_encoders
    data = {
        "DriverRating": int(row["DriverRating"]),
        "Age": int(row["Age"]),
        "WeekOfMonthClaimed": int(row["WeekOfMonthClaimed"]),
        "WeekOfMonth": int(row["WeekOfMonth"]),
        "Deductible": int(row["Deductible"]),
        "AccidentArea": str(row["AccidentArea"]),  # Send as string: "Urban" or "Rural"
        "Sex": str(row["Sex"]),  # Send as string: "Male" or "Female"
        "PolicyType": str(row["PolicyType"]),  # Send as string: "Sedan - All Perils", etc.
        "PoliceReportFiled": str(row["PoliceReportFiled"])  # Send as string: "Yes" or "No"
    }
    
    actual = int(row["FraudFound_P"])
    
    try:
        # Call Azure ML endpoint
        response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = json.loads(response.json())
            prediction = result["predictions"][0]
            pred_fraud = prediction["fraud_prediction"]
            fraud_prob = prediction["fraud_probability"]
            threshold = prediction["threshold_used"]
            
            is_correct = pred_fraud == actual
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Sample {idx+1:2d}: Prob={fraud_prob:5.1%} | Pred={pred_fraud} | Actual={actual} | Threshold={threshold}")
            
            results.append({
                'sample': idx + 1,
                'probability': fraud_prob,
                'prediction': pred_fraud,
                'actual': actual,
                'correct': is_correct,
                'threshold': threshold
            })
        else:
            print(f"❌ Sample {idx+1:2d}: HTTP {response.status_code} - {response.text[:100]}")
            results.append({
                'sample': idx + 1,
                'error': f"HTTP {response.status_code}"
            })
    
    except Exception as e:
        print(f"❌ Sample {idx+1:2d}: Error - {str(e)}")
        results.append({
            'sample': idx + 1,
            'error': str(e)
        })

print("="*80)
print(f"\n📊 RESULTS SUMMARY")
print("="*80)
print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")

if results and 'threshold' in results[0]:
    threshold = results[0]['threshold']
    print(f"Threshold used: {threshold}")
    print(f"Fraud detected (≥{threshold}): {sum(1 for r in results if 'prediction' in r and r['prediction'] == 1)} samples")
    print(f"Not fraud (<{threshold}): {sum(1 for r in results if 'prediction' in r and r['prediction'] == 0)} samples")

print("\n" + "="*80)

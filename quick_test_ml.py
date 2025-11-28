"""
Quick test of Azure ML endpoint with 3 samples
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_ML_ENDPOINT")
api_key = os.getenv("AZURE_ML_API_KEY")

print("="*70)
print("QUICK TEST - NEW AZURE ML ENDPOINT")
print("="*70)
print(f"Endpoint: {endpoint}")
print(f"API Key: {api_key[:30]}..." if api_key else "NOT FOUND")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Test 3 samples
samples = [
    {"name": "Sample 2", "data": {"DriverRating": 2, "Age": 48, "PoliceReportFiled": 1, "WeekOfMonthClaimed": 1, "PolicyType": 1, "WeekOfMonth": 1, "AccidentArea": 1, "Sex": 1, "Deductible": 400}, "actual": 0},
    {"name": "Sample 4", "data": {"DriverRating": 4, "Age": 41, "PoliceReportFiled": 0, "WeekOfMonthClaimed": 4, "PolicyType": 1, "WeekOfMonth": 5, "AccidentArea": 1, "Sex": 1, "Deductible": 400}, "actual": 1},
    {"name": "Sample 9", "data": {"DriverRating": 1, "Age": 74, "PoliceReportFiled": 0, "WeekOfMonthClaimed": 2, "PolicyType": 1, "WeekOfMonth": 2, "AccidentArea": 0, "Sex": 1, "Deductible": 400}, "actual": 1}
]

print("\n" + "="*70)
correct = 0
for s in samples:
    try:
        response = requests.post(endpoint, headers=headers, json=s["data"], timeout=10)
        if response.status_code == 200:
            result = json.loads(response.json())
            pred = result["predictions"][0]
            prob = pred["fraud_probability"]
            fraud = pred["fraud_prediction"]
            threshold = pred["threshold_used"]
            
            is_correct = fraud == s["actual"]
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} {s['name']}: Prob={prob:.1%}, Pred={fraud}, Actual={s['actual']}, Threshold={threshold}")
        else:
            print(f"❌ {s['name']}: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ {s['name']}: Error - {str(e)}")

print("="*70)
print(f"Results: {correct}/3 correct")
print("="*70)

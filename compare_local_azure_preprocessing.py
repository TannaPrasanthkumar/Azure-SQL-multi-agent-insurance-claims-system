"""
Test what Azure ML actually receives and compare with local processing
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_API_KEY = os.getenv("AZURE_ML_API_KEY")

print("=" * 80)
print("AZURE ML vs LOCAL PREPROCESSING COMPARISON")
print("=" * 80)

# Sample 4 - should be FRAUD with 75.9% probability locally
sample_4 = {
    "DriverRating": 4,
    "Age": 41,
    "WeekOfMonthClaimed": 4,
    "WeekOfMonth": 5,
    "Deductible": 400,
    "AccidentArea": "Urban",
    "Sex": "Male",
    "PolicyType": "Utility - All Perils",
    "PoliceReportFiled": "No"
}

print("\nSample 4 (Expected FRAUD):")
print(json.dumps(sample_4, indent=2))

print("\n" + "=" * 80)
print("LOCAL PROCESSING RESULTS")
print("=" * 80)
print("""
After encoding:    [  4  41   0   4   6   5   1   1 400]
  DriverRating: 4
  Age: 41
  PoliceReportFiled: 0 (No)
  WeekOfMonthClaimed: 4
  PolicyType: 6 (Utility - All Perils)
  WeekOfMonth: 5
  AccidentArea: 1 (Urban)
  Sex: 1 (Male)
  Deductible: 400

After reordering:  [  4  41   4   5 400   1   1   6   0]
  1. DriverRating: 4
  2. Age: 41
  3. WeekOfMonthClaimed: 4
  4. WeekOfMonth: 5
  5. Deductible: 400
  6. AccidentArea: 1
  7. Sex: 1
  8. PolicyType: 6
  9. PoliceReportFiled: 0

After scaling:     [ 1.349  0.084  1.035  1.724 -0.177  0.345  0.429  3.978 -0.166]

Prediction: 75.9% probability → FRAUD ✅
""")

print("=" * 80)
print("AZURE ML PROCESSING")
print("=" * 80)

try:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_API_KEY}"
    }
    
    response = requests.post(
        AZURE_ML_ENDPOINT,
        headers=headers,
        data=json.dumps(sample_4),
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, str):
            result = json.loads(result)
        
        pred_data = result["predictions"][0]
        prob = pred_data["fraud_probability"]
        
        print(f"\nAzure ML Result:")
        print(f"  Probability: {prob * 100:.1f}%")
        print(f"  Prediction: {'FRAUD' if pred_data['fraud_prediction'] == 1 else 'NOT FRAUD'}")
        print(f"  Threshold: {pred_data['threshold_used']}")
        
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"\nLocal:    75.9% → FRAUD ✅")
        print(f"Azure ML: {prob * 100:.1f}% → {'FRAUD' if prob >= 0.65 else 'NOT FRAUD'} {'✅' if prob >= 0.65 else '❌'}")
        
        diff = abs(75.9 - prob * 100)
        print(f"\n⚠️  Probability difference: {diff:.1f}%")
        
        if diff > 5:
            print("\n🔍 SIGNIFICANT DIFFERENCE DETECTED!")
            print("\nPossible causes:")
            print("  1. ❌ Categorical encoding is different")
            print("  2. ❌ Feature ordering is wrong in Azure ML")
            print("  3. ❌ Scaler parameters don't match")
            print("  4. ❌ Wrong encoder classes being used")
            print("\n💡 SOLUTION NEEDED:")
            print("  → Check scoring.py encoding logic")
            print("  → Verify encoder classes match")
            print("  → Ensure feature ordering is correct")
            print("  → Verify scaler mean/scale values")
        else:
            print("\n✅ Probabilities are similar - preprocessing is correct!")
            
    else:
        print(f"❌ Error: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

print("\n" + "=" * 80)

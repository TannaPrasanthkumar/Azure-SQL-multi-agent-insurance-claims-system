"""
Test to debug feature ordering issue with Azure ML endpoint
"""
import os
import json
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()

AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_API_KEY = os.getenv("AZURE_ML_API_KEY")

print("=" * 80)
print("FEATURE ORDER DEBUG TEST")
print("=" * 80)
print(f"Endpoint: {AZURE_ML_ENDPOINT}")
print(f"API Key: {AZURE_ML_API_KEY[:20]}...\n")

# Expected feature order from model_metadata_0.pkl:
# ['DriverRating', 'Age', 'WeekOfMonthClaimed', 'WeekOfMonth', 'Deductible', 
#  'AccidentArea', 'Sex', 'PolicyType', 'PoliceReportFiled']

# Test with Sample 4 (should be fraud)
# From test dataset: DriverRating=4, Age=23, PoliceReportFiled=Yes, 
# WeekOfMonthClaimed=3, PolicyType=Sedan - All Perils, WeekOfMonth=4, 
# AccidentArea=Urban, Sex=Male, Deductible=500

test_sample = {
    "DriverRating": 4,
    "Age": 23,
    "WeekOfMonthClaimed": 3,
    "WeekOfMonth": 4,
    "Deductible": 500,
    "AccidentArea": "Urban",
    "Sex": "Male",
    "PolicyType": "Sedan - All Perils",
    "PoliceReportFiled": "Yes"
}

print("Test Sample (Sample 4 - should be FRAUD):")
print(json.dumps(test_sample, indent=2))
print("\nExpected: fraud_probability > 0.65 (FRAUD)")
print("Current result from Azure: 60.2% (NOT FRAUD) ❌")
print("\n" + "=" * 80)

try:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_API_KEY}"
    }
    
    response = requests.post(
        AZURE_ML_ENDPOINT,
        headers=headers,
        data=json.dumps(test_sample),
        timeout=60
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Raw response: {result}")
        
        # Handle double-encoded JSON
        if isinstance(result, str):
            result = json.loads(result)
        
        # Handle both formats
        if "predictions" in result:
            pred_data = result["predictions"][0]
        else:
            pred_data = result
        
        prob = pred_data.get("fraud_probability", 0)
        threshold = pred_data.get("threshold_used", 0.5)
        pred = pred_data.get("fraud_prediction", 0)
        
        print(f"\n✅ Response received:")
        print(f"  - Fraud Probability: {prob * 100:.1f}%")
        print(f"  - Threshold: {threshold}")
        print(f"  - Prediction: {'FRAUD' if pred == 1 else 'NOT FRAUD'}")
        print(f"  - Expected: FRAUD (probability should be > 65%)")
        
        if prob < 0.65:
            print(f"\n❌ STILL INCORRECT - Probability {prob * 100:.1f}% < 65%")
            print("\n🔍 POSSIBLE CAUSES:")
            print("  1. Azure ML endpoint still has OLD scoring.py")
            print("  2. Azure ML endpoint still has OLD model files")
            print("  3. Azure ML needs redeployment to use new files")
            print("  4. Feature order still not matching in deployed version")
        else:
            print(f"\n✅ CORRECT - Probability {prob * 100:.1f}% >= 65%")
            
    else:
        print(f"❌ Error: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("If still showing 60.2% (incorrect):")
print("  → Azure ML endpoint needs to be REDEPLOYED with new files")
print("  → Simply uploading files may not update the active deployment")
print("  → Check Azure ML Studio to redeploy or update the endpoint")
print("\nIf showing > 65% (correct):")
print("  → Feature order fix is working!")
print("  → Run full test suite to verify all samples")
print("=" * 80)

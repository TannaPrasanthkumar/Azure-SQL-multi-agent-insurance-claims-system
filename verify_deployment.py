"""
Verify if the new model is actually deployed by checking response details
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_API_KEY = os.getenv("AZURE_ML_API_KEY")

print("=" * 80)
print("DEPLOYMENT VERIFICATION TEST")
print("=" * 80)
print(f"Endpoint: {AZURE_ML_ENDPOINT}")
print(f"API Key: {AZURE_ML_API_KEY[:20]}...")

# Send a simple test to see if we get any deployment info
test_data = {
    "DriverRating": 1,
    "Age": 30,
    "WeekOfMonthClaimed": 1,
    "WeekOfMonth": 1,
    "Deductible": 500,
    "AccidentArea": "Urban",
    "Sex": "Male",
    "PolicyType": "Sedan - Liability",
    "PoliceReportFiled": "No"
}

try:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_API_KEY}"
    }
    
    response = requests.post(
        AZURE_ML_ENDPOINT,
        headers=headers,
        data=json.dumps(test_data),
        timeout=60
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse Headers:")
    for key, value in response.headers.items():
        if 'date' in key.lower() or 'version' in key.lower() or 'model' in key.lower():
            print(f"  {key}: {value}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Handle double-encoded JSON
        if isinstance(result, str):
            result = json.loads(result)
        
        print(f"\nFull Response:")
        print(json.dumps(result, indent=2))
        
        if "predictions" in result:
            pred = result["predictions"][0]
            print(f"\n✅ Model is responding")
            print(f"  - Threshold: {pred.get('threshold_used', 'NOT FOUND')}")
            print(f"  - Probability: {pred.get('fraud_probability', 0) * 100:.1f}%")
            
            threshold = pred.get('threshold_used', 0)
            if threshold == 0.65:
                print(f"\n✅ Threshold is correct (0.65)")
                print(f"\n⚠️  BUT probabilities are still wrong (66.7% accuracy)")
                print(f"\n🔍 POSSIBLE ISSUES:")
                print(f"  1. Wrong model files deployed (not the _0 versions)")
                print(f"  2. Scoring.py is old version (not reordering features)")
                print(f"  3. Model is correct but trained data was different")
                print(f"  4. Need to check if files in Azure match local files")
                print(f"\n💡 RECOMMENDATION:")
                print(f"  - Verify in Azure ML Studio that these files exist:")
                print(f"    ✓ balanced_random_forest_fraud_detector_0.pkl")
                print(f"    ✓ label_encoders_0.pkl")
                print(f"    ✓ scaler_0.pkl")
                print(f"    ✓ model_metadata_0.pkl")
                print(f"    ✓ scoring.py (with feature reordering code)")
                print(f"  - Check deployment logs in Azure ML Studio")
                print(f"  - Verify scoring.py is set as entry script")
            else:
                print(f"\n❌ Threshold is WRONG: {threshold} (should be 0.65)")
                print(f"  → Old model still deployed!")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

print("\n" + "=" * 80)

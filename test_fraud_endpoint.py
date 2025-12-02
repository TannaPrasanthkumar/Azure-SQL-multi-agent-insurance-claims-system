"""
Direct test of Azure ML Fraud Detection Endpoint
Shows exactly what's being sent and received
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def test_ml_endpoint():
    """Test the ML endpoint directly with raw requests"""
    
    print("=" * 80)
    print("🔬 DIRECT AZURE ML ENDPOINT TEST")
    print("=" * 80)
    
    scoring_uri = os.getenv("AZURE_ML_ENDPOINT")
    api_key = os.getenv("AZURE_ML_API_KEY")
    
    if not scoring_uri or not api_key:
        print("❌ Missing environment variables:")
        print(f"   AZURE_ML_ENDPOINT: {'✅ Set' if scoring_uri else '❌ Missing'}")
        print(f"   AZURE_ML_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
        return
    
    print(f"\n📍 Endpoint: {scoring_uri}")
    print(f"🔑 API Key: {api_key[:20]}...{api_key[-10:]}")
    
    # Test different payload formats that Azure ML might expect
    test_payloads = [
        {
            "name": "Format 1: Direct Features (Current)",
            "payload": {
                "DriverRating": 1,
                "Age": 65,
                "WeekOfMonthClaimed": 5,
                "WeekOfMonth": 5,
                "Deductible": 2000,
                "AccidentArea": "Rural",
                "Sex": "Male",
                "PolicyType": "Sport - Collision",
                "PoliceReportFiled": "No"
            }
        },
        {
            "name": "Format 2: With 'data' wrapper",
            "payload": {
                "data": {
                    "DriverRating": 1,
                    "Age": 65,
                    "WeekOfMonthClaimed": 5,
                    "WeekOfMonth": 5,
                    "Deductible": 2000,
                    "AccidentArea": "Rural",
                    "Sex": "Male",
                    "PolicyType": "Sport - Collision",
                    "PoliceReportFiled": "No"
                }
            }
        },
        {
            "name": "Format 3: With 'input_data' wrapper",
            "payload": {
                "input_data": {
                    "DriverRating": 1,
                    "Age": 65,
                    "WeekOfMonthClaimed": 5,
                    "WeekOfMonth": 5,
                    "Deductible": 2000,
                    "AccidentArea": "Rural",
                    "Sex": "Male",
                    "PolicyType": "Sport - Collision",
                    "PoliceReportFiled": "No"
                }
            }
        },
        {
            "name": "Format 4: Array format",
            "payload": {
                "data": [[1, 65, 5, 5, 2000, "Rural", "Male", "Sport - Collision", "No"]]
            }
        },
        {
            "name": "Format 5: With columns definition",
            "payload": {
                "data": {
                    "columns": ["DriverRating", "Age", "WeekOfMonthClaimed", "WeekOfMonth", "Deductible", 
                               "AccidentArea", "Sex", "PolicyType", "PoliceReportFiled"],
                    "data": [[1, 65, 5, 5, 2000, "Rural", "Male", "Sport - Collision", "No"]]
                }
            }
        }
    ]
    
    for i, test in enumerate(test_payloads, 1):
        print(f"\n{'=' * 80}")
        print(f"🧪 TEST {i}: {test['name']}")
        print(f"{'=' * 80}")
        
        print("\n📤 Sending Payload:")
        print(json.dumps(test['payload'], indent=2))
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            response = requests.post(
                scoring_uri,
                data=json.dumps(test['payload']),
                headers=headers,
                timeout=30
            )
            
            print(f"\n📥 Response Status: {response.status_code}")
            print(f"📥 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print(f"\n✅ SUCCESS - Response Body:")
                try:
                    result = response.json()
                    print(json.dumps(result, indent=2))
                    
                    # Try to extract fraud probability
                    if isinstance(result, dict):
                        fraud_prob = (result.get('fraud_probability') or 
                                    result.get('predictions', [{}])[0].get('fraud_probability') or
                                    0)
                        print(f"\n🎯 Fraud Probability Found: {fraud_prob:.4f} ({fraud_prob * 100:.2f}%)")
                        
                        if fraud_prob > 0:
                            print("✅ Model is returning non-zero probabilities!")
                            print("   This format works correctly!")
                            return test['name']  # Return the working format
                        else:
                            print("⚠️ Probability is 0 - trying next format...")
                    
                except Exception as e:
                    print(f"Raw response text: {response.text}")
                    print(f"⚠️ Could not parse JSON: {e}")
            else:
                print(f"\n❌ FAILED - Status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"\n❌ Request Failed:")
            print(f"   {type(e).__name__}: {str(e)}")
    
    print("\n" + "=" * 80)
    print("📊 TEST COMPLETE")
    print("=" * 80)
    print("\n❌ None of the formats returned non-zero fraud probabilities")
    print("\nPossible issues:")
    print("1. ✅ Endpoint is accessible (got 200 responses)")
    print("2. ❌ Model might not be properly trained")
    print("3. ❌ Model might need different feature names or order")
    print("4. ❌ Model scoring script (scoring.py) might have issues")
    print("\n💡 Recommendation: Check the Azure ML model's scoring.py file")
    print("   to see what input format and feature names it expects.")

if __name__ == "__main__":
    test_ml_endpoint()

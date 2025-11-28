"""
Quick test to check Azure ML endpoint access and see actual error messages
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

scoring_uri = os.getenv("AZURL_ML_ENDPOINT")
api_key = os.getenv("AZURE_ML_API_KEY")

print("Testing Azure ML Endpoint...")
print(f"Endpoint: {scoring_uri}")
print(f"API Key: {api_key[:20]}..." if api_key else "No API Key")

# Test data
test_data = {
    "DriverRating": 1,
    "Age": 35,
    "PoliceReportFiled": 1,
    "WeekOfMonthClaimed": 2,
    "PolicyType": 1,
    "WeekOfMonth": 2,
    "AccidentArea": 1,
    "Sex": 1,
    "Deductible": 500
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

print(f"\nSending request...")
print(f"Request body: {json.dumps(test_data, indent=2)}")

try:
    response = requests.post(
        scoring_uri,
        data=json.dumps(test_data),
        headers=headers,
        timeout=30
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Body:")
    print(response.text)
    
    if response.status_code == 200:
        print("\n✅ SUCCESS! Endpoint is accessible.")
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"\n❌ FAILED with status {response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"\n❌ Request Exception: {e}")
except Exception as e:
    print(f"\n❌ Error: {e}")

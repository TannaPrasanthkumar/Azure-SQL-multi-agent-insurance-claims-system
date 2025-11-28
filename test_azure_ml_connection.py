"""
Test Azure ML endpoint connection
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Get credentials
endpoint = os.getenv("AZURE_ML_ENDPOINT")
api_key = os.getenv("AZURE_ML_API_KEY")

print("="*70)
print("TESTING AZURE ML ENDPOINT CONNECTION")
print("="*70)
print(f"\nEndpoint: {endpoint}")
print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT FOUND")

if not endpoint or not api_key:
    print("\n❌ ERROR: Missing Azure ML credentials in .env file")
    exit(1)

# Test data
test_data = {
    "DriverRating": 2,
    "Age": 48,
    "PoliceReportFiled": 1,
    "WeekOfMonthClaimed": 1,
    "PolicyType": 1,
    "WeekOfMonth": 1,
    "AccidentArea": 1,
    "Sex": 1,
    "Deductible": 400
}

print(f"\n📊 Test Input:")
print(json.dumps(test_data, indent=2))

# Make request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

try:
    print(f"\n🔄 Sending request to Azure ML endpoint...")
    response = requests.post(endpoint, headers=headers, json=test_data, timeout=30)
    
    print(f"\n📊 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ SUCCESS!")
        print(f"\n📊 Response:")
        print(json.dumps(result, indent=2))
    elif response.status_code == 401:
        print(f"❌ AUTHENTICATION FAILED (401)")
        print(f"\nPossible issues:")
        print(f"  1. API key is expired or incorrect")
        print(f"  2. API key format is wrong (should not include 'Bearer')")
        print(f"  3. Endpoint requires different authentication")
        print(f"\n📄 Response: {response.text}")
    elif response.status_code == 404:
        print(f"❌ ENDPOINT NOT FOUND (404)")
        print(f"  The endpoint URL might be incorrect or the deployment was deleted")
        print(f"\n📄 Response: {response.text}")
    else:
        print(f"❌ ERROR: Status {response.status_code}")
        print(f"\n📄 Response: {response.text}")
        
except requests.exceptions.Timeout:
    print(f"❌ ERROR: Request timed out after 30 seconds")
except requests.exceptions.RequestException as e:
    print(f"❌ ERROR: {str(e)}")

print("\n" + "="*70)

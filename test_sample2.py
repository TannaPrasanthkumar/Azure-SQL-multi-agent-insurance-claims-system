"""
Test Azure ML endpoint with exact Sample 2 values
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Sample 2 exact values
data = {
    'DriverRating': 2,
    'Age': 48,
    'PoliceReportFiled': 1,
    'WeekOfMonthClaimed': 1,
    'PolicyType': 1,
    'WeekOfMonth': 1,
    'AccidentArea': 1,
    'Sex': 1,
    'Deductible': 400
}

print("Testing Azure ML endpoint with Sample 2 values:")
print(f"Input: {data}")

endpoint = os.getenv('AZURL_ML_ENDPOINT')
api_key = os.getenv('AZURE_ML_API_KEY')

response = requests.post(
    endpoint,
    data=json.dumps(data),
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    },
    timeout=30
)

print(f"\nStatus Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    if isinstance(result, str):
        result = json.loads(result)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if 'predictions' in result:
        pred = result['predictions'][0]
        print(f"\nFraud Probability: {pred['fraud_probability']:.4f}")
        print(f"Fraud Prediction: {pred['fraud_prediction']}")
        print(f"Risk Level: {pred['fraud_risk']}")
else:
    print(f"Error: {response.text}")

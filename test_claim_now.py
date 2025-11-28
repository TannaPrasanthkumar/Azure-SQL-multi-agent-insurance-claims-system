"""
Test specific claim against Azure ML fraud detection model
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_API_KEY = os.getenv("AZURE_ML_API_KEY")

print("=" * 80)
print("FRAUD DETECTION TEST")
print("=" * 80)

# Claim data (encoded)
claim_encoded = {
    "DriverRating": 1,
    "Age": 74,
    "PoliceReportFiled": 0,
    "WeekOfMonthClaimed": 2,
    "PolicyType": 1,
    "WeekOfMonth": 2,
    "AccidentArea": 2,
    "Sex": 1,
    "Deductible": 400
}

print("\nClaim Data (Encoded):")
for key, value in claim_encoded.items():
    print(f"  {key}: {value}")

# Convert to strings for Azure ML
accident_area_map = {0: "Rural", 1: "Urban", 2: "Rural"}  # Assuming 2 is also Rural
sex_map = {0: "Female", 1: "Male"}
policy_type_map = {
    0: "Sedan - All Perils",
    1: "Sedan - Collision", 
    2: "Sedan - Liability",
    3: "Sport - All Perils",
    4: "Sport - Collision",
    5: "Sport - Liability",
    6: "Utility - All Perils",
    7: "Utility - Collision",
    8: "Utility - Liability"
}
police_report_map = {0: "No", 1: "Yes"}

# Prepare data for Azure ML (with strings)
claim_data = {
    "DriverRating": claim_encoded["DriverRating"],
    "Age": claim_encoded["Age"],
    "WeekOfMonthClaimed": claim_encoded["WeekOfMonthClaimed"],
    "WeekOfMonth": claim_encoded["WeekOfMonth"],
    "Deductible": claim_encoded["Deductible"],
    "AccidentArea": accident_area_map.get(claim_encoded["AccidentArea"], "Rural"),
    "Sex": sex_map.get(claim_encoded["Sex"], "Male"),
    "PolicyType": policy_type_map.get(claim_encoded["PolicyType"], "Sedan - Collision"),
    "PoliceReportFiled": police_report_map.get(claim_encoded["PoliceReportFiled"], "No")
}

print("\n" + "=" * 80)
print("CONVERTED FOR AZURE ML (Categorical as Strings)")
print("=" * 80)
print(json.dumps(claim_data, indent=2))

print("\n" + "=" * 80)
print("CALLING AZURE ML ENDPOINT")
print("=" * 80)
print(f"Endpoint: {AZURE_ML_ENDPOINT}")

try:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_ML_API_KEY}"
    }
    
    response = requests.post(
        AZURE_ML_ENDPOINT,
        headers=headers,
        data=json.dumps(claim_data),
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, str):
            result = json.loads(result)
        
        prediction = result["predictions"][0]
        fraud_prob = prediction["fraud_probability"]
        fraud_pred = prediction["fraud_prediction"]
        threshold = prediction["threshold_used"]
        risk_level = prediction["fraud_risk"]
        
        print("\n" + "=" * 80)
        print("FRAUD DETECTION RESULT")
        print("=" * 80)
        print(f"\n📊 Fraud Probability: {fraud_prob * 100:.1f}%")
        print(f"🎯 Threshold: {threshold}")
        print(f"⚠️  Risk Level: {risk_level}")
        print(f"🔍 Prediction: {fraud_pred} ({'FRAUD' if fraud_pred == 1 else 'NOT FRAUD'})")
        
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        
        if fraud_pred == 1:
            print("\n🚨 FRAUD DETECTED!")
            print(f"   Probability {fraud_prob * 100:.1f}% exceeds threshold {threshold}")
            print("\n   Risk Factors:")
            if claim_encoded["DriverRating"] == 1:
                print("   ⚠️  Poor driver rating (1)")
            if claim_encoded["Age"] > 70:
                print("   ⚠️  Elderly driver (age 74)")
            if claim_encoded["PoliceReportFiled"] == 0:
                print("   ⚠️  No police report filed")
            if claim_encoded["AccidentArea"] == 2:
                print("   ℹ️  Rural area accident")
        else:
            print("\n✅ NO FRAUD DETECTED")
            print(f"   Probability {fraud_prob * 100:.1f}% is below threshold {threshold}")
            
            if fraud_prob > 0.5:
                print(f"\n   ⚠️  MEDIUM RISK - Close to threshold!")
                print("   Claim should be reviewed carefully")
            elif fraud_prob > 0.3:
                print(f"\n   ℹ️  LOW-MEDIUM RISK")
                print("   Standard review recommended")
            else:
                print(f"\n   ✅ LOW RISK")
                print("   Likely legitimate claim")
        
        print("\n" + "=" * 80)
        print("CLAIM CHARACTERISTICS")
        print("=" * 80)
        print(f"""
   Driver Rating: {claim_encoded['DriverRating']} (1=Poor, 4=Excellent)
   Age: {claim_encoded['Age']} years
   Police Report: {'Yes' if claim_encoded['PoliceReportFiled'] == 1 else 'No'}
   Week of Month Claimed: {claim_encoded['WeekOfMonthClaimed']}
   Week of Accident: {claim_encoded['WeekOfMonth']}
   Policy Type: {claim_data['PolicyType']}
   Accident Area: {claim_data['AccidentArea']}
   Gender: {claim_data['Sex']}
   Deductible: ${claim_encoded['Deductible']}
        """)
        
    else:
        print(f"\n❌ ERROR: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Request failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)

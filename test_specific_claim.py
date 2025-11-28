"""
Test specific claim data through fraud detector agent
"""
import sys
sys.path.append('c:/Projects/DEMO')

from fraud_detector_agent import FraudDetectorAgent

# Initialize agent
agent = FraudDetectorAgent()

# Test claim data
claim_data = {
    "DriverRating": 2,
    "Age": 27,
    "PoliceReportFiled": 0,
    "WeekOfMonthClaimed": 4,
    "PolicyType": 1,
    "WeekOfMonth": 4,
    "AccidentArea": 1,
    "Sex": 1,
    "Deductible": 400
}

print("=" * 80)
print("TESTING FRAUD DETECTION")
print("=" * 80)
print("\nClaim Data (encoded):")
for key, value in claim_data.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("CONVERTING TO STRINGS FOR AZURE ML")
print("=" * 80)

# Show what will be sent
accident_area_map = {0: "Rural", 1: "Urban"}
sex_map = {0: "Female", 1: "Male"}
policy_type_map = {
    0: "Sedan - All Perils",
    1: "Sedan - Collision", 
    2: "Sedan - Liability"
}
police_report_map = {0: "No", 1: "Yes"}

print("\nConverted values:")
print(f"  AccidentArea: {claim_data['AccidentArea']} → {accident_area_map.get(claim_data['AccidentArea'], 'Urban')}")
print(f"  Sex: {claim_data['Sex']} → {sex_map.get(claim_data['Sex'], 'Male')}")
print(f"  PolicyType: {claim_data['PolicyType']} → {policy_type_map.get(claim_data['PolicyType'], 'Sedan - Collision')}")
print(f"  PoliceReportFiled: {claim_data['PoliceReportFiled']} → {police_report_map.get(claim_data['PoliceReportFiled'], 'No')}")

print("\n" + "=" * 80)
print("CALLING AZURE ML")
print("=" * 80)

# Call fraud detection
result = agent.detect_fraud(claim_data)

print(f"\nResult:")
print(f"  Fraud Detected: {result.get('fraud_detected', False)}")
print(f"  Fraud Probability: {result.get('fraud_probability', 0) * 100:.1f}%")
print(f"  Risk Level: {result.get('fraud_risk', 'Unknown')}")
print(f"  Threshold: {result.get('threshold_used', 0.65)}")

if result.get('fraud_probability', 0) >= 0.65:
    print(f"\n🚨 FRAUD ALERT - Probability {result.get('fraud_probability', 0) * 100:.1f}% exceeds threshold!")
else:
    print(f"\n✅ NO FRAUD - Probability {result.get('fraud_probability', 0) * 100:.1f}% below threshold")

print("\n" + "=" * 80)

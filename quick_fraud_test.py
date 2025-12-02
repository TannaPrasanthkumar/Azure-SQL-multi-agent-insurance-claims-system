"""Quick test of fraud detection ML model"""
from fraud_detector_agent import FraudDetectorAgent
import json

print("="*70)
print("🧪 FRAUD DETECTION MODEL TEST")
print("="*70)

agent = FraudDetectorAgent()

# Test 1: Low Risk
print("\n✅ Test 1: Low Risk (Good driver, police report)")
result1 = agent.detect_fraud({
    'DriverRating': 4,
    'Age': 25,
    'PoliceReportFiled': 1,
    'WeekOfMonthClaimed': 2,
    'PolicyType': 2,
    'WeekOfMonth': 2,
    'AccidentArea': 1,
    'Sex': 1,
    'Deductible': 500
})
print(f"   Fraud: {result1['is_fraud']}")
print(f"   Probability: {result1['fraud_probability']:.2%}")
print(f"   Risk: {result1['fraud_risk']}")

# Test 2: High Risk
print("\n⚠️ Test 2: High Risk (Poor driver, no police report)")
result2 = agent.detect_fraud({
    'DriverRating': 1,
    'Age': 65,
    'PoliceReportFiled': 0,
    'WeekOfMonthClaimed': 5,
    'PolicyType': 4,
    'WeekOfMonth': 5,
    'AccidentArea': 0,
    'Sex': 1,
    'Deductible': 2000
})
print(f"   Fraud: {result2['is_fraud']}")
print(f"   Probability: {result2['fraud_probability']:.2%}")
print(f"   Risk: {result2['fraud_risk']}")

# Test 3: Medium Risk
print("\n⚡ Test 3: Medium Risk (Average factors)")
result3 = agent.detect_fraud({
    'DriverRating': 2,
    'Age': 45,
    'PoliceReportFiled': 0,
    'WeekOfMonthClaimed': 3,
    'PolicyType': 6,
    'WeekOfMonth': 3,
    'AccidentArea': 0,
    'Sex': 0,
    'Deductible': 1000
})
print(f"   Fraud: {result3['is_fraud']}")
print(f"   Probability: {result3['fraud_probability']:.2%}")
print(f"   Risk: {result3['fraud_risk']}")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED")
print("="*70)

from fraud_detector_agent import FraudDetectorAgent
import time

print("Testing ML Model with Updated Credentials...")
print("="*70)

agent = FraudDetectorAgent()

test_data = {
    'DriverRating': 1,
    'Age': 30,
    'PoliceReportFiled': 0,
    'WeekOfMonthClaimed': 1,
    'PolicyType': 1,
    'WeekOfMonth': 1,
    'AccidentArea': 1,
    'Sex': 1,
    'Deductible': 500
}

start = time.time()
result = agent.detect_fraud(test_data)
elapsed = time.time() - start

print(f"⏱️  Response Time: {elapsed:.3f}s")
print(f"✅ Success: {result['success']}")
print(f"🚨 Fraud Detected: {result.get('is_fraud', False)}")
print(f"📊 Fraud Probability: {result.get('fraud_probability', 0):.2%}")
print(f"⚠️  Risk Level: {result.get('fraud_risk', 'Unknown')}")
print(f"🤖 Status: {'ONLINE' if result['success'] else 'OFFLINE - ' + result.get('error', 'Unknown')}")

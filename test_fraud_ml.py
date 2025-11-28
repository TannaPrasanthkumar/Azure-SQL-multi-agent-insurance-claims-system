"""
Test script for Fraud Detection ML Model
Demonstrates training and prediction capabilities
"""

from fraud_ml_model import FraudMLModel
from fraud_detection_agent import FraudDetectionAgent
import json

print("="*80)
print("🧪 FRAUD DETECTION ML MODEL TEST")
print("="*80)

# Initialize ML model
print("\n1️⃣ Initializing ML Model...")
ml_model = FraudMLModel()

# Test scenarios
test_scenarios = [
    {
        "name": "Legitimate Small Claim",
        "claim": {
            'claim_info': {
                'claim_amount': 15000,
                'claim_date': '2025-06-15'
            }
        },
        "policy": {
            'validation': {
                'details': {
                    'policy_limit': 200000,
                    'past_claims_amount': 10000,
                    'claim_history_count': 1,
                    'policy_expiry_date': '2026-01-01'
                }
            }
        }
    },
    {
        "name": "Suspicious High-Limit Claim",
        "claim": {
            'claim_info': {
                'claim_amount': 95000,
                'claim_date': '2025-11-28'
            }
        },
        "policy": {
            'validation': {
                'details': {
                    'policy_limit': 100000,
                    'past_claims_amount': 40000,
                    'claim_history_count': 3,
                    'policy_expiry_date': '2025-12-05'
                }
            }
        }
    },
    {
        "name": "Round Amount Near Expiry",
        "claim": {
            'claim_info': {
                'claim_amount': 50000,
                'claim_date': '2025-11-20'
            }
        },
        "policy": {
            'validation': {
                'details': {
                    'policy_limit': 150000,
                    'past_claims_amount': 80000,
                    'claim_history_count': 2,
                    'policy_expiry_date': '2025-12-01'
                }
            }
        }
    },
    {
        "name": "Very High Value Claim",
        "claim": {
            'claim_info': {
                'claim_amount': 180000,
                'claim_date': '2025-11-15'
            }
        },
        "policy": {
            'validation': {
                'details': {
                    'policy_limit': 200000,
                    'past_claims_amount': 50000,
                    'claim_history_count': 4,
                    'policy_expiry_date': '2025-11-25'
                }
            }
        }
    }
]

print("\n2️⃣ Testing ML Predictions...\n")

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*80}")
    print(f"📋 Test Case {i}: {scenario['name']}")
    print(f"{'='*80}")
    
    # ML Model Prediction
    ml_result = ml_model.predict_fraud(scenario['claim'], scenario['policy'])
    
    print(f"\n🤖 ML Model Results:")
    print(f"   Fraud Probability: {ml_result['ml_fraud_probability']:.2%}")
    print(f"   Risk Score: {ml_result['ml_risk_score']}/100")
    print(f"   Risk Level: {ml_result['ml_risk_level']}")
    print(f"   Prediction: {ml_result['ml_prediction']}")
    print(f"   Confidence: {ml_result['ml_confidence']:.1f}%")
    print(f"   Model: {ml_result['model_used']}")
    
    if ml_result.get('feature_importance'):
        print(f"\n   🔍 Top 3 Feature Importance:")
        sorted_features = sorted(
            ml_result['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for feature, importance in sorted_features:
            print(f"      • {feature}: {importance:.3f}")

# Test with Full Fraud Detection Agent (includes ML + Rules + AI)
print(f"\n\n{'='*80}")
print("3️⃣ Testing Full Fraud Detection Agent (ML + Rules + AI)")
print(f"{'='*80}\n")

fraud_agent = FraudDetectionAgent()

if fraud_agent.enabled and fraud_agent.ml_enabled:
    print("✅ Fraud Agent initialized with ML model\n")
    
    # Test with suspicious case
    test_claim = {
        'claim_info': {
            'policy_number': 'POL12345',
            'claim_amount': 98000,
            'reason_for_claim': 'Vehicle accident with multiple damages',
            'claim_date': '2025-11-25'
        }
    }
    
    test_policy = {
        'policy_info': {'policy_number': 'POL12345'},
        'validation': {
            'details': {
                'policy_limit': 100000,
                'past_claims_amount': 60000,
                'claim_history_count': 3,
                'policy_status': 'Active',
                'policy_expiry_date': '2025-12-01',
                'policy_type': 'Vehicle'
            }
        }
    }
    
    full_result = fraud_agent.analyze_fraud_risk(test_claim, test_policy, None)
    
    print("🔍 Hybrid Fraud Detection Results:")
    print(f"   Risk Score: {full_result['fraud_risk_score']}/100")
    print(f"   Risk Level: {full_result['risk_level']}")
    print(f"   Detection Method: {full_result.get('detection_method', 'N/A')}")
    print(f"   ML Prediction: {full_result.get('ml_prediction', 'N/A')}")
    print(f"   ML Fraud Probability: {full_result.get('ml_fraud_probability', 0):.2%}")
    print(f"   Indicators Detected: {full_result['indicator_count']}")
    print(f"   Recommendation: {full_result['recommendation']}")
    print(f"   Requires Investigation: {'Yes' if full_result['requires_investigation'] else 'No'}")
    
    if full_result['fraud_indicators']:
        print(f"\n   📊 Fraud Indicators:")
        for ind in full_result['fraud_indicators']:
            print(f"      • [{ind['severity']}] {ind['indicator']}: {ind['description']}")
    
else:
    print("⚠️ Fraud Agent or ML Model not fully initialized")

print(f"\n{'='*80}")
print("✅ Testing Complete!")
print(f"{'='*80}\n")

# Summary
print("\n📝 Summary:")
print("   • ML model trained with synthetic data (1000 samples)")
print("   • 10 engineered features for fraud prediction")
print("   • Random Forest classifier with 100 trees")
print("   • Combines Rule-based + ML + AI for comprehensive detection")
print("   • Model saved to: models/fraud_model.pkl")
print("\n💡 Next Steps:")
print("   • Replace synthetic data with real historical fraud data")
print("   • Retrain model periodically with new fraud patterns")
print("   • Monitor false positive/negative rates")
print("   • Integrate into workflow_visualizer.py for live detection")

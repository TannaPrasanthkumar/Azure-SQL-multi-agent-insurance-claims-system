"""
Test script to verify Azure ML Fraud Detection Model is working correctly
Tests with various parameter combinations to validate model responses
"""

import json
from fraud_detector_agent import FraudDetectorAgent
import requests

def test_fraud_model():
    """Test fraud detection model with different scenarios"""
    
    print("=" * 80)
    print("🧪 FRAUD DETECTION MODEL TEST")
    print("=" * 80)
    
    # Initialize fraud detector
    try:
        fraud_agent = FraudDetectorAgent()
        print("✅ Fraud Detector Agent initialized successfully")
        print(f"   Endpoint: {fraud_agent.scoring_uri[:50]}...")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize Fraud Detector Agent: {e}")
        return
    
    # Test scenarios with different parameter combinations
    test_scenarios = [
        {
            "name": "Low Risk Scenario",
            "description": "Good driver, young age, urban area, police report filed",
            "data": {
                "DriverRating": 4,  # Excellent driver
                "Age": 25,
                "PoliceReportFiled": "Yes",
                "WeekOfMonthClaimed": 2,
                "PolicyType": "Sedan - Liability",
                "WeekOfMonth": 2,
                "AccidentArea": "Urban",
                "Sex": "Male",
                "Deductible": 500
            }
        },
        {
            "name": "Medium Risk Scenario",
            "description": "Average driver, middle age, rural area, no police report",
            "data": {
                "DriverRating": 2,
                "Age": 45,
                "PoliceReportFiled": "No",
                "WeekOfMonthClaimed": 3,
                "PolicyType": "Utility - All Perils",
                "WeekOfMonth": 3,
                "AccidentArea": "Rural",
                "Sex": "Female",
                "Deductible": 1000
            }
        },
        {
            "name": "High Risk Scenario",
            "description": "Poor driver, older age, multiple suspicious factors",
            "data": {
                "DriverRating": 1,  # Poor driver
                "Age": 65,
                "PoliceReportFiled": "No",
                "WeekOfMonthClaimed": 5,  # End of month
                "PolicyType": "Sport - Collision",
                "WeekOfMonth": 5,
                "AccidentArea": "Rural",
                "Sex": "Male",
                "Deductible": 2000
            }
        },
        {
            "name": "Default Values Test",
            "description": "Testing with default/minimum values",
            "data": {
                "DriverRating": 1,
                "Age": 30,
                "PoliceReportFiled": 0,
                "WeekOfMonthClaimed": 1,
                "PolicyType": 1,
                "WeekOfMonth": 1,
                "AccidentArea": 1,
                "Sex": 1,
                "Deductible": 500
            }
        },
        {
            "name": "Mixed Types Test",
            "description": "Testing with mixed string and numeric types",
            "data": {
                "DriverRating": 3,
                "Age": 35,
                "PoliceReportFiled": "Yes",
                "WeekOfMonthClaimed": 4,
                "PolicyType": "Sedan - All Perils",
                "WeekOfMonth": 2,
                "AccidentArea": "Urban",
                "Sex": "Female",
                "Deductible": 750
            }
        }
    ]
    
    # Run tests
    results_summary = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'=' * 80}")
        print(f"🔬 TEST {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'=' * 80}")
        
        print("\n📊 Input Parameters:")
        print(json.dumps(scenario['data'], indent=2))
        
        try:
            # Call fraud detection
            result = fraud_agent.detect_fraud(scenario['data'])
            
            print("\n📈 Fraud Detection Result:")
            print(json.dumps(result, indent=2))
            
            # Extract key metrics
            if result.get('success'):
                fraud_prob = result.get('fraud_probability', 0)
                fraud_risk = result.get('fraud_risk', 'Unknown')
                is_fraud = result.get('is_fraud', False)
                threshold = result.get('threshold_used', 0.5)
                
                print("\n✅ Model Response Summary:")
                print(f"   Fraud Probability: {fraud_prob:.4f} ({fraud_prob * 100:.2f}%)")
                print(f"   Risk Level: {fraud_risk}")
                print(f"   Is Fraud: {'YES ⚠️' if is_fraud else 'NO ✅'}")
                print(f"   Threshold Used: {threshold}")
                
                results_summary.append({
                    "scenario": scenario['name'],
                    "success": True,
                    "fraud_prob": fraud_prob,
                    "risk": fraud_risk,
                    "is_fraud": is_fraud
                })
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"\n❌ Model Failed:")
                print(f"   Error: {error_msg}")
                
                results_summary.append({
                    "scenario": scenario['name'],
                    "success": False,
                    "error": error_msg
                })
                
        except Exception as e:
            print(f"\n❌ Test Failed with Exception:")
            print(f"   {type(e).__name__}: {str(e)}")
            
            results_summary.append({
                "scenario": scenario['name'],
                "success": False,
                "error": str(e)
            })
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(1 for r in results_summary if r['success'])
    total_tests = len(results_summary)
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {(successful_tests / total_tests) * 100:.1f}%")
    
    print("\n📋 Results Table:")
    print(f"{'Scenario':<30} {'Status':<12} {'Fraud Prob':<15} {'Risk Level':<15} {'Is Fraud':<10}")
    print("-" * 95)
    
    for result in results_summary:
        scenario = result['scenario'][:28]
        if result['success']:
            status = "✅ SUCCESS"
            fraud_prob = f"{result['fraud_prob']:.4f}"
            risk = result['risk']
            is_fraud = "⚠️ YES" if result['is_fraud'] else "✅ NO"
        else:
            status = "❌ FAILED"
            fraud_prob = "N/A"
            risk = "N/A"
            is_fraud = "N/A"
        
        print(f"{scenario:<30} {status:<12} {fraud_prob:<15} {risk:<15} {is_fraud:<10}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    if successful_tests == 0:
        print("\n❌ ALL TESTS FAILED - CRITICAL ISSUES:")
        print("   1. Check if Azure ML endpoint is accessible")
        print("   2. Verify API key is correct in .env file")
        print("   3. Confirm the ML model is deployed and running")
        print("   4. Check network connectivity to Azure")
    elif successful_tests < total_tests:
        print("\n⚠️ SOME TESTS FAILED:")
        print("   1. Review failed scenarios above for specific errors")
        print("   2. Verify parameter formatting is correct")
        print("   3. Check if certain parameter combinations cause issues")
    else:
        print("\n✅ ALL TESTS PASSED!")
        print("   The fraud detection model is working correctly.")
        print("   Fraud probabilities are being calculated for all scenarios.")
        
        # Check if all probabilities are zero
        all_zero = all(r.get('fraud_prob', 1) == 0 for r in results_summary if r['success'])
        if all_zero:
            print("\n⚠️ WARNING: All fraud probabilities are 0.00%")
            print("   This might indicate:")
            print("   1. Model is returning default values")
            print("   2. Model needs retraining with proper data")
            print("   3. Feature encoding might be incorrect")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_fraud_model()

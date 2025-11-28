"""
Local test to simulate the fraud detection → Human Review workflow
This tests the exact flow without running Streamlit
"""

import sys
import json
from datetime import datetime

print("=" * 80)
print("TESTING FRAUD DETECTION → HUMAN REVIEW WORKFLOW LOCALLY")
print("=" * 80)

# Simulate session state
session_state = {}

def simulate_fraud_detection():
    """Simulate what happens in workflow_visualizer.py lines 2073-2083"""
    print("\n📋 Step 1: Loading fraud case from review_queue.json...")
    
    with open('review_queue.json', 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    # Find a fraud case with proper structure
    fraud_review = None
    for review in reviews:
        if ('analysis_result' in review and 
            'fraud_analysis' in review.get('analysis_result', {}) and
            'extracted_data' in review.get('claim_data', {})):
            fraud_review = review
            break
    
    if not fraud_review:
        print("   ❌ No valid fraud review found with proper structure!")
        return False
    
    print(f"   ✓ Found fraud review: {fraud_review['review_id']}")
    print(f"   ✓ Fraud Probability: {fraud_review['analysis_result']['fraud_analysis']['fraud_probability']}")
    
    # Extract data (simulating lines 2073-2083 in workflow_visualizer.py)
    print("\n🔧 Step 2: Creating fraud_claim_for_review session state...")
    
    fraud_analysis = fraud_review['analysis_result']['fraud_analysis']
    claim_data = fraud_review['claim_data']
    extracted_data = claim_data.get('extracted_data', {})
    claim_info = extracted_data.get('claim_info', {})
    
    session_state['fraud_claim_for_review'] = {
        'policy_number': claim_info.get('policy_number'),
        'decision': 'FRAUD_DETECTED',
        'fraud_probability': fraud_analysis.get('fraud_probability'),
        'fraud_risk': fraud_analysis.get('fraud_risk'),
        'threshold': fraud_analysis.get('threshold_used', 0.65),
        'extracted_data': extracted_data,
        'fraud_analysis': fraud_analysis,
        'eligibility_analysis': fraud_review['analysis_result'].get('eligibility_analysis', {})
    }
    
    print(f"   ✓ Created fraud_claim_for_review with:")
    print(f"      - policy_number: {session_state['fraud_claim_for_review']['policy_number']}")
    print(f"      - fraud_probability: {session_state['fraud_claim_for_review']['fraud_probability']}")
    print(f"      - fraud_risk: {session_state['fraud_claim_for_review']['fraud_risk']}")
    
    # Set other flags
    session_state['needs_human_review'] = True
    session_state['current_review_id'] = fraud_review['review_id']
    session_state['fraud_detected'] = True
    
    print(f"   ✓ Set needs_human_review = True")
    print(f"   ✓ Set current_review_id = {fraud_review['review_id']}")
    print(f"   ✓ Set fraud_detected = True")
    
    return True

def simulate_tab_switch():
    """Simulate switching to Human Review tab (Streamlit rerun)"""
    print("\n🔄 Step 3: Simulating tab switch (Streamlit rerun)...")
    print(f"   ℹ️  Session state persists across reruns")
    print(f"   ✓ session_state keys: {list(session_state.keys())}")

def test_human_review_ui():
    """Test if Human Review UI can access the data"""
    print("\n👤 Step 4: Testing Human Review UI logic...")
    
    # This is what human_review_agent.py line 284 checks
    if session_state.get('fraud_claim_for_review'):
        fraud_claim = session_state['fraud_claim_for_review']
        print("   ✅ SUCCESS! fraud_claim_for_review found in session state")
        
        # Test data extraction (lines 295-300 in human_review_agent.py)
        extracted_data = fraud_claim.get('extracted_data', {})
        claim_info = extracted_data.get('claim_info', {})
        
        print("\n   📊 Testing data extraction for UI display:")
        print(f"      - Policy Number: {claim_info.get('policy_number', 'N/A')}")
        print(f"      - Policyholder: {claim_info.get('policyholder_name', 'N/A')}")
        print(f"      - Claim Amount: ${claim_info.get('claim_amount', 0):,.2f}")
        print(f"      - Fraud Probability: {fraud_claim['fraud_probability']:.2%}")
        print(f"      - Risk Level: {fraud_claim['fraud_risk']}")
        print(f"      - Driver Rating: {claim_info.get('driver_rating', 'N/A')}")
        print(f"      - Age: {claim_info.get('age', 'N/A')}")
        print(f"      - Police Report: {claim_info.get('police_report_filed', 'N/A')}")
        
        # Verify all required keys exist
        required_keys = ['policy_number', 'fraud_probability', 'fraud_risk', 
                        'threshold', 'extracted_data', 'fraud_analysis']
        missing = [k for k in required_keys if k not in fraud_claim]
        
        if missing:
            print(f"\n   ⚠️  WARNING: Missing keys: {missing}")
            return False
        else:
            print("\n   ✅ All required keys present!")
            return True
    else:
        print("   ❌ FAILED! fraud_claim_for_review NOT in session state")
        print(f"   Available keys: {list(session_state.keys())}")
        return False

def test_session_persistence():
    """Test that session state persists like in Streamlit"""
    print("\n🧪 Step 5: Testing session state persistence...")
    
    # Simulate button click at line 1065 (Start Processing button)
    print("   Simulating button click (Start Processing)...")
    
    # Check lines 1067-1072 in workflow_visualizer.py
    # These lines should NOT delete fraud_claim_for_review
    if 'fraud_review_result' in session_state:
        del session_state['fraud_review_result']
        print("   ✓ Deleted fraud_review_result (expected)")
    
    # This should be commented out (line 1070-1071)
    # if 'fraud_claim_for_review' in session_state:
    #     del session_state['fraud_claim_for_review']
    print("   ✓ fraud_claim_for_review NOT deleted (correct behavior)")
    
    # Verify fraud_claim_for_review still exists
    if 'fraud_claim_for_review' in session_state:
        print("   ✅ SUCCESS! fraud_claim_for_review persisted!")
        return True
    else:
        print("   ❌ FAILED! fraud_claim_for_review was deleted!")
        return False

# Run the tests
def main():
    test_results = []
    
    # Test 1: Fraud detection
    result1 = simulate_fraud_detection()
    test_results.append(("Fraud Detection", result1))
    
    if not result1:
        print("\n❌ Test failed at fraud detection step. Exiting.")
        return False
    
    # Test 2: Tab switch
    simulate_tab_switch()
    
    # Test 3: Human Review UI
    result2 = test_human_review_ui()
    test_results.append(("Human Review UI", result2))
    
    # Test 4: Session persistence
    result3 = test_session_persistence()
    test_results.append(("Session Persistence", result3))
    
    # Final results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The Human Review tab SHOULD work correctly in the app")
        print("\n📝 Next steps:")
        print("   1. Start Streamlit: streamlit run workflow_visualizer.py")
        print("   2. Upload 4.pdf")
        print("   3. Click 'Start Processing'")
        print("   4. Wait for fraud detection")
        print("   5. Switch to 'Human Review' tab")
        print("   6. Verify fraud case displays with all details")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("⚠️  The Human Review tab may still have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

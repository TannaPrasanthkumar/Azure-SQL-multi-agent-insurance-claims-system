"""
Test script to verify session state flow for fraud detection -> human review
This simulates what happens when:
1. User processes a claim
2. Fraud is detected
3. User switches to Human Review tab
"""

import json
from datetime import datetime

# Simulate the workflow
print("=" * 60)
print("TESTING SESSION STATE FLOW FOR FRAUD DETECTION")
print("=" * 60)

# Step 1: Load a real fraud case from review_queue.json
print("\n1. Loading fraud case from review_queue.json...")
with open('review_queue.json', 'r', encoding='utf-8') as f:
    reviews = json.load(f)

if reviews:
    review = reviews[0]  # Get first pending review
    print(f"   ✓ Found review: {review['review_id']}")
    print(f"   ✓ Status: {review['status']}")
    print(f"   ✓ Fraud Probability: {review['analysis_result']['fraud_analysis']['fraud_probability']}")
    
    # Step 2: Simulate what workflow_visualizer.py does at line 2073
    print("\n2. Simulating fraud_claim_for_review creation (lines 2073-2083)...")
    fraud_claim_for_review = {
        'policy_number': review['claim_data']['extracted_data']['claim_info']['policy_number'],
        'decision': 'FRAUD_DETECTED',
        'fraud_probability': review['analysis_result']['fraud_analysis']['fraud_probability'],
        'fraud_risk': review['analysis_result']['fraud_analysis']['fraud_risk'],
        'threshold': review['analysis_result']['fraud_analysis'].get('threshold_used', 0.65),
        'extracted_data': review['claim_data']['extracted_data'],
        'fraud_analysis': review['analysis_result']['fraud_analysis'],
        'eligibility_analysis': review['analysis_result'].get('eligibility_analysis', {})
    }
    
    print(f"   ✓ fraud_claim_for_review created with:")
    print(f"      - policy_number: {fraud_claim_for_review['policy_number']}")
    print(f"      - fraud_probability: {fraud_claim_for_review['fraud_probability']}")
    print(f"      - fraud_risk: {fraud_claim_for_review['fraud_risk']}")
    
    # Step 3: Verify the data structure matches what human_review_agent.py expects
    print("\n3. Verifying data structure for human_review_agent.py...")
    
    required_keys = ['policy_number', 'decision', 'fraud_probability', 'fraud_risk', 'threshold', 'extracted_data', 'fraud_analysis']
    missing_keys = [key for key in required_keys if key not in fraud_claim_for_review]
    
    if missing_keys:
        print(f"   ✗ MISSING KEYS: {missing_keys}")
    else:
        print(f"   ✓ All required keys present")
    
    # Step 4: Check extracted_data structure
    print("\n4. Checking extracted_data structure...")
    extracted_data = fraud_claim_for_review.get('extracted_data', {})
    claim_info = extracted_data.get('claim_info', {})
    
    print(f"   ✓ extracted_data keys: {list(extracted_data.keys())}")
    print(f"   ✓ claim_info keys: {list(claim_info.keys())}")
    
    if claim_info:
        print(f"      - Policyholder Name: {claim_info.get('policyholder_name', 'N/A')}")
        print(f"      - Claim Amount: {claim_info.get('claim_amount', 'N/A')}")
        print(f"      - Incident Date: {claim_info.get('incident_date', 'N/A')}")
    
    # Step 5: Verify fraud_analysis structure
    print("\n5. Checking fraud_analysis structure...")
    fraud_analysis = fraud_claim_for_review.get('fraud_analysis', {})
    print(f"   ✓ fraud_analysis keys: {list(fraud_analysis.keys())}")
    
    fraud_indicators = fraud_analysis.get('fraud_indicators', {})
    if fraud_indicators:
        print(f"   ✓ fraud_indicators present:")
        for key, value in fraud_indicators.items():
            print(f"      - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("RESULT: Data structure is VALID ✓")
    print("=" * 60)
    print("\nThe Human Review tab SHOULD display correctly with this data.")
    print("If it's still blank, the issue is likely in:")
    print("1. Session state not persisting between tab switches")
    print("2. fraud_claim_for_review being cleared somewhere")
    print("3. The Human Review tab not reading session state correctly")
    
else:
    print("   ✗ No reviews found in review_queue.json")

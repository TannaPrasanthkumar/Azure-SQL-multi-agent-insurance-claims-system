"""
Complete Performance Testing for All Agents
Tests Document Intelligence, Human Review, and Audit Agent
"""

import asyncio
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Test 1: Document Intelligence Performance
def test_document_intelligence():
    print("\n" + "="*70)
    print("TEST 1: Document Intelligence Agent (Azure OCR)")
    print("="*70)
    
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not endpoint or not key:
        print("❌ Document Intelligence credentials not found")
        return None
    
    # Use first test PDF
    test_pdf = "data/1.pdf"
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return None
    
    start_time = time.time()
    
    try:
        client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
        
        with open(test_pdf, "rb") as f:
            poller = client.begin_analyze_document("prebuilt-document", f)
            result = poller.result()
        
        elapsed = time.time() - start_time
        
        # Extract data
        extracted_fields = 0
        if result.key_value_pairs:
            extracted_fields = len(result.key_value_pairs)
        
        text_length = len(result.content) if result.content else 0
        
        print(f"✅ Document Intelligence completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"📄 Pages Processed: {len(result.pages) if result.pages else 0}")
        print(f"🔑 Key-Value Pairs: {extracted_fields}")
        print(f"📝 Text Length: {text_length} characters")
        print(f"🧠 Model: prebuilt-document")
        print(f"📊 Status: ONLINE")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


# Test 2: Human Review Agent Performance
def test_human_review_agent():
    print("\n" + "="*70)
    print("TEST 2: Human Review Agent (Queue Management)")
    print("="*70)
    
    from human_review_agent import HumanReviewAgent
    
    start_time = time.time()
    
    try:
        agent = HumanReviewAgent(confidence_threshold=50.0)
        
        # Test flagging for review
        test_claim = {
            'extracted_data': {
                'claim_info': {
                    'policy_number': 'POL90927',
                    'claim_amount': 5000
                }
            },
            'ai_summary': 'Test claim for human review',
            'validation_result': {'success': True}
        }
        
        test_analysis = {
            'fraud_analysis': {
                'fraud_probability': 0.75,
                'fraud_risk': 'HIGH',
                'is_fraud': True
            },
            'eligibility_analysis': {
                'decision': 'ELIGIBLE',
                'confidence': 85
            },
            'confidence_score': 25
        }
        
        # Flag for review
        review_record = agent.flag_for_review(
            claim_data=test_claim,
            analysis_result=test_analysis,
            reason="High fraud probability detected"
        )
        
        # Get pending reviews
        pending = agent.get_pending_reviews()
        
        elapsed = time.time() - start_time
        
        print(f"✅ Human Review Agent completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"📋 Review Record Created: {review_record['review_id']}")
        print(f"📊 Pending Reviews: {len(pending)}")
        print(f"⚠️  Flagging Reason: {review_record['reason']}")
        print(f"📁 Queue File: review_queue.json")
        print(f"🔄 Status: ONLINE")
        
        # Cleanup - remove test review
        if os.path.exists('review_queue.json'):
            with open('review_queue.json', 'r') as f:
                queue = json.load(f)
            queue = [r for r in queue if r['review_id'] != review_record['review_id']]
            with open('review_queue.json', 'w') as f:
                json.dump(queue, f, indent=2)
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


# Test 3: Audit Agent Performance
def test_audit_agent():
    print("\n" + "="*70)
    print("TEST 3: Audit Agent (Logging & Storage)")
    print("="*70)
    
    from audit_agent import get_audit_agent
    
    start_time = time.time()
    
    try:
        agent = get_audit_agent()
        
        # Test logging document extraction
        log1_time = time.time()
        agent.log_document_agent_action(
            policy_number="POL90927",
            action="ocr_extraction",
            inputs={"file": "test.pdf"},
            outputs={"extracted_fields": 10},
            metadata={"test": True}
        )
        log1_elapsed = time.time() - log1_time
        
        # Test logging fraud detection
        log2_time = time.time()
        agent.log_fraud_detection_action(
            policy_number="POL90927",
            action="fraud_detection_ml",
            inputs={"DriverRating": 1, "Age": 30},
            outputs={"fraud_probability": 0.25},
            fraud_probability=0.25,
            fraud_prediction=0,
            fraud_risk_level="LOW",
            metadata={"test": True}
        )
        log2_elapsed = time.time() - log2_time
        
        # Test logging eligibility analysis
        log3_time = time.time()
        agent.log_eligibility_agent_action(
            policy_number="POL90927",
            action="eligibility_check",
            inputs={"claim_amount": 5000},
            outputs={"decision": "ELIGIBLE"},
            eligibility_decision="ELIGIBLE",
            confidence_score=85,
            metadata={"test": True}
        )
        log3_elapsed = time.time() - log3_time
        
        elapsed = time.time() - start_time
        
        print(f"✅ Audit Agent completed")
        print(f"⏱️  Total Response Time: {elapsed:.3f}s")
        print(f"   - Document Log: {log1_elapsed:.3f}s")
        print(f"   - Fraud Log: {log2_elapsed:.3f}s")
        print(f"   - Eligibility Log: {log3_elapsed:.3f}s")
        print(f"📊 Average Per Log: {elapsed/3:.3f}s")
        print(f"📁 Storage: Azure Blob Storage")
        print(f"📝 Log Format: JSON with timestamp")
        print(f"🔄 Status: ONLINE")
        
        # Get audit trail summary
        trail = agent.get_audit_trail(
            policy_number="POL90927",
            action_type="fraud_detection"
        )
        print(f"📋 Audit Records for POL90927: {len(trail)} entries")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all performance tests"""
    print("\n" + "🔍"*35)
    print("COMPLETE AGENT PERFORMANCE TESTING")
    print("Testing Document Intelligence, Human Review & Audit")
    print("🔍"*35)
    
    results = {}
    
    # Test each component
    results['document_intelligence'] = test_document_intelligence()
    results['human_review'] = test_human_review_agent()
    results['audit_agent'] = test_audit_agent()
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE PERFORMANCE SUMMARY")
    print("="*70)
    
    total_time = 0
    successful_tests = 0
    
    agent_names = {
        'document_intelligence': '📄 Document Intelligence (OCR)',
        'human_review': '👤 Human Review Agent',
        'audit_agent': '📝 Audit Agent'
    }
    
    for component, elapsed in results.items():
        name = agent_names.get(component, component)
        if elapsed is not None and elapsed > 0:
            print(f"✅ {name}: {elapsed:.3f}s")
            total_time += elapsed
            successful_tests += 1
        else:
            print(f"❌ {name}: FAILED")
    
    print(f"\n📊 Total Time: {total_time:.3f}s")
    print(f"✅ Successful Tests: {successful_tests}/{len(results)}")
    
    # Performance Ratings
    print("\n" + "="*70)
    print("PERFORMANCE RATINGS")
    print("="*70)
    
    if results.get('document_intelligence'):
        rating = "EXCELLENT" if results['document_intelligence'] < 3 else "GOOD" if results['document_intelligence'] < 5 else "ACCEPTABLE"
        print(f"📄 Document OCR: {results['document_intelligence']:.3f}s - {rating}")
    
    if results.get('human_review'):
        rating = "EXCELLENT" if results['human_review'] < 0.5 else "GOOD" if results['human_review'] < 1 else "ACCEPTABLE"
        print(f"👤 Human Review: {results['human_review']:.3f}s - {rating}")
    
    if results.get('audit_agent'):
        avg_log = results['audit_agent'] / 3
        rating = "EXCELLENT" if avg_log < 0.3 else "GOOD" if avg_log < 0.5 else "ACCEPTABLE"
        print(f"📝 Audit Logging: {avg_log:.3f}s/log - {rating}")
    
    print("\n" + "="*70)
    print("FULL WORKFLOW ESTIMATE")
    print("="*70)
    
    doc_time = results.get('document_intelligence') or 3.0
    hr_time = results.get('human_review') or 0.5
    audit_time = results.get('audit_agent') or 1.5
    
    print("🎯 Orchestrator: 0.5s")
    print(f"📄 Document Intelligence: {doc_time:.1f}s")
    print("🗄️  Azure SQL: 1.0s")
    print("🔍 Eligibility AI: 3.0s")
    print("🚨 Fraud ML: 1.3s")
    print(f"👤 Human Review (if needed): {hr_time:.1f}s")
    print("📧 Communication AI: 3.0s")
    print(f"📝 Audit Logging: {audit_time:.1f}s")
    print("="*70)
    
    estimated_total = (
        0.5 +  # Orchestrator
        doc_time +
        1.0 +  # SQL
        3.0 +  # Eligibility
        1.3 +  # Fraud
        3.0 +  # Communication
        audit_time
    )
    
    print(f"⚡ TOTAL (No Human Review): ~{estimated_total:.1f}s")
    print(f"⚡ TOTAL (With Human Review): ~{estimated_total + hr_time:.1f}s")
    print("="*70)


if __name__ == "__main__":
    main()

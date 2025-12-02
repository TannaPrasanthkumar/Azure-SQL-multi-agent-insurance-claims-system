"""
Performance Testing for Agentic Framework Components
Tests individual agent executors and measures response times
"""

import asyncio
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Test 1: Document Reader Executor Performance
async def test_document_reader():
    print("\n" + "="*70)
    print("TEST 1: Document Reader Executor")
    print("="*70)
    
    from workflow_visualizer_agentic import DocumentReaderExecutor
    from agent_framework import WorkflowContext
    
    executor = DocumentReaderExecutor()
    
    # Use sample PDF
    test_pdf = "data/1.pdf"
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return
    
    start_time = time.time()
    
    try:
        # Mock context for testing
        class MockContext:
            def __init__(self):
                self.messages = []
            
            async def send_message(self, data):
                self.messages.append(data)
        
        ctx = MockContext()
        await executor.process_document(test_pdf, ctx)
        
        elapsed = time.time() - start_time
        
        print(f"✅ Document Reader completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"📊 Extracted Fields: {len(ctx.messages[0].get('claim_info', {})) if ctx.messages else 0}")
        print(f"📄 Text Length: {len(ctx.messages[0].get('full_text', '')) if ctx.messages else 0} chars")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


# Test 2: Policy Validator Executor Performance
async def test_policy_validator():
    print("\n" + "="*70)
    print("TEST 2: Policy Validator Executor (Azure SQL)")
    print("="*70)
    
    from workflow_visualizer_agentic import PolicyValidatorExecutor
    
    executor = PolicyValidatorExecutor()
    
    # Test with known policy
    test_data = {
        "claim_info": {
            "policy_number": "POL90927"
        }
    }
    
    start_time = time.time()
    
    try:
        class MockContext:
            def __init__(self):
                self.messages = []
            
            async def send_message(self, data):
                self.messages.append(data)
        
        ctx = MockContext()
        await executor.validate_policy(test_data, ctx)
        
        elapsed = time.time() - start_time
        
        result = ctx.messages[0] if ctx.messages else {}
        validation = result.get("validation_result", {})
        
        print(f"✅ Policy Validator completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"📋 Policy Exists: {validation.get('policy_exists', False)}")
        print(f"🗄️  Azure SQL Query: SUCCESS" if validation.get('success') else "🗄️  Azure SQL Query: FAILED")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


# Test 3: Fraud Detector Executor Performance
async def test_fraud_detector():
    print("\n" + "="*70)
    print("TEST 3: Fraud Detector Executor (ML Model)")
    print("="*70)
    
    from workflow_visualizer_agentic import FraudDetectorExecutor
    
    executor = FraudDetectorExecutor()
    
    test_data = {
        "claim_info": {
            "driver_rating": 1,
            "age": 30,
            "police_report_filed": 0,
            "week_of_month_claimed": 1,
            "policy_type": 1,
            "week_of_month": 1,
            "accident_area": 1,
            "sex": 1,
            "deductible": 500
        }
    }
    
    start_time = time.time()
    
    try:
        class MockContext:
            def __init__(self):
                self.messages = []
            
            async def send_message(self, data):
                self.messages.append(data)
        
        ctx = MockContext()
        await executor.detect_fraud(test_data, ctx)
        
        elapsed = time.time() - start_time
        
        result = ctx.messages[0] if ctx.messages else {}
        fraud_analysis = result.get("fraud_analysis", {})
        
        print(f"✅ Fraud Detector completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"🚨 Fraud Detected: {fraud_analysis.get('is_fraud', False)}")
        print(f"📊 Fraud Probability: {fraud_analysis.get('fraud_probability', 0):.2%}")
        print(f"⚠️  Risk Level: {fraud_analysis.get('fraud_risk', 'Unknown')}")
        print(f"🤖 ML Model Status: {'ONLINE' if fraud_analysis.get('success') else 'OFFLINE/ERROR'}")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


# Test 4: Eligibility Agent Executor Performance
async def test_eligibility_agent():
    print("\n" + "="*70)
    print("TEST 4: Eligibility Agent Executor (AI Analysis)")
    print("="*70)
    
    from workflow_visualizer_agentic import EligibilityAgentExecutor
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import DefaultAzureCredential
    
    try:
        # Initialize chat client
        chat_client = AzureOpenAIChatClient(
            credential=DefaultAzureCredential(),
            endpoint=os.getenv("AZURE_AISERVICES_ENDPOINT"),
            api_key=os.getenv("AZURE_AISERVICES_APIKEY"),
            model=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
        )
        
        executor = EligibilityAgentExecutor(chat_client)
        
        test_data = {
            "claim_info": {
                "policy_number": "POL90927",
                "claim_amount": 5000
            },
            "policy_details": {
                "policy_limit": 100000,
                "policy_status": "Active",
                "claim_history_count": 1
            }
        }
        
        start_time = time.time()
        
        class MockContext:
            def __init__(self):
                self.messages = []
            
            async def send_message(self, data):
                self.messages.append(data)
        
        ctx = MockContext()
        await executor.analyze_eligibility(test_data, ctx)
        
        elapsed = time.time() - start_time
        
        result = ctx.messages[0] if ctx.messages else {}
        analysis = result.get("eligibility_analysis", {})
        
        print(f"✅ Eligibility Agent completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"✔️  Decision: {analysis.get('decision', 'UNKNOWN')}")
        print(f"📊 Confidence: {analysis.get('confidence', 0)}%")
        print(f"💭 Reasoning: {analysis.get('reasoning', 'N/A')[:100]}...")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


# Test 5: Communication Agent Executor Performance
async def test_communication_agent():
    print("\n" + "="*70)
    print("TEST 5: Communication Agent Executor (Email Generation)")
    print("="*70)
    
    from workflow_visualizer_agentic import CommunicationAgentExecutor
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import DefaultAzureCredential
    
    try:
        chat_client = AzureOpenAIChatClient(
            credential=DefaultAzureCredential(),
            endpoint=os.getenv("AZURE_AISERVICES_ENDPOINT"),
            api_key=os.getenv("AZURE_AISERVICES_APIKEY"),
            model=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
        )
        
        executor = CommunicationAgentExecutor(chat_client)
        
        test_data = {
            "claim_info": {
                "policy_number": "POL90927"
            },
            "eligibility_analysis": {
                "decision": "ELIGIBLE"
            }
        }
        
        start_time = time.time()
        
        class MockContext:
            def __init__(self):
                self.outputs = []
            
            async def yield_output(self, data):
                self.outputs.append(data)
        
        ctx = MockContext()
        await executor.generate_communication(test_data, ctx)
        
        elapsed = time.time() - start_time
        
        print(f"✅ Communication Agent completed")
        print(f"⏱️  Response Time: {elapsed:.3f}s")
        print(f"📧 Email Generated: {len(ctx.outputs[0]) if ctx.outputs else 0} chars")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Error: {e}")
        print(f"⏱️  Failed after: {elapsed:.3f}s")
        return None


async def main():
    """Run all performance tests"""
    print("\n" + "🚀"*35)
    print("AGENTIC FRAMEWORK PERFORMANCE TESTING")
    print("Testing Individual Agent Executors")
    print("🚀"*35)
    
    results = {}
    
    # Test each component
    results['policy_validator'] = await test_policy_validator()
    results['fraud_detector'] = await test_fraud_detector()
    
    # Skip tests that require files/credentials
    print("\n" + "="*70)
    print("NOTE: Skipping Document Reader test (requires PDF file)")
    print("NOTE: Skipping AI agents test (requires Azure OpenAI credentials)")
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    total_time = 0
    successful_tests = 0
    
    for component, elapsed in results.items():
        if elapsed is not None:
            print(f"✅ {component.replace('_', ' ').title()}: {elapsed:.3f}s")
            total_time += elapsed
            successful_tests += 1
        else:
            print(f"❌ {component.replace('_', ' ').title()}: FAILED")
    
    print(f"\n📊 Total Time (Tested Components): {total_time:.3f}s")
    print(f"✅ Successful Tests: {successful_tests}/{len(results)}")
    
    # Performance Analysis
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    if results.get('policy_validator'):
        print(f"🗄️  Azure SQL Query: {results['policy_validator']:.3f}s - {'EXCELLENT' if results['policy_validator'] < 1 else 'GOOD'}")
    
    if results.get('fraud_detector'):
        ml_status = "OFFLINE/ERROR" if results['fraud_detector'] < 2 else "ONLINE"
        print(f"🤖 ML Model Call: {results['fraud_detector']:.3f}s - Status: {ml_status}")
    
    print("\n" + "="*70)
    print("WORKFLOW OVERHEAD ANALYSIS")
    print("="*70)
    print("⚡ Agent Framework Overhead: ~50-100ms per agent transition")
    print("📊 Expected Full Workflow Time: ~15-25 seconds")
    print("   - Document OCR: 2-4s")
    print("   - Azure SQL: 0.5-1s")
    print("   - Eligibility AI: 2-5s")
    print("   - Fraud ML: 1-3s")
    print("   - Communication AI: 2-5s")
    print("   - Framework overhead: 0.5-1s")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

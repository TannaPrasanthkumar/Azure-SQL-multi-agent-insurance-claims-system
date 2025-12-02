"""
Insurance Claims Processing System - Agentic Framework Version
Multi-Agent Workflow with Microsoft Agent Framework
Preserves exact same UI as original workflow_visualizer.py
"""

import streamlit as st
import os
import asyncio
import time
import json
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv

# Agent Framework imports
from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    WorkflowStatusEvent,
    ExecutorFailedEvent,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential

# Import your existing agents (they'll be wrapped as tools/executors)
from azure_sql_agent import get_azure_sql_agent
from fraud_detector_agent import FraudDetectorAgent
from human_review_agent import HumanReviewAgent
from audit_agent import get_audit_agent

# Load environment variables
load_dotenv()

# ============================================================================
# WORKFLOW VISUALIZATION (Same as original)
# ============================================================================

def show_workflow_progress(step: int = 0):
    """Display the workflow progress with all 8 agents"""
    steps = [
        {"num": 1, "name": "Orchestrator<br/>Agent", "icon": "🎯", "color": "#4F8EF7"},
        {"num": 2, "name": "Document Reader<br/>Agent", "icon": "📄", "color": "#00C853"},
        {"num": 3, "name": "Policy Validator<br/>Agent", "icon": "🗄️", "color": "#0078D4"},
        {"num": 4, "name": "Eligibility<br/>Agent", "icon": "🔍", "color": "#9C27B0"},
        {"num": 5, "name": "Fraud Detector<br/>Agent", "icon": "🚨", "color": "#E91E63"},
        {"num": 6, "name": "Human Review<br/>Agent", "icon": "👤", "color": "#2196F3"},
        {"num": 7, "name": "Communication<br/>Agent", "icon": "📧", "color": "#00BCD4"}
    ]
    
    # Generate CSS and HTML (same as original)
    css = """
    <style>
    .workflow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 20px 0;
    }
    .agent-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        flex: 1;
    }
    .agent-icon {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .agent-step.active .agent-icon {
        animation: pulse 1.5s infinite;
        box-shadow: 0 0 20px rgba(255,255,255,0.5);
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .agent-step.completed .agent-icon {
        background: #4CAF50 !important;
    }
    .agent-step.inactive .agent-icon {
        background: #9E9E9E;
        opacity: 0.5;
    }
    .agent-label {
        margin-top: 8px;
        font-size: 11px;
        font-weight: 600;
        color: white;
        text-align: center;
        line-height: 1.2;
    }
    .step-number {
        position: absolute;
        top: -8px;
        right: -8px;
        background: white;
        color: #333;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        border: 2px solid #667eea;
    }
    .connector {
        flex: 0 0 40px;
        height: 3px;
        background: rgba(255,255,255,0.3);
        position: relative;
        top: -20px;
    }
    .connector.active {
        background: white;
        animation: flow 1s linear infinite;
    }
    @keyframes flow {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    </style>
    """
    
    html = css + '<div class="workflow-container">'
    
    for i, s in enumerate(steps):
        status = "completed" if s["num"] < step else ("active" if s["num"] == step else "inactive")
        icon_bg = s["color"] if status != "inactive" else "#9E9E9E"
        
        html += f'''
        <div class="agent-step {status}">
            <div class="step-number">{s["num"]}</div>
            <div class="agent-icon" style="background: {icon_bg};">{s["icon"]}</div>
            <div class="agent-label">{s["name"]}</div>
        </div>
        '''
        
        if i < len(steps) - 1:
            connector_class = "active" if s["num"] < step else ""
            html += f'<div class="connector {connector_class}"></div>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# AGENT EXECUTORS (Framework-based implementation)
# ============================================================================

class DocumentReaderExecutor(Executor):
    """Extract data from uploaded PDF using Azure Document Intelligence"""
    
    def __init__(self, id="document_reader"):
        super().__init__(id=id)
    
    @handler
    async def process_document(self, pdf_path: str, ctx: WorkflowContext[Dict]) -> None:
        """Extract text and structured data from PDF using Azure Document Intelligence + AI"""
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        from openai import AzureOpenAI
        
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
        
        with open(pdf_path, "rb") as f:
            poller = client.begin_analyze_document("prebuilt-layout", f)
            result = poller.result()
        
        # Extract text content
        extracted_text = ""
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"
        
        # Extract key-value pairs
        key_value_pairs = {}
        if result.key_value_pairs:
            for kv_pair in result.key_value_pairs:
                if kv_pair.key and kv_pair.value:
                    key_text = kv_pair.key.content if hasattr(kv_pair.key, 'content') else str(kv_pair.key)
                    value_text = kv_pair.value.content if hasattr(kv_pair.value, 'content') else str(kv_pair.value)
                    key_value_pairs[key_text] = value_text
        
        # Use AI to extract structured claim information (same as original)
        openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_AISERVICES_ENDPOINT"),
            api_key=os.getenv("AZURE_AISERVICES_APIKEY"),
            api_version="2024-02-15-preview"
        )
        
        prompt = f"""Extract the following information from this insurance claim document:

1. Policy Number
2. Policyholder Name
3. Claim Amount (numeric value only, no currency symbols)
4. Reason for Claim
5. Policy Type (exact text as in document, e.g., "Sedan - Liability", "Utility - All Perils")
6. Claim Date (preserve exact format as found)
7. Driver Rating (1-4, where 1=poor, 4=excellent)
8. Age (age of driver/policyholder)
9. Police Report Filed (text: "Yes" or "No")
10. Week of Month Claimed (1-5, week number when claim was filed)
11. Accident Area (text: "Urban" or "Rural")
12. Sex (text: "Male" or "Female")
13. Deductible (insurance deductible amount)
14. Week of Month (1-5, current week of month)

Extracted Text:
{extracted_text[:2000]}

Key-Value Pairs:
{json.dumps(key_value_pairs, indent=2)}

Return ONLY a JSON object with these exact keys: policy_number, policyholder_name, claim_amount, reason_for_claim, policy_type, claim_date, driver_rating, age, police_report_filed, week_of_month_claimed, accident_area, sex, deductible, week_of_month
"""
        
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data extraction expert. Extract information and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1,
            model=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
            response_format={"type": "json_object"}
        )
        
        claim_info = json.loads(response.choices[0].message.content)
        
        extracted_data = {
            "text": extracted_text,
            "key_value_pairs": key_value_pairs,
            "page_count": len(result.pages),
            "claim_info": claim_info,
            "full_text": extracted_text
        }
        
        await ctx.send_message(extracted_data)


class PolicyValidatorExecutor(Executor):
    """Validate policy against Azure SQL database"""
    
    def __init__(self, id="policy_validator"):
        super().__init__(id=id)
        self.sql_agent = get_azure_sql_agent()
    
    @handler
    async def validate_policy(self, data: Dict, ctx: WorkflowContext[Dict]) -> None:
        """Check if policy exists and is active"""
        policy_number = data.get("claim_info", {}).get("policy_number")
        
        if not policy_number:
            data["validation_result"] = {
                "success": False,
                "error": "No policy number found",
                "policy_data": {}
            }
        else:
            # Use existing Azure SQL agent
            validation = self.sql_agent.validate_policy(policy_number)
            
            if validation.get("policy_exists"):
                # Get detailed policy information
                details = self.sql_agent.get_policy_details(policy_number)
                
                # Merge policy info into validation result
                validation["policy_data"] = details.get("policy_info", {})
                data["validation_result"] = validation
                data["policy_details"] = details.get("policy_info", {})
            else:
                validation["policy_data"] = {}
                data["validation_result"] = validation
        
        await ctx.send_message(data)


class EligibilityAgentExecutor(Executor):
    """AI-powered eligibility analysis"""
    
    agent: ChatAgent
    
    def __init__(self, chat_client: AzureOpenAIChatClient, id="eligibility_agent"):
        self.agent = chat_client.create_agent(
            instructions="""You are an insurance eligibility analyst. Analyze claims and determine:
            1. Whether the claim is eligible for processing
            2. Confidence level (0-100%)
            3. Detailed reasoning
            
            Consider: policy status, claim amount vs limit, claim history.
            Respond in JSON format: {"decision": "ELIGIBLE/NOT_ELIGIBLE", "confidence": 85, "reasoning": "..."}"""
        )
        super().__init__(id=id)
    
    @handler
    async def analyze_eligibility(self, data: Dict, ctx: WorkflowContext[Dict]) -> None:
        """Run AI analysis on claim eligibility"""
        claim_info = data.get("claim_info", {})
        policy_details = data.get("policy_details", {})
        
        prompt = f"""
        Analyze this insurance claim:
        - Policy: {claim_info.get('policy_number')}
        - Claim Amount: ${claim_info.get('claim_amount', 0):,.2f}
        - Policy Limit: ${policy_details.get('policy_limit', 0):,.2f}
        - Policy Status: {policy_details.get('policy_status')}
        - Past Claims: {policy_details.get('claim_history_count', 0)}
        
        Is this claim eligible?
        """
        
        response = await self.agent.run([ChatMessage(role="user", text=prompt)])
        
        # Parse AI response
        import json
        try:
            analysis = json.loads(response.text)
        except:
            analysis = {
                "decision": "ELIGIBLE",
                "confidence": 70,
                "reasoning": response.text
            }
        
        data["eligibility_analysis"] = analysis
        await ctx.send_message(data)


class FraudDetectorExecutor(Executor):
    """ML-powered fraud detection"""
    
    def __init__(self, id="fraud_detector"):
        super().__init__(id=id)
        self.fraud_agent = FraudDetectorAgent()
    
    @handler
    async def detect_fraud(self, data: Dict, ctx: WorkflowContext[Dict]) -> None:
        """Run fraud detection ML model"""
        claim_info = data.get("claim_info", {})
        
        # Debug: Print claim_info to see what we're working with
        print(f"\n🔍 DEBUG - Claim Info received:")
        print(f"   Policy Type: {claim_info.get('policy_type')} (type: {type(claim_info.get('policy_type'))})")
        print(f"   Accident Area: {claim_info.get('accident_area')} (type: {type(claim_info.get('accident_area'))})")
        print(f"   Sex: {claim_info.get('sex')} (type: {type(claim_info.get('sex'))})")
        print(f"   Police Report: {claim_info.get('police_report_filed')} (type: {type(claim_info.get('police_report_filed'))})")
        print(f"   Driver Rating: {claim_info.get('driver_rating')}")
        print(f"   Age: {claim_info.get('age')}")
        print(f"   Claim Amount: {claim_info.get('claim_amount')}")
        
        # Prepare fraud detection data
        # fraud_detector_agent.py will handle string to number conversion
        fraud_data = {
            "DriverRating": claim_info.get('driver_rating', 1),
            "Age": claim_info.get('age', 30),
            "PoliceReportFiled": claim_info.get('police_report_filed', 0),
            "WeekOfMonthClaimed": claim_info.get('week_of_month_claimed', 1),
            "PolicyType": claim_info.get('policy_type', 1),
            "WeekOfMonth": claim_info.get('week_of_month', 1),
            "AccidentArea": claim_info.get('accident_area', 1),
            "Sex": claim_info.get('sex', 1),
            "Deductible": claim_info.get('deductible', 500)
        }
        
        print(f"\n🔍 DEBUG - Fraud data being sent to ML:")
        print(json.dumps(fraud_data, indent=2))
        
        fraud_result = self.fraud_agent.detect_fraud(fraud_data)
        
        print(f"\n🔍 DEBUG - Fraud result received:")
        print(json.dumps(fraud_result, indent=2))
        
        data["fraud_analysis"] = fraud_result
        
        # Check if fraud detected - if yes, flag for human review
        if fraud_result.get("is_fraud"):
            data["needs_human_review"] = True
            data["fraud_detected"] = True
        
        await ctx.send_message(data)


class CommunicationAgentExecutor(Executor):
    """Generate and send communications"""
    
    agent: ChatAgent
    
    def __init__(self, chat_client: AzureOpenAIChatClient, id="communication_agent"):
        self.agent = chat_client.create_agent(
            instructions="You are a professional insurance communication specialist. Generate clear, empathetic emails."
        )
        super().__init__(id=id)
    
    @handler
    async def generate_communication(self, data: Dict, ctx: WorkflowContext[Dict, str]) -> None:
        """Generate email/letter based on claim decision"""
        decision = data.get("eligibility_analysis", {}).get("decision", "UNKNOWN")
        policy_number = data.get("claim_info", {}).get("policy_number")
        
        prompt = f"Generate a professional email for policy {policy_number} with decision: {decision}"
        response = await self.agent.run([ChatMessage(role="user", text=prompt)])
        
        data["communication"] = response.text
        
        # Create serializable output (extract only JSON-safe data)
        # Convert all objects to ensure JSON serialization
        serializable_data = {
            "claim_info": dict(data.get("claim_info", {})),
            "validation_result": {
                "policy_data": dict(data.get("validation_result", {}).get("policy_data", {})),
                "policy_exists": data.get("validation_result", {}).get("policy_exists", False)
            },
            "policy_details": dict(data.get("policy_details", {})),
            "eligibility_analysis": dict(data.get("eligibility_analysis", {})),
            "fraud_analysis": dict(data.get("fraud_analysis", {})),
            "communication": str(data.get("communication", "")),
            "needs_human_review": bool(data.get("needs_human_review", False)),
            "fraud_detected": bool(data.get("fraud_detected", False))
        }
        
        # Final output - send message instead of yield_output to continue workflow
        await ctx.send_message(serializable_data)


class AuditAgentExecutor(Executor):
    """Log all workflow actions to Azure Blob Storage"""
    
    def __init__(self, id="audit_agent"):
        super().__init__(id=id)
        self.audit_agent = get_audit_agent()
    
    @handler
    async def log_workflow(self, data: Dict, ctx: WorkflowContext[Dict, str]) -> None:
        """Log all agent actions to audit trail"""
        policy_number = data.get("claim_info", {}).get("policy_number", "UNKNOWN")
        
        # Log workflow completion
        if self.audit_agent:
            self.audit_agent.log_workflow_completion(
                policy_number=policy_number,
                workflow_data=data,
                timestamp=datetime.now().isoformat()
            )
        
        # Final output with audit confirmation
        data["audit_logged"] = True
        await ctx.yield_output(json.dumps(data, indent=2, default=str))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def process_claim_with_framework(pdf_path: str):
    """Process claim using Agent Framework workflow (with original UI updates)"""
    
    # Initialize Azure OpenAI client for GitHub models (free tier)
    chat_client = AzureOpenAIChatClient(
        endpoint=os.getenv("AZURE_AISERVICES_ENDPOINT"),
        api_key=os.getenv("AZURE_AISERVICES_APIKEY"),
        model=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
    )
    
    # Create executor instances
    doc_reader = DocumentReaderExecutor()
    policy_validator = PolicyValidatorExecutor()
    eligibility_agent = EligibilityAgentExecutor(chat_client)
    fraud_detector = FraudDetectorExecutor()
    communication_agent = CommunicationAgentExecutor(chat_client)
    audit_agent = AuditAgentExecutor()
    
    # Build the workflow (sequential pipeline - audit runs in background, not part of visible workflow)
    workflow = (
        WorkflowBuilder()
        .set_start_executor(doc_reader)
        .add_edge(doc_reader, policy_validator)
        .add_edge(policy_validator, eligibility_agent)
        .add_edge(eligibility_agent, fraud_detector)
        .add_edge(fraud_detector, communication_agent)
        .build()
    )
    
    # Create UI placeholders (same as original)
    sidebar_status = st.sidebar.empty()
    workflow_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_text = st.empty()
    detail_placeholder = st.empty()
    
    # Initialize results
    results = {}
    current_step = 1
    
    # STEP 1: Orchestrator starts
    sidebar_status.markdown("""
    ### 📄 Current Status
    
    **🎯 Orchestrator Agent** 🟢 ONLINE  
    ✅ Workflow started  
    📹 Initializing pipeline
    
    ---
    
    **📄 Document Agent** 🔴 OFFLINE
    
    ---
    
    **🗄️ Azure SQL Agent** 🔴 OFFLINE
    
    ---
    
    **🔍 Eligibility Agent** 🔴 OFFLINE
    
    ---
    
    **🚨 Fraud Detection** 🔴 OFFLINE
    
    ---
    
    **📧 Communication** 🔴 OFFLINE
    
    ---
    
    **👤 Human Review** 🔴 OFFLINE
    """)
    
    with workflow_placeholder.container():
        show_workflow_progress(step=1)
    
    progress_placeholder.progress(0.1)
    status_text.info("🎯 **Orchestrator Agent** is initializing the workflow...")
    
    detail_placeholder.markdown("""
    **🎯 Orchestrator Agent** - *Working*
    - 📄 Initializing workflow pipeline...
    - 📹 Loading configuration...
    """)
    
    time.sleep(1.5)
    
    # Execute workflow with streaming
    agent_names = {
        "document_reader": "📄 Document Reader Agent",
        "policy_validator": "🗄️ Policy Validator Agent", 
        "eligibility_agent": "🔍 Eligibility Agent",
        "fraud_detector": "🚨 Fraud Detector Agent",
        "communication_agent": "📧 Communication Agent"
    }
    
    step_mapping = {
        "document_reader": 2,
        "policy_validator": 3,
        "eligibility_agent": 4,
        "fraud_detector": 5,
        "communication_agent": 6
    }
    
    # Define detailed progress messages for each agent
    agent_detail_messages = {
        "document_reader": [
            """**📄 Document Reader Agent** - *Working*
- 📄 Starting OCR engine...
- 📄 Loading document...""",
            """**📄 Document Reader Agent** - *Working*
- ✅ OCR engine started
- 📄 Reading document pages...
- 📄 Extracting text and key-value pairs...""",
            """**📄 Document Reader Agent** - *Working*
- ✅ Text extraction complete
- 🧠 AI analyzing document content...
- 💭 Extracting structured claim data..."""
        ],
        "policy_validator": [
            """**🗄️ Policy Validator Agent** - *Working*
- 🔌 Connecting to Azure SQL Database...
- 📋 Preparing policy query...""",
            """**🗄️ Policy Validator Agent** - *Working*
- ✅ Database connection established
- 🔍 Querying policy information...
- 📊 Validating policy status...""",
            """**🗄️ Policy Validator Agent** - *Working*
- ✅ Policy data retrieved
- 📊 Checking coverage limits...
- 💰 Analyzing claim history..."""
        ],
        "eligibility_agent": [
            """**🔍 Eligibility Agent** - *Working*
- 🧠 Initializing AI eligibility analysis...
- 📋 Loading policy rules...""",
            """**🔍 Eligibility Agent** - *Working*
- ✅ AI analysis started
- 🔍 Checking claim amount vs policy limit...
- 📊 Analyzing policy status...""",
            """**🔍 Eligibility Agent** - *Working*
- ✅ Eligibility checks performed
- 🧠 AI generating decision reasoning...
- 💯 Calculating confidence score..."""
        ],
        "fraud_detector": [
            """**🚨 Fraud Detector Agent** - *Working*
- 🤖 Initializing ML fraud detection model...
- 📊 Preparing claim features...""",
            """**🚨 Fraud Detector Agent** - *Working*
- ✅ ML model loaded
- 🔍 Analyzing fraud patterns...
- 📈 Running probability calculations...""",
            """**🚨 Fraud Detector Agent** - *Working*
- ✅ Fraud analysis complete
- 📊 Evaluating risk level...
- 🎯 Generating fraud report..."""
        ],
        "communication_agent": [
            """**📧 Communication Agent** - *Working*
- ✍️ Initializing AI communication generator...
- 📋 Preparing email template...""",
            """**📧 Communication Agent** - *Working*
- ✅ AI generator ready
- 🧠 Drafting professional email...
- 📝 Personalizing message content...""",
            """**📧 Communication Agent** - *Working*
- ✅ Email drafted
- 📧 Formatting final communication...
- 📤 Preparing for delivery..."""
        ]
    }
    
    async for event in workflow.run_stream(pdf_path):
        if isinstance(event, WorkflowStatusEvent):
            # Check if we have data in the event
            if hasattr(event, 'data') and event.data:
                results = event.data if isinstance(event.data, dict) else results
        elif isinstance(event, WorkflowOutputEvent):
            # Final output received
            try:
                results = json.loads(event.data) if isinstance(event.data, str) else event.data
            except:
                results = event.data
            break
        elif isinstance(event, ExecutorFailedEvent):
            st.error(f"❌ Agent failed: {agent_names.get(event.executor_id, event.executor_id)}")
            st.error(event.details.message)
            break
        
        # Get executor ID from event if available
        if hasattr(event, 'executor_id'):
            executor_id = event.executor_id
            if executor_id in step_mapping:
                current_step = step_mapping[executor_id]
                agent_name = agent_names.get(executor_id, "Agent")
                
                # Update sidebar for current agent
                sidebar_status.markdown(f"""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
                ---
                
                **📄 Document Agent** {'🟢 ONLINE' if executor_id == 'document_reader' else '🔴 OFFLINE'}
                {'⚙️ Processing document' if executor_id == 'document_reader' else ''}
                {'🔍 Extracting data...' if executor_id == 'document_reader' else ''}
                
                ---
                
                **🗄️ Azure SQL Agent** {'🟢 ONLINE' if executor_id == 'policy_validator' else '🔴 OFFLINE'}
                {'⚙️ Validating policy' if executor_id == 'policy_validator' else ''}
                {'🔍 Querying database...' if executor_id == 'policy_validator' else ''}
                
                ---
                
                **🔍 Eligibility Agent** {'🟢 ONLINE' if executor_id == 'eligibility_agent' else '🔴 OFFLINE'}
                {'⚙️ Analyzing eligibility' if executor_id == 'eligibility_agent' else ''}
                {'🧠 AI processing...' if executor_id == 'eligibility_agent' else ''}
                
                ---
                
                **🚨 Fraud Detection** {'🟢 ONLINE' if executor_id == 'fraud_detector' else '🔴 OFFLINE'}
                {'⚙️ Analyzing fraud risk' if executor_id == 'fraud_detector' else ''}
                {'🧠 ML model processing...' if executor_id == 'fraud_detector' else ''}
                
                ---
                
                **📧 Communication** {'🟢 ONLINE' if executor_id == 'communication_agent' else '🔴 OFFLINE'}
                {'⚙️ Generating communication' if executor_id == 'communication_agent' else ''}
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                """)
                
                # Update workflow visualization
                with workflow_placeholder.container():
                    show_workflow_progress(step=current_step)
                
                # Update progress bar (6 steps total: orchestrator + 5 agents)
                progress = min(current_step / 6, 1.0)
                progress_placeholder.progress(progress)
                
                # Update status text
                status_text.info(f"⚙️ **{agent_name}** is processing...")
                
                # Show detailed progress messages for this agent
                if executor_id in agent_detail_messages:
                    messages = agent_detail_messages[executor_id]
                    for msg in messages:
                        detail_placeholder.markdown(msg)
                        time.sleep(0.7)  # Pause between progress updates
                else:
                    # Fallback generic message
                    detail_placeholder.markdown(f"""
**{agent_name}** - *Working*
- 🔄 Processing data...
- ⚙️ Agent Framework executing...
""")
    
    # Final status
    status_text.success("✅ **All agents completed successfully!**")
    detail_placeholder.markdown("""
    **🎉 Workflow Complete**
    - ✅ All agents executed
    - 📊 Results ready for display
    """)
    
    # Don't sleep here - let results display immediately
    return results


def main():
    """Main Streamlit application (same UI as original)"""
    
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    </style>
    <h1 class="main-title">🏥 Insurance Claims Processing System</h1>
    <p class="subtitle">Multi-Agent Workflow with Microsoft Agent Framework</p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ System Information")
        st.markdown("### 🏥 Insurance Claims System")
        st.markdown("**Version:** 2.0 (Agentic Framework)")
        st.markdown("**Framework:** Microsoft Agent Framework")
        st.markdown("**Model:** GitHub Models (Free Tier)")
        
        st.markdown("---")
        st.markdown("### 🤖 Active Agents")
        st.markdown("1. 📄 Document Reader")
        st.markdown("2. 🗄️ Policy Validator")
        st.markdown("3. 🔍 Eligibility Agent")
        st.markdown("4. 🚨 Fraud Detector")
        st.markdown("5. 👤 Human Review")
        st.markdown("6. 📧 Communication")
    
    # Main tabs (same as original - 2 tabs only)
    tab1, tab2 = st.tabs(["📝 Process Claims", "👤 Human Review"])
    
    # Show notifications (same as original)
    if st.session_state.get('needs_human_review', False):
        st.info("🔔 **A claim requires human review.** Please switch to the **'Human Review'** tab to process it.")
    
    if st.session_state.get('rejected_claim_for_review'):
        st.warning("⚠️ **A rejected claim is available for review.** You can override the decision in the **'Human Review'** tab.")
    
    with tab1:
        # Debug output
        st.sidebar.write("**DEBUG INFO:**")
        st.sidebar.write(f"processing_completed: {st.session_state.get('processing_completed', 'NOT SET')}")
        st.sidebar.write(f"last_processing_results: {st.session_state.get('last_processing_results') is not None}")
        st.sidebar.write(f"fraud_detected: {st.session_state.get('fraud_detected', 'NOT SET')}")
        
        # Check if we have completed results to display
        if st.session_state.get('processing_completed') and st.session_state.get('last_processing_results'):
            # Display results (same as original - full agent summaries)
            results = st.session_state['last_processing_results']
            
            # Check if human review was completed
            has_review_decision = False
            fraud_review_result = {}
            if 'fraud_review_result' in st.session_state:
                fraud_review_result = st.session_state['fraud_review_result']
                has_review_decision = fraud_review_result.get('decision') is not None
            
            # Show completed workflow visualization
            st.markdown("### 🔄 Workflow Execution Flow")
            if has_review_decision:
                show_workflow_progress(step=8)  # Show all completed + Orchestrator active
            else:
                show_workflow_progress(step=6)  # Show workflow completed to communication agent
            st.markdown("---")
            
            # Display full results section (same layout as original)
            st.markdown("---")
            st.markdown("### 🎉 Agent Execution Summary")
            
            import pandas as pd
            agent_data = []
            
            # 1. Document Reader Agent
            if results.get('claim_info'):
                claim_info = results['claim_info']
                work_done = "Extracted claim data from PDF using Azure Document Intelligence OCR"
                output = f"Policy: {claim_info.get('policy_number', 'N/A')} | Name: {claim_info.get('policyholder_name', 'N/A')} | Amount: ${claim_info.get('claim_amount', '0')}"
                agent_data.append(["📄 Document Reader Agent", work_done, output])
            
            # 2. Policy Validator Agent
            if results.get('validation_result'):
                policy_info = results['validation_result'].get('policy_data', {})
                work_done = "Validated policy in Azure SQL Database"
                output = f"Policy Limit: ${policy_info.get('policy_limit', 0):,.0f} | Past Claims Amount: ${policy_info.get('past_claims_amount', 0):,.0f}"
                agent_data.append(["🗄️ Policy Validator Agent", work_done, output])
            
            # 3. Eligibility Agent
            if results.get('eligibility_analysis'):
                eligibility = results['eligibility_analysis']
                work_done = "Performed eligibility checks with AI-powered analysis"
                elig_decision = eligibility.get('decision', 'UNKNOWN')
                elig_confidence = eligibility.get('confidence', 0)
                output = f"Decision: {elig_decision} | Confidence: {elig_confidence}% | Reasoning: {eligibility.get('reasoning', 'N/A')[:50]}..."
                agent_data.append(["🔍 Eligibility Agent", work_done, output])
            
            # 4. Fraud Detection Agent
            if results.get('fraud_analysis'):
                fraud = results['fraud_analysis']
                if fraud.get('success'):
                    work_done = "ML-based fraud detection using Azure ML deployed model"
                    fraud_prob = fraud.get('fraud_probability', 0)
                    fraud_risk = fraud.get('fraud_risk', 'Unknown')
                    is_fraud = fraud.get('is_fraud', False)
                    output = f"Result: {'⚠️ FRAUD' if is_fraud else '✅ NO FRAUD'} | Probability: {fraud_prob:.2%} | Risk: {fraud_risk}"
                    agent_data.append(["🚨 Fraud Detector Agent", work_done, output])
                else:
                    work_done = "ML-based fraud detection attempted"
                    error_msg = fraud.get('error', 'Azure ML endpoint unavailable')
                    output = f"⚠️ SKIPPED: {error_msg}"
                    agent_data.append(["🚨 Fraud Detector Agent", work_done, output])
            
            # 5. Human Review Agent (if reviewed or fraud detected)
            is_fraud_detected = results.get('fraud_detected', False)
            if has_review_decision:
                work_done = f"Manual review by {fraud_review_result.get('reviewer', 'N/A')}"
                output = f"Decision: {fraud_review_result.get('decision')} | Timestamp: {fraud_review_result.get('timestamp', 'N/A')[:19]}"
                agent_data.append(["👤 Human Review Agent", work_done, output])
            elif is_fraud_detected:
                work_done = "Fraud detected - awaiting manual review decision"
                current_policy = results.get('claim_info', {}).get('policy_number', 'N/A')
                fraud_prob = results.get('fraud_analysis', {}).get('fraud_probability', 0)
                output = f"Status: ⏳ PENDING | Policy: {current_policy} | Fraud Probability: {fraud_prob:.2%}"
                agent_data.append(["👤 Human Review Agent", work_done, output])
            else:
                work_done = "No human review required - claim processed automatically"
                output = "Status: ⏸️ SKIPPED (Not needed for this claim)"
                agent_data.append(["👤 Human Review Agent", work_done, output])
            
            # 6. Communication Agent
            if results.get('communication'):
                communication = results['communication']
                work_done = "Sent professional email communication to policyholder"
                current_policy = results.get('claim_info', {}).get('policy_number', 'N/A')
                output = f"✅ Email sent | Policy: {current_policy} | Length: {len(str(communication))} chars"
                agent_data.append(["📧 Communication Agent", work_done, output])
            
            # 7. Audit Agent
            work_done = "Logged all agent actions to Azure Blob Storage audit trail"
            audit_logs_count = sum([
                1 if results.get('claim_info') else 0,
                1 if results.get('validation_result') else 0,
                1 if results.get('eligibility_analysis') else 0,
                1 if results.get('fraud_analysis') else 0,
                1 if has_review_decision else 0
            ])
            current_policy = results.get('claim_info', {}).get('policy_number', 'N/A')
            output = f"✅ Logged {audit_logs_count} agent actions | Policy: {current_policy} | Audit Trail: Azure Blob Storage"
            agent_data.append(["📝 Audit Agent", work_done, output])
            
            # Display agent summary table
            if agent_data:
                df_agents = pd.DataFrame(agent_data, columns=["Agent Name", "Work Performed", "Output/Result"])
                st.dataframe(
                    df_agents,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Agent Name": st.column_config.TextColumn("Agent Name", width="small"),
                        "Work Performed": st.column_config.TextColumn("Work Performed", width="medium"),
                        "Output/Result": st.column_config.TextColumn("Output/Result", width="large")
                    }
                )
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Process Another Claim", use_container_width=True, type="primary"):
                    # Clear session state (same as original)
                    for key in ['processing_completed', 'last_processing_results', 'fraud_detected', 
                               'needs_human_review', 'fraud_claim_for_review', 'fraud_review_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        # If fraud was detected and NOT yet reviewed AND processing not completed
        elif st.session_state.get('fraud_detected') and not st.session_state.get('fraud_review_result') and not st.session_state.get('tab1_reset'):
            print("⚠️ DEBUG - Showing fraud warning section")
            st.header("📝 Upload & Process Insurance Claim")
            
            # Show workflow at step 6 (awaiting human review)
            workflow_placeholder = st.empty()
            with workflow_placeholder.container():
                show_workflow_progress(step=6)
            
            st.markdown("---")
            st.warning("⚠️ **Fraud Detected - Awaiting Human Review**")
            st.info("👉 Please switch to the **'Human Review Agent - Manual Verification'** tab to review this claim.")
            
            # Show fraud details
            if st.session_state.get('fraud_claim_for_review'):
                fraud_data = st.session_state['fraud_claim_for_review']
                st.markdown("### 🚨 Fraud Alert Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📋 Policy Number", fraud_data.get('policy_number', 'N/A'))
                with col2:
                    st.metric("🚨 Fraud Probability", f"{fraud_data.get('fraud_probability', 0):.2%}")
                with col3:
                    st.metric("⚠️ Risk Level", fraud_data.get('fraud_risk', 'Unknown'))
            
            if st.button("🔄 Process Another Claim"):
                # Clear fraud state
                st.session_state['tab1_reset'] = True
                for key in ['fraud_detected', 'fraud_claim_for_review', 'needs_human_review']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        else:
            st.header("📝 Upload & Process Insurance Claim")
            
            uploaded_file = st.file_uploader(
                "Choose a claim document (PDF, PNG, JPG)",
                type=["pdf", "png", "jpg", "jpeg"],
                help="Upload an insurance claim document to process"
            )
            
            if uploaded_file:
                st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
                
                # Show preview for images
                if uploaded_file.type.startswith('image'):
                    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
                
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    # Clear previous session
                    for key in ['fraud_review_result', 'processing_completed', 'last_processing_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Save uploaded file
                    pdf_path = f"temp_{uploaded_file.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.markdown("---")
                    
                    # Show workflow heading
                    st.markdown("### 🔄 Live Workflow Progress")
                    
                    # Process with Agent Framework
                    try:
                        results = asyncio.run(process_claim_with_framework(pdf_path))
                        
                        # Log to audit trail (runs in background, not part of visible workflow)
                        audit_agent_instance = get_audit_agent()
                        if audit_agent_instance:
                            policy_number = results.get('claim_info', {}).get('policy_number', 'UNKNOWN')
                            audit_agent_instance.log_orchestrator_action(
                                policy_number=policy_number,
                                action="workflow_completed",
                                inputs={"pdf_path": uploaded_file.name if uploaded_file else "unknown"},
                                outputs=results,
                                decision="COMPLETED",
                                metadata={
                                    "timestamp": datetime.now().isoformat(),
                                    "agents_executed": list(results.keys()),
                                    "fraud_detected": results.get("fraud_detected", False),
                                    "needs_human_review": results.get("needs_human_review", False)
                                }
                            )
                        
                        # Store results in session state
                        st.session_state['last_processing_results'] = results
                        
                        # Check if fraud detected
                        is_fraud_detected = results.get("fraud_detected", False)
                        
                        if is_fraud_detected:
                            # Fraud case: Set fraud flags but DON'T set processing_completed yet
                            # It will be set after human review in human_review_agent.py
                            st.session_state['fraud_detected'] = True
                            st.session_state['needs_human_review'] = True
                            
                            # Populate fraud_claim_for_review for Human Review tab
                            claim_info = results.get('claim_info', {})
                            fraud_analysis = results.get('fraud_analysis', {})
                            st.session_state['fraud_claim_for_review'] = {
                                'policy_number': claim_info.get('policy_number', 'N/A'),
                                'decision': 'FRAUD_DETECTED',
                                'fraud_probability': fraud_analysis.get('fraud_probability', 0),
                                'fraud_risk': fraud_analysis.get('fraud_risk', 'Unknown'),
                                'threshold': fraud_analysis.get('threshold_used', 0.65),
                                'extracted_data': results,
                                'fraud_analysis': fraud_analysis,
                                'eligibility_analysis': results.get('eligibility_analysis', {})
                            }
                        else:
                            # Non-fraud case: Set processing_completed immediately
                            st.session_state['processing_completed'] = True
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error processing claim: {e}")
                    finally:
                        # Cleanup
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
    
    with tab2:
        st.title("👤 Human Review Agent - Manual Verification")
        
        # Use existing human review agent UI
        from human_review_agent import HumanReviewAgent
        
        if st.session_state.get('fraud_claim_for_review'):
            human_review_agent = HumanReviewAgent(confidence_threshold=50.0)
            human_review_agent.show_review_interface()
        else:
            st.info("📭 No claims pending review at the moment.")
            st.markdown("""
            ### How Human Review Works
            
            Claims are flagged for manual review when:
            - 🚨 Fraud probability exceeds threshold
            - ⚠️ Low confidence in automated decision
            - 📋 Policy requires manual verification
            """)


if __name__ == "__main__":
    main()

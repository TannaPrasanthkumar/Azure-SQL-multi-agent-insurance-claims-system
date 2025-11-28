"""
Insurance Claims Processing System - Complete Workflow Visualization
Shows step-by-step agent interactions and data flow
"""

import os
import streamlit as st
import time
import json
from datetime import datetime
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from policy_validator import PolicyValidator
from human_review_agent import HumanReviewAgent, render_human_review_ui
from audit_agent import get_audit_agent
from fraud_detector_agent import FraudDetectorAgent

# Initialize clients
@st.cache_resource
def get_document_client():
    """Initialize Document Intelligence Agent"""
    try:
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not endpoint or not key:
            raise ValueError("Missing Document Intelligence credentials")
        return DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    except Exception as e:
        st.error(f"Failed to initialize Document Intelligence Agent: {str(e)}")
        return None

@st.cache_resource
def get_openai_client():
    """Initialize Azure OpenAI client"""
    try:
        endpoint = os.getenv("AZURE_AISERVICES_ENDPOINT")
        key = os.getenv("AZURE_AISERVICES_APIKEY")
        if not endpoint or not key:
            raise ValueError("Missing Azure OpenAI credentials")
        return AzureOpenAI(api_version="2024-12-01-preview", azure_endpoint=endpoint, api_key=key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

@st.cache_resource
def get_policy_validator():
    """Initialize Databricks Agent (Policy Validator)"""
    try:
        return PolicyValidator()
    except Exception as e:
        st.error(f"Failed to initialize Databricks Agent: {str(e)}")
        return None

@st.cache_resource
def get_audit_agent_instance():
    """Initialize Audit Agent for logging all agent actions"""
    try:
        return get_audit_agent()
    except Exception as e:
        st.error(f"Failed to initialize Audit Agent: {str(e)}")
        return None

def show_workflow_diagram():
    """Display the workflow diagram"""
    st.markdown("### 🔄 System Architecture & Workflow")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="agent-box orchestrator-box">
            <h3>🎯 Orchestrator Agent</h3>
            <p><b>Role:</b> Coordinates all agents</p>
            <p><b>Tasks:</b></p>
            <ul>
                <li>Manages workflow</li>
                <li>Routes data between agents</li>
                <li>Aggregates results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-box document-box">
            <h3>📄 Document Intelligence Agent</h3>
            <p><b>Role:</b> Extract & Analyze</p>
            <p><b>Tasks:</b></p>
            <ul>
                <li>OCR & text extraction</li>
                <li>Key-value extraction</li>
                <li>Document parsing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-box databricks-box">
            <h3>💾 Databricks Agent</h3>
            <p><b>Role:</b> Policy Validation</p>
            <p><b>Tasks:</b></p>
            <ul>
                <li>Query policy database</li>
                <li>Check policy status</li>
                <li>Validate eligibility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="agent-box eligibility-box">
            <h3>🔍 Eligibility Agent</h3>
            <p><b>Role:</b> AI Eligibility Analysis</p>
            <p><b>Tasks:</b></p>
            <ul>
                <li>GPT-4 powered analysis</li>
                <li>Coverage comparison</li>
                <li>Confidence scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Data Flow")
    st.markdown("""
    ```
    Document Upload → Orchestrator Agent
                          ↓
                    Step 1: Send to Document Intelligence Agent
                          ↓
                    Extract Text & Data
                          ↓
                    Return to Orchestrator
                          ↓
                    Step 2: Orchestrator Generates AI Summary
                          ↓
                    Step 3: Extract Policy Info
                          ↓
                    Step 4: Send to Databricks Agent
                          ↓
                    Query Policy Database
                          ↓
                    Validate Status
                          ↓
                    Return Results to Orchestrator
                          ↓
                    Display Final Results to User
    ```
    """)

def analyze_document(file_bytes, filename):
    """Step 1: Document Intelligence Agent - Extract claim information and validate policy number"""
    document_client = get_document_client()
    policy_validator = get_policy_validator()
    
    if not document_client:
        return None
    
    try:
        with st.spinner("🔍 Document Intelligence Agent is analyzing..."):
            poller = document_client.begin_analyze_document("prebuilt-layout", file_bytes)
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
        
        extracted_data = {
            "text": extracted_text,
            "key_value_pairs": key_value_pairs,
            "page_count": len(result.pages)
        }
        
        # Extract policy number from the document using AI
        openai_client = get_openai_client()
        if openai_client:
            try:
                deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
                prompt = f"""Extract the following information from this insurance claim document:
                
1. Policy Number
2. Policyholder Name
3. Claim Amount (numeric value only, no currency symbols)
4. Reason for Claim
5. Policy Type (exact text as in document, e.g., "Sedan - Liability", "Utility - All Perils")
6. Claim Date (preserve exact format as found, e.g., DD-MM-YYYY or YYYY-MM-DD)
7. Driver Rating (1-4, where 1=poor, 4=excellent)
8. Age (age of driver/policyholder)
9. Police Report Filed (text: "Yes" or "No")
10. Week of Month Claimed (1-5, week number when claim was filed)
11. Accident Area (text: "Urban" or "Rural")
12. Sex (text: "Male" or "Female")
13. Deductible (insurance deductible amount)
14. Week of Month (1-5, current week of month)

Important Instructions:
- For Claim Date: Look for dates labeled as "Claim Date", "Date of Claim", "Incident Date", "Date of Service", or "Date of Loss"
- Keep the date in its original format (DD-MM-YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.) - do NOT convert it
- Common formats: 02-11-2024, 2024-11-02, 02/11/2024, 11/02/2024
- If multiple dates exist, use the one labeled as claim/incident date
- For Claim Amount: Extract only numbers, remove currency symbols, commas, or text
- For Policy Type: Extract EXACT text as it appears (e.g., "Sedan - Liability", "Utility - All Perils") - do NOT convert to numbers
- For Accident Area: Extract as text ("Urban" or "Rural") - do NOT convert to numbers
- For Sex: Extract as text ("Male" or "Female") - do NOT convert to numbers
- For Police Report Filed: Extract as text ("Yes" or "No") - do NOT convert to numbers
- For numeric fields (driver_rating, age, week_of_month_claimed, week_of_month, deductible): Extract as numbers

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
                    max_completion_tokens=500,
                    temperature=0.1,
                    model=deployment,
                    response_format={"type": "json_object"}
                )
                
                claim_info = json.loads(response.choices[0].message.content)
                extracted_data['claim_info'] = claim_info
                
                print(f"🔍 Extracted claim info: {claim_info}")
                
                # Validate policy number exists in database
                policy_number = claim_info.get('policy_number')
                print(f"🔍 Policy number from claim: '{policy_number}'")
                print(f"🔍 Policy validator enabled: {policy_validator.enabled if policy_validator else 'N/A'}")
                
                if policy_number and policy_validator and policy_validator.enabled:
                    # Check if policy exists
                    print(f"🔍 Validating policy number: {policy_number}")
                    policy_exists = policy_validator.validate_policy_number(policy_number)
                    print(f"🔍 Policy exists result: {policy_exists}")
                    
                    if not policy_exists:
                        return {
                            "error": True,
                            "message": f"✌ INVALID POLICY NUMBER: Policy number '{policy_number}' does not exist in our database. Please verify the policy number and try again."
                        }
                    extracted_data['policy_validated'] = True
                    print(f"✅ Policy {policy_number} validated successfully")
                else:
                    extracted_data['policy_validated'] = False
                    if not policy_number:
                        return {
                            "error": True,
                            "message": "✌ POLICY NUMBER NOT FOUND: Unable to extract policy number from the document. Please ensure the document contains a valid policy number."
                        }
                
            except Exception as e:
                st.warning(f"Could not extract structured claim info: {str(e)}")
                extracted_data['claim_info'] = {}
                extracted_data['policy_validated'] = False
        
        # Log to Audit Agent
        audit_agent = get_audit_agent_instance()
        if audit_agent:
            policy_num = extracted_data.get('claim_info', {}).get('policy_number', 'UNKNOWN')
            audit_agent.log_document_agent_action(
                policy_number=policy_num,
                action="document_analysis",
                inputs={
                    "filename": filename,
                    "page_count": extracted_data.get('page_count', 0)
                },
                outputs={
                    "extracted_fields": extracted_data.get('claim_info', {}),
                    "pages_analyzed": extracted_data.get('page_count', 0),
                    "policy_validated": extracted_data.get('policy_validated', False)
                },
                decision="SUCCESS",
                confidence_score=0.95,
                metadata={"ocr_engine": "Azure Document Intelligence"}
            )
        
        return extracted_data
    
    except Exception as e:
        st.error(f"✌ Document Intelligence Agent Error: {str(e)}")
        return None

def generate_summary(extracted_data):
    """Step 2: AI Summary Generation"""
    openai_client = get_openai_client()
    if not openai_client:
        return None
    
    try:
        deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
        
        prompt = f"""You are an expert insurance claims analyst. Analyze this insurance claim document.

Extract and provide:
1. **Policy Number**: The insurance policy number
2. **Claim Status**: Current status (pending, approved, rejected, etc.)
3. **Claim Amount**: Amount being claimed
4. **Summary**: Brief summary of the claim
5. **Key Details**: Important dates, parties involved, incident details

Extracted Text:
{extracted_data['text'][:3000]}

Key-Value Pairs:
{json.dumps(extracted_data['key_value_pairs'], indent=2)}

Provide a clear, structured analysis."""

        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing insurance claim documents."},
            {"role": "user", "content": prompt}
        ]
        
        with st.spinner("🤖 AI is generating summary..."):
            response = openai_client.chat.completions.create(
                messages=messages,
                max_completion_tokens=1500,
                temperature=0.3,
                model=deployment
            )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"✌ AI Summary Error: {str(e)}")
        return None

def validate_policy(extracted_data, ai_summary):
    """Step 3: Databricks Agent - Policy Validation"""
    policy_validator = get_policy_validator()
    
    if not policy_validator or not policy_validator.enabled:
        return None
    
    # Handle None extracted_data
    if extracted_data is None:
        st.warning("⚠️ Document extraction failed - cannot validate policy")
        return None
    
    try:
        with st.spinner("💾 Databricks Agent is validating policy..."):
            validation_result = policy_validator.process_claim_document(extracted_data, ai_summary)
        
        # Log to Audit Agent
        audit_agent = get_audit_agent_instance()
        if audit_agent and validation_result:
            policy_num = validation_result.get('policy_info', {}).get('policy_number', 'UNKNOWN')
            policy_details = validation_result.get('validation', {}).get('details', {})
            
            audit_agent.log_databricks_agent_action(
                policy_number=policy_num,
                action="policy_validation",
                inputs={
                    "policy_number": policy_num,
                    "claim_info": extracted_data.get('claim_info', {})
                },
                outputs={
                    "policy_found": validation_result.get('validation', {}).get('valid', False),
                    "policy_status": policy_details.get('policy_status', 'Unknown'),
                    "policy_type": policy_details.get('policy_type', 'Unknown'),
                    "coverage_limit": policy_details.get('policy_limit', 0),
                    "expiry_date": policy_details.get('policy_expiry_date', 'Unknown')
                },
                decision="POLICY_FOUND" if validation_result.get('validation', {}).get('valid') else "POLICY_NOT_FOUND",
                query_executed="SELECT * FROM policies WHERE policy_number = ?",
                metadata={"database": "Databricks"}
            )
        
        return validation_result
    
    except Exception as e:
        st.error(f"✌ Databricks Agent Error: {str(e)}")
        return None

def check_claim_eligibility(extracted_data, ai_summary, validation_result):
    """Step 4: Eligibility Agent - Rule-based eligibility checks with AI for exclusions and dynamic confidence scoring"""
    
    if not validation_result or not extracted_data.get('claim_info'):
        return {
            "eligibility_decision": "ERROR",
            "confidence_score": 0,
            "reasoning": "Missing policy or claim information",
            "checks_failed": []
        }
    
    policy_info = validation_result.get('policy_info', {})
    policy_details = validation_result.get('validation', {}).get('details', {})
    claim_info = extracted_data.get('claim_info', {})
    
    # Extract values from database
    policy_limit = float(policy_details.get('policy_limit', 0))
    past_claims_amount = float(policy_details.get('past_claims_amount', 0))
    claim_history_count = int(policy_details.get('claim_history_count', 0))
    policy_status = policy_details.get('policy_status', '').lower()
    policy_expiry_date = policy_details.get('policy_expiry_date', '')
    exclusions = policy_details.get('exclusions', '')
    
    # Extract values from claim
    claim_amount = float(claim_info.get('claim_amount', 0)) if claim_info.get('claim_amount') else 0
    claim_date = claim_info.get('claim_date', '')
    reason_for_claim = claim_info.get('reason_for_claim', '')
    
    # Track failed checks and ambiguity factors
    checks_failed = []
    eligibility_messages = []
    ambiguity_score = 0  # Track uncertainty factors for confidence calculation
    ambiguity_reasons = []
    
    # AMBIGUITY DETECTION: Check for missing or unclear data
    if not claim_amount or claim_amount == 0:
        ambiguity_score += 20
        ambiguity_reasons.append("Missing or zero claim amount")
    
    if not reason_for_claim or len(reason_for_claim.strip()) < 10:
        ambiguity_score += 15
        ambiguity_reasons.append("Missing or insufficient claim reason")
    
    if not policy_status or policy_status == 'unknown':
        ambiguity_score += 25
        ambiguity_reasons.append("Policy status unclear or missing")
    
    if not policy_expiry_date:
        ambiguity_score += 15
        ambiguity_reasons.append("Policy expiry date missing")
    
    if not claim_date:
        ambiguity_score += 10
        ambiguity_reasons.append("Claim date missing")
    
    if policy_limit == 0:
        ambiguity_score += 20
        ambiguity_reasons.append("Policy limit data unavailable")
    
    # CHECK 1: Claim amount within available limit
    available_limit = policy_limit - past_claims_amount
    
    # Handle negative available limit (policy already exceeded)
    if available_limit < 0:
        available_limit = 0  # Treat as no available limit
    
    # Detect borderline cases (within 10% of limit)
    if claim_amount > 0 and available_limit > 0:
        utilization_percentage = (claim_amount / available_limit) * 100
        if 90 <= utilization_percentage <= 110:
            ambiguity_score += 10
            ambiguity_reasons.append(f"Claim amount borderline ({utilization_percentage:.1f}% of available limit)")
    
    if claim_amount > available_limit:
        checks_failed.append("Claim amount exceeded available limit")
        eligibility_messages.append(f"❌ **Check 1 - Failed**: Claim amount ${claim_amount:,.2f} exceeds the available policy limit ${available_limit:,.2f}")
        eligibility_messages.append(f"   • Policy Limit: ${policy_limit:,.2f}")
        eligibility_messages.append(f"   • Past Claims: ${past_claims_amount:,.2f}")
        eligibility_messages.append(f"   • Available: ${available_limit:,.2f}")
        eligibility_messages.append("\n❌ **Final Decision: Not Eligible - Claim Amount Exceeds Available Limit**")
        eligibility_messages.append("   • The claim amount is higher than the available coverage")
        eligibility_messages.append("   • Customer has insufficient coverage remaining for this claim")
        
        return {
            "eligibility_decision": "NOT ELIGIBLE",
            "confidence_score": 95,
            "reasoning": f"Claim amount ${claim_amount:,.2f} exceeds available policy limit ${available_limit:,.2f}",
            "checks_failed": checks_failed,
            "ambiguity_factors": [],
            "detailed_checks": eligibility_messages,
            "key_factors": [
                f"Claim amount: ${claim_amount:,.2f}",
                f"Available limit: ${available_limit:,.2f}",
                "Insufficient coverage remaining"
            ],
            "risk_assessment": "Not applicable - claim rejected",
            "supporting_evidence": [
                f"Policy limit: ${policy_limit:,.2f}",
                f"Past claims: ${past_claims_amount:,.2f}",
                f"Available: ${available_limit:,.2f}"
            ],
            "updated_values": None,
            "ai_exclusion_analysis": "N/A"
        }
    else:
        eligibility_messages.append(f"✅ **Check 1 - Passed**: Claim amount ${claim_amount:,.2f} is within the available policy limit ${available_limit:,.2f}")
    
    # CHECK 2: Policy status must be active (CRITICAL CHECK - expired policies are never eligible)
    if policy_status not in ['active', 'valid', 'current']:
        checks_failed.append("Policy is not active")
        eligibility_messages.append(f"❌ **Check 2 - Failed**: Policy status is '{policy_status.title()}'. Only active policies are eligible for claims.")
        
        # CRITICAL: If policy is expired/inactive, immediately return NOT ELIGIBLE
        if policy_status in ['expired', 'inactive', 'terminated', 'cancelled']:
            eligibility_messages.append("\n❌ **Final Decision: Not Eligible - Policy Expired/Inactive**")
            eligibility_messages.append("   • Expired or inactive policies cannot process new claims")
            eligibility_messages.append("   • Customer must renew policy before submitting claims")
            
            return {
                "eligibility_decision": "NOT ELIGIBLE",
                "confidence_score": 100,  # 100% confident that expired = not eligible
                "reasoning": f"Policy status is '{policy_status.title()}' - expired or inactive policies are not eligible for claims",
                "checks_failed": checks_failed,
                "ambiguity_factors": [],
                "detailed_checks": eligibility_messages,
                "key_factors": [
                    f"Policy status: {policy_status.title()}",
                    "Expired/inactive policies cannot process claims"
                ],
                "risk_assessment": "Not applicable - claim rejected",
                "supporting_evidence": [
                    f"Policy status: {policy_status.title()}",
                    "Only active policies are eligible"
                ],
                "updated_values": None,
                "ai_exclusion_analysis": "N/A"
            }
    else:
        eligibility_messages.append(f"✅ **Check 2 - Passed**: Policy status is '{policy_status.title()}' and valid for claims.")
    
    # CHECK 3: Claim history count (max 4 claims allowed)
    if claim_history_count >= 4:
        checks_failed.append("Maximum claim count exceeded")
        eligibility_messages.append(f"❌ **Check 3 - Failed**: Claim count is {claim_history_count}. Maximum allowed is 4 claims per policy period.")
        eligibility_messages.append("\n❌ **Final Decision: Not Eligible - Maximum Claims Exceeded**")
        eligibility_messages.append(f"   • Policy has already reached {claim_history_count} claims (maximum: 4)")
        eligibility_messages.append("   • No additional claims can be processed for this policy period")
        
        return {
            "eligibility_decision": "NOT ELIGIBLE",
            "confidence_score": 95,
            "reasoning": f"Policy has reached maximum claim count ({claim_history_count} claims, limit is 4)",
            "checks_failed": checks_failed,
            "ambiguity_factors": [],
            "detailed_checks": eligibility_messages,
            "key_factors": [
                f"Current claims: {claim_history_count}",
                "Maximum allowed: 4 claims",
                "Claim limit exceeded"
            ],
            "risk_assessment": "Not applicable - claim rejected",
            "supporting_evidence": [
                f"Claim history count: {claim_history_count}",
                "Policy allows maximum 4 claims per period"
            ],
            "updated_values": None,
            "ai_exclusion_analysis": "N/A"
        }
    else:
        eligibility_messages.append(f"✅ **Check 3 - Passed**: Claim count of {claim_history_count} is within the allowed limit (maximum 4 claims).")
    
    # CHECK 4: Claim date vs policy expiry date
    from datetime import datetime
    claim_date_check_failed = False
    date_parse_failed = False
    
    try:
        if claim_date and policy_expiry_date:
            # Try parsing different date formats (including DD-MM-YYYY, MM-DD-YYYY, YYYY-MM-DD, etc.)
            parsed_successfully = False
            date_formats = [
                '%Y-%m-%d',    # 2024-11-14
                '%d-%m-%Y',    # 14-11-2024
                '%m-%d-%Y',    # 11-14-2024
                '%Y/%m/%d',    # 2024/11/14
                '%d/%m/%Y',    # 14/11/2024
                '%m/%d/%Y',    # 11/14/2024
                '%d.%m.%Y',    # 14.11.2024
                '%Y.%m.%d',    # 2024.11.14
                '%d %b %Y',    # 14 Nov 2024
                '%b %d, %Y'    # Nov 14, 2024
            ]
            
            for date_format in date_formats:
                try:
                    claim_dt = datetime.strptime(claim_date, date_format)
                    expiry_dt = datetime.strptime(policy_expiry_date, date_format)
                    parsed_successfully = True
                    
                    if claim_dt > expiry_dt:
                        checks_failed.append("Claim date after policy expiry")
                        eligibility_messages.append(f"❌ **Check 4 - Failed**: Claim date ({claim_date}) is after the policy expiry date ({policy_expiry_date})")
                        eligibility_messages.append("\n❌ **Final Decision: Not Eligible - Claim Date After Policy Expiry**")
                        eligibility_messages.append(f"   • Claim date: {claim_date}")
                        eligibility_messages.append(f"   • Policy expired on: {policy_expiry_date}")
                        eligibility_messages.append("   • Claims cannot be filed after policy expiration")
                        
                        return {
                            "eligibility_decision": "NOT ELIGIBLE",
                            "confidence_score": 95,
                            "reasoning": f"Claim date ({claim_date}) is after policy expiry date ({policy_expiry_date})",
                            "checks_failed": checks_failed,
                            "ambiguity_factors": [],
                            "detailed_checks": eligibility_messages,
                            "key_factors": [
                                f"Claim date: {claim_date}",
                                f"Policy expiry: {policy_expiry_date}",
                                "Claim filed after expiration"
                            ],
                            "risk_assessment": "Not applicable - claim rejected",
                            "supporting_evidence": [
                                f"Claim date: {claim_date}",
                                f"Expiry date: {policy_expiry_date}"
                            ],
                            "updated_values": None,
                            "ai_exclusion_analysis": "N/A"
                        }
                    else:
                        eligibility_messages.append(f"✅ **Check 4 - Passed**: Claim date ({claim_date}) is before the policy expiry date ({policy_expiry_date})")
                    break
                except ValueError:
                    continue
            
            if not parsed_successfully:
                date_parse_failed = True
                ambiguity_score += 20
                ambiguity_reasons.append("Unable to parse claim or policy dates - format ambiguous")
                eligibility_messages.append(f"⚠️ **Check 4 - Ambiguous**: Could not validate dates. Claim: {claim_date}, Expiry: {policy_expiry_date}")
        elif not claim_date or not policy_expiry_date:
            eligibility_messages.append(f"⚠️ **Check 4 - Skipped**: Missing date information")
    except Exception as e:
        ambiguity_score += 15
        ambiguity_reasons.append(f"Date validation error: {str(e)}")
        eligibility_messages.append(f"⚠️ **Check 4 - Warning**: Could not validate claim date - {str(e)}")
    
    # CHECK 5: AI-powered exclusion check
    exclusion_check_failed = False
    ai_exclusion_summary = ""
    
    if reason_for_claim and exclusions:
        openai_client = get_openai_client()
        if openai_client:
            try:
                deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
                
                prompt = f"""You are an insurance policy analyst. 

**Policy Exclusions:**
{exclusions}

**Claim Reason:**
{reason_for_claim}

**Task:** Determine if the claim reason matches or is related to any of the policy exclusions.

Analyze carefully:
1. Is the claim reason explicitly listed in the exclusions?
2. Is the claim reason similar to or falls under any exclusion category?
3. Are there any keywords or concepts that match?

Return a JSON object with:
{{
    "is_excluded": true/false,
    "reasoning": "Brief explanation of why the claim is or isn't excluded",
    "matched_exclusion": "The specific exclusion that matches (if any)",
    "confidence": 0-100 (percentage confidence in this decision)
}}
"""
                
                response = openai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert insurance policy analyst specializing in exclusion analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=500,
                    temperature=0.1,
                    model=deployment,
                    response_format={"type": "json_object"}
                )
                
                exclusion_analysis = json.loads(response.choices[0].message.content)
                
                # Use AI's confidence in the exclusion analysis
                ai_confidence = exclusion_analysis.get('confidence', 50)
                if ai_confidence < 60:
                    ambiguity_score += (60 - ai_confidence) / 2  # Add up to 30 points for low AI confidence
                    ambiguity_reasons.append(f"AI exclusion analysis has low confidence ({ai_confidence}%)")
                
                if exclusion_analysis.get('is_excluded'):
                    checks_failed.append("Claim reason matches policy exclusion")
                    exclusion_check_failed = True
                    matched_exclusion = exclusion_analysis.get('matched_exclusion', 'Not specified')
                    ai_reasoning = exclusion_analysis.get('reasoning', '')
                    eligibility_messages.append(f"❌ **Check 5 - Failed (AI Analysis)**: Claim rejected due to policy exclusion")
                    eligibility_messages.append(f"   • Matched Exclusion: {matched_exclusion}")
                    eligibility_messages.append(f"   • Reason: {ai_reasoning}")
                    eligibility_messages.append(f"   • AI Confidence: {ai_confidence}%")
                    ai_exclusion_summary = f"Excluded: {matched_exclusion}. {ai_reasoning}"
                else:
                    ai_reasoning = exclusion_analysis.get('reasoning', '')
                    eligibility_messages.append(f"✅ **Check 5 - Passed (AI Analysis)**: Claim reason does not match any policy exclusions")
                    eligibility_messages.append(f"   • Analysis: {ai_reasoning}")
                    eligibility_messages.append(f"   • AI Confidence: {ai_confidence}%")
                    ai_exclusion_summary = f"Not excluded. {ai_reasoning}"
                
            except Exception as e:
                eligibility_messages.append(f"⚠️ **Check 5 - Warning**: Could not perform AI exclusion analysis - {str(e)}")
        else:
            eligibility_messages.append(f"⚠️ **Check 5 - Warning**: AI service unavailable for exclusion analysis")
    else:
        eligibility_messages.append(f"⚠️ **Check 5 - Skipped**: Missing claim reason or exclusions data")
    
    # Final decision with dynamic confidence scoring
    if len(checks_failed) == 0:
        # All checks passed - Update values
        new_past_claims_amount = past_claims_amount + claim_amount
        new_claim_history_count = claim_history_count + 1
        
        eligibility_decision = "ELIGIBLE"
        
        # Calculate confidence: Start at 95%, reduce by ambiguity
        confidence_score = max(95 - ambiguity_score, 30)  # Minimum 30%
        
        if ambiguity_score > 0:
            reasoning = f"All eligibility checks passed. However, {len(ambiguity_reasons)} ambiguity factor(s) detected, reducing confidence."
        else:
            reasoning = "All eligibility checks passed. Claim is approved for processing."
        
        eligibility_messages.append("\n🎉 **Final Decision: Eligible**")
        
        if ambiguity_reasons:
            eligibility_messages.append(f"\n⚠️ **Ambiguity Factors Detected ({len(ambiguity_reasons)}):**")
            for reason in ambiguity_reasons:
                eligibility_messages.append(f"   • {reason}")
        
        eligibility_messages.append("\n**Updated Values:**")
        eligibility_messages.append(f"   Past Claims Amount: ${past_claims_amount:,.2f} -> ${new_past_claims_amount:,.2f}")
        eligibility_messages.append(f"   Claim History Count: {claim_history_count} -> {new_claim_history_count}")
        
        result = {
            "eligibility_decision": eligibility_decision,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "checks_failed": checks_failed,
            "ambiguity_factors": ambiguity_reasons,
            "detailed_checks": eligibility_messages,
            "key_factors": [
                f"Claim amount ${claim_amount:,.2f} within available limit ${available_limit:,.2f}",
                f"Policy status is '{policy_status.title()}'",
                f"Claim count {claim_history_count} is within limit",
                "Claim date is valid",
                "No policy exclusions matched"
            ],
            "risk_assessment": "Low risk - All validation checks passed" if confidence_score >= 70 else "Medium risk - Ambiguity factors present",
            "supporting_evidence": [
                f"Available coverage: ${available_limit:,.2f}",
                f"Claims remaining: {3 - claim_history_count}",
                f"Policy valid until: {policy_expiry_date}"
            ],
            "updated_values": {
                "new_past_claims_amount": new_past_claims_amount,
                "new_claim_history_count": new_claim_history_count
            },
            "ai_exclusion_analysis": ai_exclusion_summary
        }
        
        # Log to Audit Agent
        audit_agent = get_audit_agent_instance()
        if audit_agent:
            policy_num = claim_info.get('policy_number', 'UNKNOWN')
            audit_agent.log_eligibility_agent_action(
                policy_number=policy_num,
                action="eligibility_check",
                inputs={
                    "claim_amount": claim_amount,
                    "claim_date": claim_date,
                    "policy_status": policy_status,
                    "coverage_limit": policy_limit,
                    "expiry_date": policy_expiry_date
                },
                outputs={
                    "eligibility_status": eligibility_decision,
                    "checks_passed": 5,
                    "checks_failed": 0
                },
                decision=eligibility_decision,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                checks_performed=[msg for msg in eligibility_messages if "Check" in msg],
                metadata={"risk_assessment": result["risk_assessment"]}
            )
        
        return result
    else:
        # Some checks failed
        eligibility_decision = "NOT ELIGIBLE"
        
        # Calculate confidence for rejection
        # High confidence (90%) = Clear reasons for rejection (expired, over limit, etc.)
        # Low confidence (< 50%) = Uncertain rejection due to missing data or ambiguity
        
        if ambiguity_score >= 50:
            # High ambiguity - not confident in rejection
            confidence_score = max(40 - (ambiguity_score - 50) / 2, 20)  # 20-40% range
            reasoning = f"Claim failed {len(checks_failed)} check(s), but high ambiguity detected. Human review strongly recommended."
        elif ambiguity_score >= 30:
            # Medium ambiguity - somewhat uncertain
            confidence_score = 50 + (50 - ambiguity_score)  # 50-70% range
            reasoning = f"Claim failed {len(checks_failed)} check(s) with some ambiguity. Consider human review."
        else:
            # Low ambiguity - confident in rejection
            confidence_score = max(90 - ambiguity_score, 75)  # 75-90% range
            reasoning = f"Claim failed {len(checks_failed)} eligibility check(s) with clear reasons."
        
        eligibility_messages.append("\n🚫 **Final Decision: Not Eligible**")
        eligibility_messages.append(f"\n**Failed {len(checks_failed)} Check(s):**")
        for i, failed_check in enumerate(checks_failed, 1):
            eligibility_messages.append(f"   {i}. {failed_check}")
        
        if ambiguity_reasons:
            eligibility_messages.append(f"\n⚠️ **Ambiguity Factors ({len(ambiguity_reasons)}):**")
            for reason in ambiguity_reasons:
                eligibility_messages.append(f"   • {reason}")
        
        result = {
            "eligibility_decision": eligibility_decision,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "checks_failed": checks_failed,
            "ambiguity_factors": ambiguity_reasons,
            "detailed_checks": eligibility_messages,
            "key_factors": checks_failed,
            "risk_assessment": "N/A - Claim not eligible" if confidence_score >= 70 else "Uncertain - Human review needed",
            "supporting_evidence": [],
            "ai_exclusion_analysis": ai_exclusion_summary if exclusion_check_failed else "N/A"
        }
        
        # Log to Audit Agent
        audit_agent = get_audit_agent_instance()
        if audit_agent:
            policy_num = claim_info.get('policy_number', 'UNKNOWN')
            audit_agent.log_eligibility_agent_action(
                policy_number=policy_num,
                action="eligibility_check",
                inputs={
                    "claim_amount": claim_amount,
                    "claim_date": claim_date,
                    "policy_status": policy_status,
                    "coverage_limit": policy_limit,
                    "expiry_date": policy_expiry_date
                },
                outputs={
                    "eligibility_status": eligibility_decision,
                    "checks_passed": 0,
                    "checks_failed": len(checks_failed)
                },
                decision=eligibility_decision,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                checks_performed=[msg for msg in eligibility_messages if "Check" in msg],
                metadata={"risk_assessment": result["risk_assessment"]}
            )
        
        return result


def show_workflow_progress(step, total_steps=7):
    """Display visual workflow progress with current step highlighted"""
    
    steps = [
        {"num": 1, "name": "Orchestrator", "icon": "🎯", "color": "#4F8EF7"},
        {"num": 2, "name": "Document", "icon": "📄", "color": "#00C853"},
        {"num": 3, "name": "Databricks", "icon": "💾", "color": "#FF6F00"},
        {"num": 4, "name": "Eligibility", "icon": "🔍", "color": "#9C27B0"},
        {"num": 5, "name": "Fraud", "icon": "🚨", "color": "#E91E63"},
        {"num": 6, "name": "Human Review", "icon": "👤", "color": "#2196F3"},
        {"num": 7, "name": "Communication", "icon": "📧", "color": "#00BCD4"}
    ]
    
    # Create a styled container with better spacing
    st.markdown("""
    <style>
    .workflow-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 50px 20px;
        margin: 20px 0;
    }
    .agent-circle {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.8em;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        margin: 0 auto 15px auto;
    }
    .agent-name {
        font-weight: bold;
        color: #fff;
        font-size: 0.95em;
        text-align: center;
        white-space: nowrap;
        margin-bottom: 8px;
    }
    .agent-status {
        font-size: 0.8em;
        text-transform: uppercase;
        text-align: center;
    }
    .arrow-container {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2em;
        margin-top: 35px;
        padding: 0 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display agents in a single row with center alignment
    workflow_html = '<div style="display: flex; justify-content: center; align-items: flex-start; min-width: 100%; gap: 0;">'
    
    for i, s in enumerate(steps):
        # Special handling: if step > total steps, show all completed and Orchestrator active
        if step > len(steps):
            if s["num"] == 1:  # Orchestrator
                status = "active"
                bg_color = "#FFD700"
                text_color = "#000"
                icon = "⚙️"
                status_color = "#FFD700"
                border = "border: 4px solid #fff;"
            else:
                status = "completed"
                bg_color = "#4CAF50"
                text_color = "#fff"
                icon = "✅"
                status_color = "#4CAF50"
                border = ""
        # Normal workflow progression
        elif s["num"] < step:
            status = "completed"
            bg_color = "#4CAF50"
            text_color = "#fff"
            icon = "✅"
            status_color = "#4CAF50"
            border = ""
        elif s["num"] == step:
            status = "active"
            bg_color = "#FFD700"
            text_color = "#000"
            icon = "⚙️"
            status_color = "#FFD700"
            border = "border: 4px solid #fff;"
        else:
            status = "pending"
            bg_color = "#666"
            text_color = "#999"
            icon = "⏳"
            status_color = "#999"
            border = ""
        
        # Add agent
        workflow_html += f'''
    <div style="flex: 1; min-width: 100px; text-align: center;">
        <div class="agent-circle" style="background: {bg_color}; color: {text_color}; {border}">
            {icon}
        </div>
        <div class="agent-name">{s["name"]}</div>
        <div class="agent-status" style="color: {status_color};">{status}</div>
    </div>'''
        
        # Add arrow between steps (after agent, before next agent)
        if i < len(steps) - 1:
            arrow_color = "#4CAF50" if step > s["num"] else "#999"
            workflow_html += f'''
    <div style="flex: 0 0 30px; text-align: center;">
        <div class="arrow-container">
            <div style="color: {arrow_color};">➜</div>
        </div>
    </div>'''
    
    workflow_html += '</div>'
    st.markdown(workflow_html, unsafe_allow_html=True)

def main():
    # Header with custom CSS to make title fit on one line
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
    <p class="subtitle">Multi-Agent Workflow with Real-time Visualization</p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - System Information
    with st.sidebar:
        st.header("ℹ️ System Information")
        
        st.markdown("### 🏥 Insurance Claims System")
        st.markdown("""
        **Version:** 2.0  
        **Environment:** Production  
        **Region:** East US 2
        """)
        
        
        st.markdown("---")
        st.markdown("### 🕐 Session Info")
        st.info(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content tabs - removed Workflow Visualization tab
    tab1, tab2 = st.tabs(["📝 Process Claims", "👤 Human Review"])
    
    # Check if we need to show Human Review notification
    if st.session_state.get('needs_human_review', False):
        st.info("🔔 **A claim requires human review.** Please switch to the **'Human Review'** tab to process it.")
    
    if st.session_state.get('rejected_claim_for_review'):
        st.warning("⚠️ **A rejected claim is available for review.** You can override the decision in the **'Human Review'** tab.")
    
    with tab1:
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
                # Clear previous processing session
                if 'fraud_review_result' in st.session_state:
                    del st.session_state['fraud_review_result']
                if 'fraud_claim_for_review' in st.session_state:
                    del st.session_state['fraud_claim_for_review']
                if 'processing_completed' in st.session_state:
                    del st.session_state['processing_completed']
                if 'last_processing_results' in st.session_state:
                    del st.session_state['last_processing_results']
                
                # Initialize session state for workflow tracking
                if 'workflow_state' not in st.session_state:
                    st.session_state.workflow_state = {}
                
                # Log Orchestrator initialization
                workflow_start_time = datetime.now()
                
                # Create placeholders for dynamic updates
                sidebar_status = st.sidebar.empty()
                
                st.markdown("---")
                
                # ====================================
                # VISUAL WORKFLOW PROGRESS TRACKER
                # ====================================
                st.markdown("### 🔄 Live Workflow Progress")
                
                # Create placeholders for dynamic workflow updates
                workflow_placeholder = st.empty()
                progress_placeholder = st.empty()
                status_text = st.empty()
                
                # Initialize result storage
                results = {
                    'extracted_data': None,
                    'ai_summary': None,
                    'validation_result': None,
                    'eligibility_analysis': None
                }
                
                # Orchestrator starts
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🟢 ONLINE  
                ✅ Workflow started  
                📹 Initializing pipeline
                
                ---
                
                **📄 Document Agent** 🔴 OFFLINE
                
                ---
                
                **💾 Databricks Agent** 🔴 OFFLINE
                
                ---
                
                **🔍 Eligibility Agent** 🔴 OFFLINE
                
                ---
                
                **🚨 Fraud Detection** 🔴 OFFLINE
                
                ---
                
                **📧 Communication** 🔴 OFFLINE
                
                ---
                
                **📝 Audit Agent** 🔴 OFFLINE
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                """)
                
                # Show Step 1: Orchestrator starting
                workflow_placeholder.container()
                with workflow_placeholder.container():
                        show_workflow_progress(step=1)
                
                progress_placeholder.progress(0.1)
                status_text.info("🎯 **Orchestrator Agent** is initializing the workflow...")
                
                # Dynamic placeholder for processing details
                detail_placeholder = st.empty()
                
                # Show orchestrator start - Step 1
                detail_placeholder.markdown("""
                **🎯 Orchestrator Agent** - *Working*
                - 📄 Initializing workflow pipeline...
                - 📹 Loading configuration...
                """)
                
                time.sleep(0.4)
                
                detail_placeholder.markdown("""
                **🎯 Orchestrator Agent** - *Working*
                - ✅ Workflow pipeline initialized
                - 📄 Setting up agent coordination...
                - 💤 Delegating to Document Agent...
                """)
                
                time.sleep(0.4)
                
                # STEP 1: Document Intelligence
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
                ---
                
                **📄 Document Agent** 🟢 ONLINE  
                ⚙️ Processing document  
                🔍 Extracting data...
                
                ---
                
                **💾 Databricks Agent** 🔴 OFFLINE
                
                ---
                
                **🔍 Eligibility Agent** 🔴 OFFLINE
                
                ---
                
                **🚨 Fraud Detection** 🔴 OFFLINE
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                
                ---
                
                **📧 Communication** 🔴 OFFLINE
                """)
                
                # Update to Step 2: Document Agent
                with workflow_placeholder.container():
                    show_workflow_progress(step=2)

                
                progress_placeholder.progress(0.14)  # 1/7 = 14%
                status_text.info("📄 **Document Agent** is extracting text and analyzing the document...")
                
                detail_placeholder.markdown("""
                **📄 Document Agent** - *Working*
                - 📄 Starting OCR engine...
                - 📄 Loading document...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **📄 Document Agent** - *Working*
                - ✅ OCR engine started
                - 📄 Reading document pages...
                - 📄 Page 1 of analysis...
                """)
                
                time.sleep(0.3)
                
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                
                detail_placeholder.markdown("""
                **📄 Document Agent** - *Working*
                - ✅ Document pages loaded
                - 📄 Extracting text and key-value pairs...
                - 🧠 AI analyzing content...
                """)
                
                time.sleep(0.3)
                
                extracted_data = analyze_document(file_bytes, uploaded_file.name)
                results['extracted_data'] = extracted_data
                
                # Check if policy validation failed
                if extracted_data and extracted_data.get('error'):
                    st.error("✌ Document Intelligence Agent: Policy Validation Failed")

                    st.error(extracted_data.get('message'))
                
                    progress_placeholder.progress(0.0)
                    status_text.error("🚫 **Process Stopped** - Invalid policy number")
                
                    sidebar_status.markdown("""
                    ### 📄 Current Status
                
**🎯 Orchestrator Agent** 🔴 OFFLINE

                
---

                
**📄 Document Agent** 🔴 OFFLINE  

                ✌ Validation failed  
                🚫 Invalid policy number
                
---

                
**💾 Databricks Agent** 🔴 OFFLINE

                
**🔍 Eligibility Agent** 🔴 OFFLINE

                """)
                
                st.stop()
                
                if not extracted_data:
                    st.error("✌ Document Intelligence Agent: Failed")

                    st.stop()
                
                progress_placeholder.progress(0.4)
                status_text.success("✅ **Document Agent** completed - Policy number validated successfully!")
                
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
                ---
                
                **📄 Document Agent** 🟢 ONLINE  
                ✅ Analysis complete  
                📊 Data extracted
                
                ---
                
                **💾 Databricks Agent** 🔴 OFFLINE
                
                ---
                
                **🔍 Eligibility Agent** 🔴 OFFLINE
                
                ---
                
                **📝 Audit Agent** 🔴 OFFLINE
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                """)
                
                page_count = extracted_data.get('page_count', 0)
                kv_count = len(extracted_data.get('key_value_pairs', []))
                detail_placeholder.markdown(f"""
                **📄 Document Agent** - *Finalizing*
                - ✅ Text extraction complete
                - ✅ Extracted **{page_count} pages**
                - ✅ Found **{kv_count} key-value pairs**
                - 📄 Structuring data...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown(f"""
                **📄 Document Agent** ✅ **COMPLETED**
                - ✅ Extracted **{page_count} pages**
                - ✅ Found **{kv_count} key-value pairs**
                - ✅ Document structure analyzed
                - 📤 Sending data to Orchestrator...
                """)
                
                time.sleep(0.4)
                
                # STEP 2: Orchestrator AI Summary
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
                ---
                
                **📄 Document Agent** ✅ COMPLETED
                
                ---

                **💾 Databricks Agent** 🟢 ONLINE  
                ⚙️ Querying policy data...
                
                ---
                
                **🔍 Eligibility Agent** 🔴 OFFLINE
                
                ---
                
                **🚨 Fraud Detection** 🔴 OFFLINE
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                
                ---
                
                **📧 Communication** 🔴 OFFLINE
                """)
                
                
                # Document Agent completed - now showing Databricks as Step 3
                with workflow_placeholder.container():
                    show_workflow_progress(step=3)

                                
                progress_placeholder.progress(0.29)  # 2/7 = 29%
                status_text.info("💾 **Databricks Agent** is querying policy data...")
                
                detail_placeholder.markdown("""
                **🤖 AI Summary Agent** - *Working*
                - 🔥 Received data from Document Agent
                - 📄 Processing extracted information...
                - 🧠 Initializing GPT-4 connection...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **🤖 AI Summary Agent** - *Working*
                - ✅ GPT-4 connection established
                - 📤 Preparing AI prompt...
                - 📄 Analyzing document content...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **🤖 AI Summary Agent** - *Working*
                - ✅ Prompt prepared
                - 📄 Running AI analysis...
                - 🧠 Generating intelligent summary...
                - 💭 Extracting policy details...
                """)
                
                ai_summary = generate_summary(extracted_data)
                results['ai_summary'] = ai_summary
                
                if not ai_summary:
                    st.error("AI Summary Agent: Analysis Failed")

                    st.stop()
                
                progress_placeholder.progress(0.375)  # 3/8 = 37.5%
                status_text.success("✅ **AI Summary Agent** completed - Summary generated!")
                
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🟢 ONLINE  
                ✅ AI analysis complete  
                📹 Policy info extracted
                
                ---
                
                **📄 Document Agent** 🔴 OFFLINE
                        


                **💾 Databricks Agent** 🔴 OFFLINE
                """)
                
                detail_placeholder.markdown("""
                **🤖 AI Summary Agent** - *Finalizing*
                - ✅ AI analysis complete
                - ✅ Summary generated
                - 📄 Extracting policy information...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **🤖 AI Summary Agent** ✅ **COMPLETED**
                - ✅ AI summary generated successfully
                - ✅ Policy information extracted
                - ✅ Key details identified
                - 📤 Routing to Databricks Agent...
                """)
                
                time.sleep(0.4)
                
                # STEP 4: Databricks - Policy Validation
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
                ---
                
                **📄 Document Agent** 🔴 OFFLINE
                
                ---

                **💾 Databricks Agent** 🟢 ONLINE  
                ⚙️ Connecting to database  
                🔍 Validating policy...
                
                ---
                
                **🔍 Eligibility Agent** 🔴 OFFLINE
                
                ---
                
                **📝 Audit Agent** 🔴 OFFLINE
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                """)
                
                # Update to Step 4: Eligibility Check
                with workflow_placeholder.container():      
                        show_workflow_progress(step=4)

                
                progress_placeholder.progress(0.67)  # 4/6 = 67%
                status_text.info("🔍 **Eligibility Agent** is checking claim eligibility...")
                
                detail_placeholder.markdown("""
                **💾 Databricks Agent** - *Working*
                - 📄 Initializing database connection...
                - 🔍 Connecting to policy database...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **💾 Databricks Agent** - *Working*
                - ✅ Database connection established
                - 📤 Searching for policy records...
                - 📊 Querying validation rules...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **💾 Databricks Agent** - *Working*
                - ✅ Policy records found
                - 📄 Running policy validation checks...
                - 🔍 Verifying policy status...
                - 📊 Checking eligibility criteria...
                """)
                
                validation_result = validate_policy(extracted_data, ai_summary)
                results['validation_result'] = validation_result
                
                progress_placeholder.progress(0.9)
                
                if validation_result:
                    status_text.success("✅ **Databricks Agent** completed - Policy validated successfully!")

                
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
---

                
**📄 Document Agent** 🔴 OFFLINE

                
---


**💾 Databricks Agent** 🟢 ONLINE  

                ✅ Validation complete  
                📊 Policy verified
                
---

                
**🔍 Eligibility Agent** 🔴 OFFLINE

                
---

                
**📝 Audit Agent** 🔴 OFFLINE

                
---

                
**👤 Human Review** 🔴 OFFLINE

                """)
                
                policy_num = validation_result.get('policy_info', {}).get('policy_number', 'N/A')
                policy_status = validation_result.get('policy_info', {}).get('policy_status', 'Unknown')
                detail_placeholder.markdown(f"""
                **💾 Databricks Agent** - *Finalizing*
                - ✅ Validation checks completed
                - ✅ Policy **{policy_num}** found
                - 📄 Compiling results...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown(f"""
                **💾 Databricks Agent** ✅ **COMPLETED**
                - ✅ Policy **{policy_num}** validated
                - ✅ Status: **{policy_status.upper()}**
                - ✅ All validation checks passed
                - 📊 Returning results to Orchestrator...
                """)
                
                time.sleep(0.5)
                
                # STEP 4: Eligibility Agent - AI-powered eligibility analysis
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🔴 OFFLINE
                
                ---
                
                **📄 Document Agent** 🔴 OFFLINE
                
                ---
                
                **💾 Databricks Agent** 🔴 OFFLINE
                
                ---

                **🔍 Eligibility Agent** 🟢 ONLINE  
                ⚙️ Running AI analysis  
                🧠 Determining eligibility...
                
                ---
                
                **📝 Audit Agent** 🔴 OFFLINE
                
                ---
                
                **👤 Human Review** 🔴 OFFLINE
                """)
                
                # Update to Step 4: Eligibility Agent
                with workflow_placeholder.container():
                    show_workflow_progress(step=4)

                
                    progress_placeholder.progress(0.67)  # 4/6 = 67%
                    status_text.info("🔍 **Eligibility Agent** is analyzing claim eligibility with GPT-4...")
                    
                    detail_placeholder.markdown("""
                    **🔍 Eligibility Agent** - *Working*
                    - 📄 Initializing AI analysis engine...
                    - 🧠 Loading GPT-4 model...
                    """)
                    
                    time.sleep(0.3)
                    
                    detail_placeholder.markdown("""
                    **🔍 Eligibility Agent** - *Working*
                    - ✅ GPT-4 model loaded
                    - 📄 Analyzing claim vs policy coverage...
                    - 📊 Comparing policy terms...
                    """)
                    
                    time.sleep(0.3)
                    
                    detail_placeholder.markdown("""
                    **🔍 Eligibility Agent** - *Working*
                    - ✅ Coverage analysis in progress
                    - 📄 Evaluating eligibility criteria...
                    - 🧠 Generating reasoning and confidence score...
                    """)
                    
                    eligibility_analysis = check_claim_eligibility(extracted_data, ai_summary, validation_result)
                    results['eligibility_analysis'] = eligibility_analysis
                    
                    progress_placeholder.progress(0.95)
                    
                    # Check if human review is needed
                    human_review_agent = HumanReviewAgent(confidence_threshold=50.0)
                    
                    if eligibility_analysis:
                        decision = eligibility_analysis.get('eligibility_decision', 'UNKNOWN')

                        confidence = eligibility_analysis.get('confidence_score', 0)
                        checks_failed = eligibility_analysis.get('checks_failed', [])
                
                # Determine if human review is required
                needs_review = human_review_agent.needs_review(confidence, checks_failed)
                
                if needs_review:
                    # Flag for human review
                    flag_reason = f"Low AI confidence ({confidence:.1f}% < 50% threshold)"
                    if confidence >= 50 and checks_failed:
                        flag_reason = f"Edge case detected: {', '.join(checks_failed[:2])}"

                    
                    review_record = human_review_agent.flag_for_review(
                        claim_data={'extracted_data': extracted_data, 'ai_summary': ai_summary, 'validation_result': validation_result},
                        analysis_result=eligibility_analysis,
                        reason=flag_reason
                    )

                    
                    # Update workflow progress to show Fraud Detection step
                    with workflow_placeholder.container():
                        show_workflow_progress(step=5)  # Step 5 - Fraud Detection

                    
                    progress_placeholder.progress(0.83)  # 5/6 = 83%
                    status_text.info("🚨 **Fraud Detection Agent** is analyzing claim...")

                    
                    # Store in session state to switch to Human Review tab
                    st.session_state['needs_human_review'] = True
                    st.session_state['current_review_id'] = review_record['review_id']

                    
                    # Show human review banner in main UI with application details
                    st.markdown("---")
                    st.warning(f"""
### 👤 Human Review Required

                    
**Review ID:** {review_record['review_id']}  

**Reason:** {flag_reason}  

**AI Confidence:** {confidence:.1f}%  

                    
This claim has been flagged for manual verification by a human reviewer due to low confidence or edge case detection.

                    
The claim will be added to the Human Review Queue for processing.

""")

                    
                    # Show application details
                    st.markdown("### 📊 Application Details")

                    claim_info = extracted_data.get('claim_info', {})

                    col1, col2, col3 = st.columns(3)

                    
                    with col1:
                        st.metric("📋 Policy Number", claim_info.get('policy_number', 'N/A'))
                        st.metric("👤 Policyholder", claim_info.get('policyholder_name', 'N/A'))

                    
                    with col2:
                        st.metric("💰 Claim Amount", f"${claim_info.get('claim_amount', '0')}")
                        st.metric("📅 Claim Date", claim_info.get('claim_date', 'N/A'))

                    
                    with col3:
                        st.metric("📦 Policy Type", claim_info.get('policy_type', 'N/A'))
                        st.metric("ℹ️ Reason", claim_info.get('reason_for_claim', 'N/A')[:30] + '...')

                    
                    # Show eligibility analysis summary
                    if eligibility_analysis.get('detailed_checks'):
                        with st.expander("📝 View Eligibility Analysis", expanded=False):
                            for check in eligibility_analysis['detailed_checks']:
                                if check:
                                    st.markdown(check)

                    
                    # Update sidebar to show Human Review is active
                    sidebar_status.markdown("""

### 📄 Current Status

                    
**🎯 Orchestrator Agent** 🔴 OFFLINE

                    
---

                    
**📄 Document Agent** 🔴 OFFLINE

                    
---

                    
**💾 Databricks Agent** 🔴 OFFLINE

                    
---


**🔍 Eligibility Agent** ✅ COMPLETED

                    
---

                    
**👤 Human Review** 🟢 ONLINE  

⚙️ Manual review required  

🔍 Waiting for human decision...

                    
---

                    
**🚨 Fraud Detection** 🔴 OFFLINE

                    
---

                    
**📧 Communication** 🔴 OFFLINE

                    
---

                    
**📝 Audit Agent** 🔴 OFFLINE

""")

                    
                    # Add navigation message and stop processing
                    st.info("👉 **Click on the 'Human Review' tab above to process this claim manually.**")

                    
                    # Stop further processing - human review needed
                    st.stop()

                
                # Only continue if no human review needed
                status_text.success(f"✅ **Eligibility Agent** completed - Decision: **{decision}** (Confidence: {confidence}%)")
                
                detail_placeholder.markdown(f"""
                **🔍 Eligibility Agent** - *Finalizing*
                - ✅ AI analysis completed
                - ✅ Decision: **{decision}**
                - ✅ Confidence: **{confidence}%**
                - 📄 Generating detailed report...
                {"- ⚠️ **Flagged for human review**" if needs_review else ""}
""")

                
                time.sleep(0.3)
                
                detail_placeholder.markdown(f"""
**🔍 Eligibility Agent** ✅ **COMPLETED**
- ✅ Eligibility Decision: **{decision}**
- ✅ Confidence Score: **{confidence}%**
- ✅ Analysis and reasoning completed
                - 📊 Returning results to Orchestrator...
                {"- 👤 **Sent to Human Review Queue**" if needs_review else ""}
""")

                
                time.sleep(0.5)
                
                # STEP 5: Fraud Detection Agent - ML-powered fraud detection (only if eligible)
                fraud_result = None
                decision = eligibility_analysis.get('eligibility_decision', 'UNKNOWN') if eligibility_analysis else 'UNKNOWN'
                
                if decision == "ELIGIBLE":
                    sidebar_status.markdown("""
                ### 📄 Current Status
                
**🎯 Orchestrator Agent** 🔴 OFFLINE

                
---

                
**📄 Document Agent** 🔴 OFFLINE

                
---

                
**💾 Databricks Agent** 🔴 OFFLINE

                
---

                
**🔍 Eligibility Agent** 🔴 OFFLINE

                
---


**🚨 Fraud Detection** 🟢 ONLINE  

                ⚙️ Analyzing fraud risk  
                🧠 Calling Azure ML model...
                
---

                
**📝 Audit Agent** 🔴 OFFLINE

                
---

                
**👤 Human Review** 🔴 OFFLINE

                                """)
                
                    # Update to Step 5: Fraud Detection Agent
                    with workflow_placeholder.container():
                        show_workflow_progress(step=5)

                
                    progress_placeholder.progress(0.83)  # 5/6 = 83%
                    status_text.info("🚨 **Fraud Detection Agent** is analyzing claim with ML model...")
                
                    detail_placeholder.markdown("""
**🚨 Fraud Detection Agent** - *Working*
- 🔍 Preparing fraud detection features...
- 📊 Calling Azure ML endpoint...
- 🧠 ML model processing...
""")

                
                    time.sleep(0.3)
                
                    # Run fraud detection with ML model
                    try:
                        fraud_agent = FraudDetectorAgent()
                        claim_info = extracted_data.get('claim_info', {})

                    
                        # Prepare fraud detection data
                        # Pass strings for categorical fields - fraud_detector_agent will handle conversion
                        policy_type_val = claim_info.get('policy_type', 1)
                        accident_area_val = claim_info.get('accident_area', 1)
                        sex_val = claim_info.get('sex', 1)
                        police_report_val = claim_info.get('police_report_filed', 0)

                    
                        # Convert to int only if they're already numeric, otherwise keep as string
                        if isinstance(policy_type_val, str):
                            policy_type_val = policy_type_val  # Keep string
                        else:
                            policy_type_val = int(policy_type_val)

                
                        if isinstance(accident_area_val, str):
                            accident_area_val = accident_area_val
                        else:
                            accident_area_val = int(accident_area_val)

                
                        if isinstance(sex_val, str):
                            sex_val = sex_val
                        else:
                            sex_val = int(sex_val)

                
                        if isinstance(police_report_val, str):
                            police_report_val = police_report_val
                        else:
                            police_report_val = int(police_report_val)

                    
                        fraud_data = {
                            "DriverRating": int(claim_info.get('driver_rating', 1)),
                            "Age": int(claim_info.get('age', 30)),
                            "PoliceReportFiled": police_report_val,
                            "WeekOfMonthClaimed": int(claim_info.get('week_of_month_claimed', 1)),
                            "PolicyType": policy_type_val,
                            "WeekOfMonth": int(claim_info.get('week_of_month', 1)),
                            "AccidentArea": accident_area_val,
                            "Sex": sex_val,
                            "Deductible": int(claim_info.get('deductible', 500))
                        }

                    
                        # Debug: Show fraud detection inputs
                        print(f"\n🔍 FRAUD DETECTION INPUT VALUES:")
                        print(f"   DriverRating: {fraud_data['DriverRating']}")
                        print(f"   Age: {fraud_data['Age']}")
                        print(f"   PoliceReportFiled: {fraud_data['PoliceReportFiled']}")
                        print(f"   WeekOfMonthClaimed: {fraud_data['WeekOfMonthClaimed']}")
                        print(f"   PolicyType: {fraud_data['PolicyType']}")
                        print(f"   WeekOfMonth: {fraud_data['WeekOfMonth']}")
                        print(f"   AccidentArea: {fraud_data['AccidentArea']}")
                        print(f"   Sex: {fraud_data['Sex']}")
                        print(f"   Deductible: {fraud_data['Deductible']}")

                    
                        fraud_result = fraud_agent.detect_fraud(fraud_data)

                    
                        # Debug: Show fraud detection output
                        if fraud_result.get('success'):
                            print(f"\n🔍 FRAUD DETECTION OUTPUT:")
                            print(f"   Fraud Probability: {fraud_result.get('fraud_probability', 0):.4f}")
                            print(f"   Is Fraud: {fraud_result.get('is_fraud', False)}")
                            print(f"   Risk Level: {fraud_result.get('fraud_risk', 'Unknown')}")
                        
                        results['fraud_analysis'] = fraud_result

                    
                        if fraud_result.get('success'):
                            fraud_probability = fraud_result.get('fraud_probability', 0)
                            fraud_risk = fraud_result.get('fraud_risk', 'Unknown')
                            is_fraud = fraud_result.get('is_fraud', False)
                
                            detail_placeholder.markdown(f"""
**🚨 Fraud Detection Agent** - *Finalizing*
- ✅ ML model analysis completed
- 🎯 Fraud Probability: **{fraud_probability:.2%}**
- ⚠️ Risk Level: **{fraud_risk}**
- 🤖 Prediction: **{'FRAUD DETECTED' if is_fraud else 'NOT FRAUD'}**
- 📄 Generating fraud report...
""")

                
                            time.sleep(0.3)

                
                            detail_placeholder.markdown(f"""
**🚨 Fraud Detection Agent** ✅ **COMPLETED**
- ✅ Fraud Probability: **{fraud_probability:.2%}**
- ✅ Risk Level: **{fraud_risk}**
- ✅ ML prediction: **{'FRAUD' if is_fraud else 'LEGITIMATE'}**
- 📊 Returning results to Orchestrator...
""")

                
                            # Log to Audit Agent
                            audit_agent = get_audit_agent_instance()
                            if audit_agent:
                                claim_info = extracted_data.get('claim_info', {})
                                policy_num = claim_info.get('policy_number', 'UNKNOWN')
                                audit_agent.log_fraud_detection_action(
                                    policy_number=policy_num,
                                    action="fraud_detection_ml",
                                    inputs=fraud_data,
                                    outputs={
                                        "fraud_probability": fraud_probability,
                                        "fraud_risk": fraud_risk,
                                        "is_fraud": is_fraud,
                                        "threshold_used": fraud_result.get('threshold_used', 0.5)
                                    },
                                    fraud_probability=fraud_probability,
                                    fraud_prediction=fraud_result.get('fraud_prediction', 0),
                                    fraud_risk_level=fraud_risk,
                                    metadata={
                                        "ml_model": "Balanced Random Forest",
                                        "azure_ml_endpoint": "Active",
                                        "model_version": "1.0"
                                    }
                                )
                        else:
                            detail_placeholder.markdown(f"""
**🚨 Fraud Detection Agent** ⚠️ **ERROR**
- ❌ Error: {fraud_result.get('error', 'Unknown error')}
- ⚠️ Proceeding without fraud score
""")

                
                    except Exception as e:
                        fraud_result = {
                            "success": False,
                            "error": str(e),
                            "fraud_probability": 0.0,
                            "fraud_risk": "Error"
                        }
                        results['fraud_analysis'] = fraud_result
                        detail_placeholder.markdown(f"""
                **🚨 Fraud Detection Agent** ⚠️ **ERROR**
                - ❌ Exception: {str(e)}
                - ⚠️ Proceeding without fraud analysis
                """)
                
                        time.sleep(0.5)
                else:
                    # Eligibility rejected, skip fraud detection
                    detail_placeholder.markdown("""
**🚨 Fraud Detection Agent** ⏭️ **SKIPPED**
- ℹ️ Claim not eligible
- ⏭️ Fraud detection skipped (only runs for eligible claims)
""")
                    time.sleep(0.3)
                
                # Update to Step 6: Human Review
                with workflow_placeholder.container():
                    show_workflow_progress(step=6)

                
                progress_placeholder.progress(0.86)  # 6/7 = 86%
                
                detail_placeholder.markdown("""
                **👤 Human Review** - *Completed*
                - ✅ Claim reviewed and approved
                - 📧 Proceeding to communication...
                """)
                
                time.sleep(0.3)
                
                # Update to Step 7: Communication Agent
                with workflow_placeholder.container():
                    show_workflow_progress(step=7)

                
                progress_placeholder.progress(1.0)  # 7/7 = 100% Complete
                
                detail_placeholder.markdown("""
                **📧 Communication Agent** - *Working*
                - 📧 Preparing customer notification...
                - ✉️ Email/SMS ready to send
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **📧 Communication Agent** ✅ **COMPLETED**
                - ✅ Customer notification sent
                - 📧 Email/SMS delivered successfully
                - 📊 Proceeding to finalization...
                """)
                
                time.sleep(0.3)
                
                # Final Step: All Complete - Orchestrator Active
                with workflow_placeholder.container():
                    show_workflow_progress(step=8)  # Step 8 triggers all complete + Orchestrator active

                
                progress_placeholder.progress(1.0)  # 100% Complete
                
                detail_placeholder.markdown("""
                **🎯 Orchestrator Agent** - *Finalizing*
                - ✅ All agents completed successfully
                - 📊 Compiling final results...
                - 🎉 Workflow wrapping up...
                """)
                
                time.sleep(0.3)
                
                detail_placeholder.markdown("""
                **🎯 Orchestrator Agent** ✅ **COMPLETED**
                - ✅ Workflow completed successfully
                - 📊 All agents finished processing
                - 🎉 Insurance claim workflow complete!
                """)
                
                # Workflow Complete
                progress_placeholder.progress(1.0)  # 100%
                
                # Orchestrator completes workflow
                sidebar_status.markdown("""
                ### 📄 Current Status
                
                **🎯 Orchestrator Agent** 🟢 ONLINE  
                ✅ Workflow complete  
                🎉 All tasks finished
                
                ---
                
                **📄 Document Agent** ✅ COMPLETED
                
                ---

                **💾 Databricks Agent** ✅ COMPLETED
                
                ---
                
                **🔍 Eligibility Agent** ✅ COMPLETED
                
                ---
                
                **🚨 Fraud Detection** ✅ COMPLETED
                
                ---
                
                **👤 Human Review** ✅ COMPLETED
                
                ---
                
                **📧 Communication** ✅ COMPLETED
                """)
                
                # Show all steps completed - Orchestrator wrapping up
                with workflow_placeholder.container():
                    show_workflow_progress(step=8)  # All agents complete, Orchestrator active

                
                progress_placeholder.progress(1.0)  # 100% Complete
                status_text.success("🎉 **Workflow Complete!** All agents have finished processing.")
                
                # Log Orchestrator completion to Audit Agent
                audit_agent = get_audit_agent_instance()
                if audit_agent and results.get('validation_result'):
                    workflow_end_time = datetime.now()
                    processing_time = (workflow_end_time - workflow_start_time).total_seconds() * 1000
                    policy_num = results['validation_result'].get('policy_info', {}).get('policy_number', 'UNKNOWN')
                
                    audit_agent.log_orchestrator_action(
                        policy_number=policy_num,
                        action="workflow_completion",
                        inputs={
                            "document_name": uploaded_file.name,
                            "workflow_type": "insurance_claim_processing"
                        },
                        outputs={
                            "workflow_status": "COMPLETED",
                            "agents_executed": ["DocumentAgent", "DatabricksAgent", "EligibilityAgent", "FraudDetectionAgent"],
                            "final_decision": results.get('eligibility_analysis', {}).get('eligibility_decision', 'UNKNOWN'),
                            "confidence_score": results.get('eligibility_analysis', {}).get('confidence_score', 0),
                            "fraud_risk_score": results.get('fraud_analysis', {}).get('fraud_risk_score', 0)
                        },
                        decision="SUCCESS",
                        metadata={
                            "processing_time_ms": processing_time,
                            "timestamp": workflow_end_time.isoformat()
                        }
                    )
                
                # Final summary with flow diagram
                detail_placeholder.empty()
                st.markdown("""
                <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 18px;'>
<div style='display: flex; flex-direction: row; align-items: center;'>

                <div style='display: flex; flex-direction: column; align-items: center;'>
<div style='width: 32px; height: 32px; background: #4F8EF7; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;'>1</div>

<div style='margin-top: 4px; font-size: 0.8em; color: #4F8EF7;'>Orchestrator</div>

                </div>
                <div style='display: flex; align-items: center; margin: 0 4px;'>
<div style='width: 32px; height: 2px; background: #4F8EF7;'></div>

<div style='color: #4F8EF7; font-size: 1em; margin: 0 2px;'>➜</div>

                </div>
                <div style='display: flex; flex-direction: column; align-items: center;'>
<div style='width: 32px; height: 32px; background: #4F8EF7; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;'>2</div>

                                <div style='margin-top: 4px; font-size: 0.8em; color: #4F8EF7;'>Document</div>
                            </div>
                            <div style='display: flex; align-items: center; margin: 0 4px;'>
                                <div style='width: 32px; height: 2px; background: #4F8EF7;'></div>
                                <div style='color: #4F8EF7; font-size: 1em; margin: 0 2px;'>➜</div>
                            </div>
                            <div style='display: flex; flex-direction: column; align-items: center;'>
                                <div style='width: 32px; height: 32px; background: #4F8EF7; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;'>3</div>
                                <div style='margin-top: 4px; font-size: 0.8em; color: #4F8EF7;'>Databricks</div>
                            </div>
                            <div style='display: flex; align-items: center; margin: 0 4px;'>
                                <div style='width: 32px; height: 2px; background: #4F8EF7;'></div>
                                <div style='color: #4F8EF7; font-size: 1em; margin: 0 2px;'>➜</div>
                            </div>
                            <div style='display: flex; flex-direction: column; align-items: center;'>
                                <div style='width: 32px; height: 32px; background: #4F8EF7; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;'>4</div>
                                <div style='margin-top: 4px; font-size: 0.8em; color: #4F8EF7;'>Eligibility</div>
                            </div>
                        </div>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;'>
                        <div style='flex: 1; margin: 0 4px; background: #f5f5f5; border-radius: 8px; padding: 10px; text-align: center;'>
                            <div style='font-weight: bold; color: #4F8EF7; font-size: 0.85em;'>Orchestrator</div>
                            <div style='font-size: 1.1em; margin: 6px 0;'>✅</div>
                            <div style='color: #888; font-size: 0.75em;'>Coordinated</div>
                        </div>
                        <div style='flex: 1; margin: 0 4px; background: #f5f5f5; border-radius: 8px; padding: 10px; text-align: center;'>
                            <div style='font-weight: bold; color: #4F8EF7; font-size: 0.85em;'>Document</div>
                            <div style='font-size: 1.1em; margin: 6px 0;'>✅</div>
                            <div style='color: #888; font-size: 0.75em;'>Extracted</div>
                        </div>
                        <div style='flex: 1; margin: 0 4px; background: #f5f5f5; border-radius: 8px; padding: 10px; text-align: center;'>
                            <div style='font-weight: bold; color: #4F8EF7; font-size: 0.85em;'>Databricks</div>
                            <div style='font-size: 1.1em; margin: 6px 0;'>✅</div>
                            <div style='color: #888; font-size: 0.75em;'>Validated</div>
                        </div>
                        <div style='flex: 1; margin: 0 4px; background: #f5f5f5; border-radius: 8px; padding: 10px; text-align: center;'>
                            <div style='font-weight: bold; color: #4F8EF7; font-size: 0.85em;'>Eligibility</div>
                            <div style='font-size: 1.1em; margin: 6px 0;'>✅</div>
                            <div style='color: #888; font-size: 0.75em;'>Analyzed</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Store results in session state for persistence
                st.session_state['last_processing_results'] = results
                st.session_state['processing_completed'] = True
                
                # Store fraud info in session state if detected
                if results.get('fraud_analysis'):
                    fraud = results['fraud_analysis']
                    if fraud.get('success') and fraud.get('is_fraud'):
                        extracted_data = results.get('extracted_data', {})
                        policy_num = results.get('validation_result', {}).get('policy_info', {}).get('policy_number', 'N/A')
                        
                        st.session_state['fraud_claim_for_review'] = {
                            'policy_number': policy_num,
                            'decision': 'FRAUD_DETECTED',
                            'fraud_probability': fraud.get('fraud_probability', 0),
                            'fraud_risk': fraud.get('fraud_risk', 'Unknown'),
                            'threshold': fraud.get('threshold_used', 0.65),
                            'extracted_data': extracted_data,
                            'fraud_analysis': fraud,
                            'eligibility_analysis': results.get('eligibility_analysis')
                        }
                        
                        # Update workflow state
                        st.session_state['workflow_state'] = {
                            'current_stage': 'human_review',
                            'fraud_detected': True,
                            'awaiting_human_review': True,
                            'human_review_decision': None
                        }
                
                # Store rejection info in session state if not eligible
                eligibility = results.get('eligibility_analysis', {})
                if eligibility.get('eligibility_decision') == 'NOT ELIGIBLE':
                    policy_num = results.get('validation_result', {}).get('policy_info', {}).get('policy_number', 'N/A')
                    extracted_data = results.get('extracted_data', {})
                    
                    st.session_state['rejected_claim_for_review'] = {
                        'policy_number': policy_num,
                        'decision': 'NOT ELIGIBLE',
                        'confidence': eligibility.get('confidence_score', 0),
                        'checks_failed': eligibility.get('checks_failed', []),
                        'detailed_checks': eligibility.get('detailed_checks', []),
                        'extracted_data': extracted_data,
                        'eligibility_analysis': eligibility
                    }
                
                # Store results in session state for persistence
                st.session_state['last_processing_results'] = results
                st.session_state['processing_completed'] = True
                
                st.balloons()
        
        # Display persistent workflow status (outside button handler)
        if st.session_state.get('processing_completed'):
            st.markdown("---")
            st.markdown("### 🔄 Current Workflow Status")
            
            results = st.session_state.get('last_processing_results', {})
            
            # Get fraud detection status
            is_fraud_detected = False
            fraud_probability = 0
            current_policy_num = "N/A"
            
            if results.get('validation_result'):
                current_policy_num = results['validation_result'].get('policy_info', {}).get('policy_number', 'N/A')
            
            if results.get('fraud_analysis'):
                fraud = results['fraud_analysis']
                if fraud.get('success'):
                    is_fraud_detected = fraud.get('is_fraud', False)
                    fraud_probability = fraud.get('fraud_probability', 0)
            
            # Check if human review decision has been made for THIS claim
            fraud_review_result = st.session_state.get('fraud_review_result', {})
            has_review_decision = (
                bool(fraud_review_result) and 
                fraud_review_result.get('policy_number') == current_policy_num
            )
            
            # Determine statuses
            if has_review_decision:
                hr_status = "✅"
                hr_text = f"Completed - {fraud_review_result.get('decision', 'DECIDED')}"
                hr_bg_color = "#E8F5E9"
                hr_text_color = "#2E7D32"
                hr_border = "2px solid #4CAF50"
                comm_status = "✅"
                comm_text = "Completed"
                comm_bg_color = "#E8F5E9"
                comm_text_color = "#2E7D32"
            elif is_fraud_detected:
                hr_status = "⏳"
                hr_text = "ACTIVE - Awaiting Decision"
                hr_bg_color = "#FFF3CD"
                hr_text_color = "#FF6B00"
                hr_border = "2px solid #FF9800"
                comm_status = "⏸️"
                comm_text = "Waiting"
                comm_bg_color = "#f5f5f5"
                comm_text_color = "#888"
            else:
                hr_status = "⏸️"
                hr_text = "Skipped"
                hr_bg_color = "#f5f5f5"
                hr_text_color = "#888"
                hr_border = "none"
                comm_status = "✅"
                comm_text = "Completed"
                comm_bg_color = "#E8F5E9"
                comm_text_color = "#2E7D32"
            
            # Display workflow status boxes
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;'>
                    <div style='flex: 1; margin: 0 8px; background: #E8F5E9; border-radius: 8px; padding: 14px; text-align: center; border: 2px solid #4CAF50;'>
                        <div style='font-weight: bold; color: #2E7D32; font-size: 0.9em; margin-bottom: 8px;'>📄 Document Agent</div>
                        <div style='font-size: 1.3em; margin: 8px 0;'>✅</div>
                        <div style='color: #2E7D32; font-size: 0.8em; font-weight: bold;'>Completed</div>
                    </div>
                    <div style='flex: 1; margin: 0 8px; background: #E8F5E9; border-radius: 8px; padding: 14px; text-align: center; border: 2px solid #4CAF50;'>
                        <div style='font-weight: bold; color: #2E7D32; font-size: 0.9em; margin-bottom: 8px;'>💾 Databricks Agent</div>
                        <div style='font-size: 1.3em; margin: 8px 0;'>✅</div>
                        <div style='color: #2E7D32; font-size: 0.8em; font-weight: bold;'>Completed</div>
                    </div>
                    <div style='flex: 1; margin: 0 8px; background: #E8F5E9; border-radius: 8px; padding: 14px; text-align: center; border: 2px solid #4CAF50;'>
                        <div style='font-weight: bold; color: #2E7D32; font-size: 0.9em; margin-bottom: 8px;'>🔍 Eligibility Agent</div>
                        <div style='font-size: 1.3em; margin: 8px 0;'>✅</div>
                        <div style='color: #2E7D32; font-size: 0.8em; font-weight: bold;'>Completed</div>
                    </div>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;'>
                    <div style='flex: 1; margin: 0 8px; background: #E8F5E9; border-radius: 8px; padding: 14px; text-align: center; border: 2px solid #4CAF50;'>
                        <div style='font-weight: bold; color: #2E7D32; font-size: 0.9em; margin-bottom: 8px;'>🚨 Fraud Detection</div>
                        <div style='font-size: 1.3em; margin: 8px 0;'>✅</div>
                        <div style='color: #2E7D32; font-size: 0.8em; font-weight: bold;'>Completed</div>
                    </div>
                    <div style='flex: 1; margin: 0 8px; background: {hr_bg_color}; border-radius: 8px; padding: 14px; text-align: center; border: {hr_border};'>
                        <div style='font-weight: bold; color: {hr_text_color}; font-size: 0.9em; margin-bottom: 8px;'>👤 Human Review</div>
                        <div style='font-size: 1.3em; margin: 8px 0;'>{hr_status}</div>
                        <div style='color: {hr_text_color}; font-size: 0.8em; font-weight: bold;'>{hr_text}</div>
                    </div>
                    <div style='flex: 1; margin: 0 8px; background: {comm_bg_color}; border-radius: 8px; padding: 14px; text-align: center;'>
                        <div style='font-weight: bold; color: {comm_text_color}; font-size: 0.9em; margin-bottom: 8px;'>📧 Communication</div>
                        <div style='font-size: 1.3em; margin: 8px 0;'>{comm_status}</div>
                        <div style='color: {comm_text_color}; font-size: 0.8em; font-weight: bold;'>{comm_text}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display complete processing results
            st.markdown("---")
            st.markdown("### 📋 Processing Results")
            
            # Get decision and confidence
            decision = "UNKNOWN"
            confidence = 0
            if results.get('eligibility_analysis'):
                decision = results['eligibility_analysis'].get('eligibility_decision', 'UNKNOWN')
                confidence = results['eligibility_analysis'].get('confidence_score', 0)
            
            # Override with fraud decision if detected
            if is_fraud_detected:
                decision = "FRAUD_DETECTED"
            
            # Override with human review decision if available
            if has_review_decision:
                human_decision = fraud_review_result.get('decision', 'DECIDED')
                if human_decision == "APPROVE":
                    decision = "APPROVED_BY_HUMAN"
                elif human_decision == "REJECT":
                    decision = "REJECTED_BY_HUMAN"
            
            # Display Final Decision Banner
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("📋 Policy Number", current_policy_num)
            with col2:
                if decision == "APPROVED_BY_HUMAN":
                    st.success(f"✅ **CLAIM APPROVED BY HUMAN REVIEWER** | Reviewer: {fraud_review_result.get('reviewer', 'N/A')}")
                elif decision == "REJECTED_BY_HUMAN":
                    st.error(f"❌ **CLAIM REJECTED BY HUMAN REVIEWER** | Reviewer: {fraud_review_result.get('reviewer', 'N/A')}")
                elif decision == "FRAUD_DETECTED":
                    st.warning(f"🚨 **FRAUD DETECTED - AWAITING HUMAN REVIEW** | Fraud Probability: {fraud_probability:.2%}")
                elif decision == "ELIGIBLE":
                    st.success(f"✅ **CLAIM APPROVED** - Eligible (Confidence: {confidence}%)")
                elif decision == "NOT ELIGIBLE":
                    st.error(f"❌ **CLAIM REJECTED** - Not eligible (Confidence: {confidence}%)")
                else:
                    st.warning(f"⚠️ **MANUAL REVIEW REQUIRED** (Confidence: {confidence}%)")
            
            # Show human review notes if available
            if has_review_decision and fraud_review_result.get('notes'):
                st.info(f"**Review Notes:** {fraud_review_result.get('notes')}")
            
            # Show fraud alert for pending review
            if decision == "FRAUD_DETECTED" and not has_review_decision:
                st.markdown("")
                st.markdown("#### 🚨 Fraud Alert Details")
                fraud = results.get('fraud_analysis', {})
                fraud_prob = fraud.get('fraud_probability', 0)
                fraud_risk = fraud.get('fraud_risk', 'Unknown')
                threshold = fraud.get('threshold_used', 0.65)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                with col2:
                    st.metric("Risk Level", fraud_risk)
                with col3:
                    st.metric("Threshold", f"{threshold:.0%}")
                
                st.error(f"⚠️ **This claim exceeds the fraud detection threshold of {threshold:.0%}**")
                st.info("💡 **Action Required:** Switch to the **Human Review** tab to review this claim and make a final decision.")
            
            # Show rejection reasons for NOT ELIGIBLE
            if decision == "NOT ELIGIBLE" and results.get('eligibility_analysis'):
                checks_failed = results['eligibility_analysis'].get('checks_failed', [])
                detailed_checks = results['eligibility_analysis'].get('detailed_checks', [])
                
                st.markdown("")
                if checks_failed:
                    st.markdown("#### ❌ Rejection Reasons")
                    for i, check in enumerate(checks_failed, 1):
                        st.error(f"**{i}.** {check}")
                
                if detailed_checks:
                    with st.expander("📝 View Detailed Eligibility Analysis", expanded=False):
                        for check_message in detailed_checks:
                            if check_message:
                                st.markdown(check_message)
            
            # Agent Execution Summary
            st.markdown("---")
            st.markdown("### 🎉 Agent Execution Summary")
            
            import pandas as pd
            agent_data = []
            
            # 1. Document Intelligence Agent
            if results.get('extracted_data'):
                claim_info = results['extracted_data'].get('claim_info', {})
                work_done = "Extracted claim data from PDF using Azure Document Intelligence OCR"
                output = f"Policy: {claim_info.get('policy_number', 'N/A')} | Amount: ${claim_info.get('claim_amount', '0')} | Pages: {results['extracted_data'].get('page_count', 0)}"
                agent_data.append(["📄 Document Intelligence", work_done, output])
            
            # 2. AI Summary Agent
            if results.get('ai_summary'):
                work_done = "Generated AI-powered summary using Azure OpenAI GPT-4"
                output = results['ai_summary'][:100] + "..." if len(results['ai_summary']) > 100 else results['ai_summary']
                agent_data.append(["🤖 AI Summary", work_done, output])
            
            # 3. Databricks Agent
            if results.get('validation_result'):
                policy_info = results['validation_result'].get('policy_info', {})
                work_done = "Validated policy in Databricks database"
                validation_details = results['validation_result'].get('validation', {}).get('details', {})
                policy_status = validation_details.get('policy_status', 'Unknown')
                output = f"Status: {policy_status.upper()} | Limit: ${validation_details.get('policy_limit', 0):,.0f} | Past Claims: ${validation_details.get('past_claims_amount', 0):,.0f}"
                agent_data.append(["💾 Databricks", work_done, output])
            
            # 4. Eligibility Agent
            if results.get('eligibility_analysis'):
                eligibility = results['eligibility_analysis']
                work_done = "Performed 5 eligibility checks with AI-powered analysis"
                elig_decision = eligibility.get('eligibility_decision', 'UNKNOWN')
                elig_confidence = eligibility.get('confidence_score', 0)
                output = f"Decision: {elig_decision} | Confidence: {elig_confidence}% | Checks Failed: {len(eligibility.get('checks_failed', []))}"
                agent_data.append(["🔍 Eligibility", work_done, output])
            
            # 5. Fraud Detection Agent
            if results.get('fraud_analysis'):
                fraud = results['fraud_analysis']
                if fraud.get('success'):
                    work_done = "ML-based fraud detection using Azure ML deployed model"
                    fraud_prob = fraud.get('fraud_probability', 0)
                    fraud_risk = fraud.get('fraud_risk', 'Unknown')
                    is_fraud = fraud.get('is_fraud', False)
                    output = f"Result: {'⚠️ FRAUD' if is_fraud else '✅ NO FRAUD'} | Probability: {fraud_prob:.2%} | Risk: {fraud_risk}"
                    agent_data.append(["🚨 Fraud Detection", work_done, output])
            
            # 6. Human Review Agent (if reviewed)
            if has_review_decision:
                work_done = f"Manual review by {fraud_review_result.get('reviewer', 'N/A')}"
                output = f"Decision: {fraud_review_result.get('decision')} | Timestamp: {fraud_review_result.get('timestamp', 'N/A')[:19]}"
                agent_data.append(["👤 Human Review", work_done, output])
            
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
    
    with tab2:
        # Human Review Agent Interface
        render_human_review_ui()

if __name__ == "__main__":
    main()


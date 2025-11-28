"""
Policy Validator Agent
Integrates Document Intelligence with Azure SQL Policy Lookup
"""

import os
import re
from dotenv import load_dotenv
from azure_sql_agent import get_azure_sql_agent
from datetime import datetime

load_dotenv()

class PolicyValidator:
    def __init__(self):
        """Initialize Policy Validator with Azure SQL connection"""
        # Initialize Azure SQL Agent
        self.sql_agent = get_azure_sql_agent()
        
        # Check if Azure SQL is configured and can connect
        if self.sql_agent and self.sql_agent.connect():
            self.enabled = True
            print("✅ Azure SQL Agent connected successfully")
        else:
            self.sql_agent = None
            self.enabled = False
            print("⚠️ Policy validation disabled - Azure SQL not configured")
    
    def extract_policy_info(self, extracted_data, ai_summary):
        """
        Extract policy information from document data and AI summary
        Returns: dict with policy_number, status, etc.
        """
        policy_info = {
            'policy_number': None,
            'claim_status': None,
            'claim_number': None,
            'claim_amount': None
        }
        
        # Handle None extracted_data
        if extracted_data is None:
            print("⚠️ Warning: extracted_data is None, returning empty policy info")
            return policy_info
        
        # PRIORITY 1: Check if AI has already extracted claim_info (most reliable)
        claim_info = extracted_data.get('claim_info', {})
        if claim_info:
            policy_info['policy_number'] = claim_info.get('policy_number')
            policy_info['claim_amount'] = claim_info.get('claim_amount')
            print(f"✅ Using AI-extracted policy number: {policy_info['policy_number']}")
        
        # PRIORITY 2: Search in extracted text (only if not found in claim_info)
        if not policy_info['policy_number']:
            text = extracted_data.get('text', '')
            
            # Look for policy number patterns (more precise patterns)
            policy_patterns = [
                r'Policy\s*(?:Number|No\.?|#)\s*:?\s*([A-Z]{3}\d+)',  # POL12345 format
                r'\b(POL\d{5,})\b',  # POL followed by 5+ digits
                r'Policy\s*:?\s*([A-Z]{3}\d{5,})'  # Policy: POL12345
            ]
            
            for pattern in policy_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    policy_info['policy_number'] = match.group(1).strip()
                    print(f"✅ Extracted policy number from text: {policy_info['policy_number']}")
                    break
        
        # Look for claim status keywords
        status_keywords = {
            'pending': ['pending', 'under review', 'processing', 'submitted'],
            'approved': ['approved', 'accepted', 'granted'],
            'rejected': ['rejected', 'denied', 'declined'],
            'closed': ['closed', 'completed', 'settled']
        }
        
        text = extracted_data.get('text', '')
        text_lower = text.lower()
        for status, keywords in status_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                policy_info['claim_status'] = status
                break
        
        # PRIORITY 3: Look in key-value pairs (only if still not found)
        if not policy_info['policy_number']:
            kv_pairs = extracted_data.get('key_value_pairs', {})
            for key, value in kv_pairs.items():
                key_lower = key.lower()
                if 'policy' in key_lower and 'number' in key_lower:
                    policy_info['policy_number'] = value.strip()
                    print(f"✅ Extracted policy number from key-value: {policy_info['policy_number']}")
                elif 'claim' in key_lower and 'number' in key_lower:
                    policy_info['claim_number'] = value.strip()
                elif 'status' in key_lower:
                    policy_info['claim_status'] = value.strip().lower()
                elif 'amount' in key_lower:
                    policy_info['claim_amount'] = value.strip()
        
        # Also check AI summary for extracted information
        if ai_summary:
            summary_lower = ai_summary.lower()
            if not policy_info['claim_status']:
                for status, keywords in status_keywords.items():
                    if any(keyword in summary_lower for keyword in keywords):
                        policy_info['claim_status'] = status
                        break
        
        return policy_info
    
    def validate_policy_number(self, policy_number):
        """
        Check if policy number exists in database
        Returns: True if exists, False otherwise
        """
        if not self.enabled or not policy_number:
            print(f"⚠️ Validation disabled or no policy number. Enabled: {self.enabled}, Policy: {policy_number}")
            return False
        
        try:
            print(f"🔍 Validating policy number: {policy_number}")
            
            result = self.sql_agent.validate_policy(policy_number)
            
            print(f"🔍 Validation result: {result}")
            
            if result and result.get('success') and result.get('policy_exists'):
                print(f"✅ Policy number {policy_number} EXISTS in database")
                return True
            else:
                print(f"❌ Policy number {policy_number} NOT FOUND")
                return False
        
        except Exception as e:
            print(f"❌ Error checking policy number: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_policy(self, policy_number):
        """
        Check policy in Azure SQL Database
        Returns: dict with validation results
        """
        if not self.enabled or not policy_number:
            return {
                'found': False,
                'message': 'Policy validation not available',
                'details': None
            }
        
        try:
            print(f"🔍 Checking policy {policy_number} in database...")
            
            result = self.sql_agent.get_policy_details(policy_number)
            
            print(f"🔍 Query result: {result}")
            
            if result and result.get('success') and result.get('policy_info'):
                # Azure SQL agent returns policy details
                policy_data = result['policy_info']
                
                print(f"✅ Policy data found: {policy_data}")
                print(f"✅ Policy data type: {type(policy_data)}")
                
                validation_result = {
                    'found': True,
                    'policy_number': policy_number,
                    'policy_data': policy_data,
                    'message': None,
                    'details': policy_data  # Store the full dictionary as details
                }
                
                # Check if policy is active
                policy_status = None
                
                if isinstance(policy_data, dict):
                    # Get status directly from the 'policy_status' key
                    if 'policy_status' in policy_data:
                        policy_status = str(policy_data['policy_status']).lower().strip()
                        print(f"✅ Found policy_status in dict: '{policy_status}'")
                    else:
                        print(f"⚠️ 'policy_status' key not found in dict. Available keys: {list(policy_data.keys())}")
                else:
                    print(f"⚠️ Unexpected data type: {type(policy_data)}")
                
                # Store status in validation result
                validation_result['policy_status'] = policy_status
                
                # Determine message based on status
                if policy_status in ['active', '1', 'true']:
                    validation_result['message'] = "✅ Policy is ACTIVE"
                    validation_result['recommendation'] = "✅ Policy is valid - You can proceed with claim processing"
                    validation_result['alert_level'] = 'success'
                    validation_result['claim_eligible'] = True
                elif policy_status in ['expired', 'inactive', 'cancelled', '0', 'false']:
                    validation_result['message'] = "❌ Policy is EXPIRED/INACTIVE"
                    validation_result['recommendation'] = "🚫 CLAIM REJECTED - Policy has EXPIRED and is NOT ELIGIBLE for insurance claim. Customer must renew policy before submitting claims."
                    validation_result['alert_level'] = 'error'
                    validation_result['claim_eligible'] = False
                elif policy_status is None:
                    validation_result['message'] = "⚠️ Policy found but status field is empty or missing"
                    validation_result['recommendation'] = "⚠️ Manual verification required - Status information not available"
                    validation_result['alert_level'] = 'warning'
                else:
                    validation_result['message'] = f"⚠️ Policy found with unclear status: '{policy_status}'"
                    validation_result['recommendation'] = "⚠️ Manual verification required - Check policy status in system"
                    validation_result['alert_level'] = 'warning'
                
                return validation_result
            else:
                # Policy not found in database
                return {
                    'found': False,
                    'policy_number': policy_number,
                    'message': '❌ Policy NOT FOUND in Database',
                    'recommendation': '🚫 CANNOT PROCESS CLAIM - Policy number does not exist in our records. Please verify the policy number or contact customer.',
                    'alert_level': 'error',
                    'details': None
                }
        
        except Exception as e:
            print(f"❌ Error validating policy: {str(e)}")
            return {
                'found': False,
                'message': f'Error validating policy: {str(e)}',
                'details': None
            }
    
    def process_claim_document(self, extracted_data, ai_summary):
        """
        Complete workflow: Extract policy info and validate
        Returns: dict with all results
        """
        print("\n" + "="*80)
        print("🔍 POLICY VALIDATION WORKFLOW")
        print("="*80)
        
        # Handle None extracted_data
        if extracted_data is None:
            print("⚠️ Warning: extracted_data is None, cannot process claim document")
            return {
                'policy_info': {'policy_number': None, 'claim_status': None, 'claim_number': None, 'claim_amount': None},
                'validation': {'found': False, 'message': 'Document extraction failed'},
                'should_validate': False
            }
        
        # Step 1: Extract policy information from document
        print("\n📄 Step 1: Extracting policy information from document...")
        policy_info = self.extract_policy_info(extracted_data, ai_summary)
        
        print(f"   Policy Number: {policy_info['policy_number'] or 'Not found'}")
        print(f"   Claim Status: {policy_info['claim_status'] or 'Not specified'}")
        
        # Step 2: Check if claim is pending
        result = {
            'policy_info': policy_info,
            'validation': None,
            'should_validate': False
        }
        
        if policy_info['claim_status'] == 'pending':
            print("\n⏳ Step 2: Claim status is PENDING - proceeding to validation...")
            result['should_validate'] = True
            
            # Step 3: Validate policy in database
            if policy_info['policy_number']:
                print(f"\n🔍 Step 3: Validating policy {policy_info['policy_number']} in database...")
                validation = self.validate_policy(policy_info['policy_number'])
                result['validation'] = validation
                
                print(f"\n   {validation.get('message', 'Validation complete')}")
                if validation.get('recommendation'):
                    print(f"   💡 {validation['recommendation']}")
            else:
                print("\n⚠️ Step 3: Cannot validate - Policy number not found in document")
                result['validation'] = {
                    'found': False,
                    'message': '⚠️ Policy number not found in document',
                    'recommendation': '⚠️ Cannot validate policy - Please ensure policy number is clearly visible in the claim document',
                    'alert_level': 'warning',
                    'details': None
                }
        else:
            print(f"\n✋ Step 2: Claim status is '{policy_info['claim_status'] or 'unknown'}' - validation not needed")
            result['should_validate'] = False
        
        print("\n" + "="*80)
        return result


if __name__ == "__main__":
    # Test the policy validator
    validator = PolicyValidator()
    
    if validator.enabled:
        print("✅ Policy Validator initialized successfully")
        
        # Test with a sample policy number
        test_policy = input("\nEnter a policy number to test (or press Enter to skip): ").strip()
        if test_policy:
            result = validator.validate_policy(test_policy)
            print(f"\nValidation Result:")
            print(f"  Found: {result['found']}")
            print(f"  Message: {result.get('message', 'N/A')}")
            if result.get('details'):
                print(f"  Details: {result['details']}")
    else:
        print("❌ Policy Validator not available - check Databricks configuration")

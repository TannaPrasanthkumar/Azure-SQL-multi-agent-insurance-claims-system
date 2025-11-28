"""
Audit Agent - Comprehensive Logging for Transparency and Compliance

Logs all decisions, inputs, and outputs from all agents:
- Orchestrator Agent
- Document Intelligence Agent
- Databricks Agent
- Eligibility Agent
- Human Review Agent

Stores logs in Azure Blob Storage following Responsible AI standards.
Structure: audit-logs/YYYY-MM-DD/policy_number/AgentName_timestamp.json
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import traceback

# Load environment variables
load_dotenv()


class AuditAgent:
    """
    Audit Agent for logging all agent decisions, inputs, and outputs.
    Ensures transparency and compliance with Responsible AI standards.
    """
    
    def __init__(self):
        """Initialize the Audit Agent with Azure Blob Storage connection."""
        self.storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "audit-logs")
        
        # Validate configuration
        if not self.storage_account_name or not self.storage_account_key:
            raise ValueError("Azure Storage credentials not found in .env file")
        
        # Initialize blob service client
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={self.storage_account_name};"
            f"AccountKey={self.storage_account_key};"
            f"EndpointSuffix=core.windows.net"
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Ensure container exists
        self._ensure_container_exists()
        
        print(f"✅ Audit Agent initialized - Logging to: {self.container_name}")
    
    def _ensure_container_exists(self):
        """Ensure the audit-logs container exists in Azure Blob Storage."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except Exception:
            # Container doesn't exist, create it
            try:
                container_client.create_container()
                print(f"✅ Created container: {self.container_name}")
            except Exception as e:
                print(f"⚠️ Could not create container: {e}")
    
    def log_orchestrator_action(
        self,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        decision: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log Orchestrator Agent actions.
        
        Args:
            policy_number: Policy number for the claim
            action: Action performed (e.g., "workflow_initiation", "agent_coordination")
            inputs: Input data received by the orchestrator
            outputs: Output data produced by the orchestrator
            decision: Decision made (e.g., "SUCCESS", "FAILED", "PENDING")
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        return self._log_agent_action(
            agent_name="OrchestratorAgent",
            policy_number=policy_number,
            action=action,
            inputs=inputs,
            outputs=outputs,
            decision=decision,
            metadata=metadata
        )
    
    def log_document_agent_action(
        self,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        decision: str,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log Document Intelligence Agent actions.
        
        Args:
            policy_number: Policy number for the claim
            action: Action performed (e.g., "document_analysis", "ocr_extraction")
            inputs: Input data (document info, pages, etc.)
            outputs: Extracted data from document
            decision: Decision made (e.g., "SUCCESS", "FAILED")
            confidence_score: OCR/extraction confidence score
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if metadata is None:
            metadata = {}
        if confidence_score is not None:
            metadata["confidence_score"] = confidence_score
        
        return self._log_agent_action(
            agent_name="DocumentAgent",
            policy_number=policy_number,
            action=action,
            inputs=inputs,
            outputs=outputs,
            decision=decision,
            metadata=metadata
        )
    
    def log_databricks_agent_action(
        self,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        decision: str,
        query_executed: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log Databricks Agent actions.
        
        Args:
            policy_number: Policy number for the claim
            action: Action performed (e.g., "policy_validation", "query_execution")
            inputs: Input data (query parameters, filters, etc.)
            outputs: Query results from Databricks
            decision: Decision made (e.g., "POLICY_FOUND", "POLICY_NOT_FOUND")
            query_executed: SQL query executed
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if metadata is None:
            metadata = {}
        if query_executed is not None:
            metadata["query_executed"] = query_executed
        
        return self._log_agent_action(
            agent_name="DatabricksAgent",
            policy_number=policy_number,
            action=action,
            inputs=inputs,
            outputs=outputs,
            decision=decision,
            metadata=metadata
        )
    
    def log_eligibility_agent_action(
        self,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        decision: str,
        confidence_score: Optional[float] = None,
        ambiguity_score: Optional[float] = None,
        checks_performed: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log Eligibility Agent actions.
        
        Args:
            policy_number: Policy number for the claim
            action: Action performed (e.g., "eligibility_check", "policy_validation")
            inputs: Input data (claim details, policy data, etc.)
            outputs: Eligibility determination results
            decision: Decision made (e.g., "ELIGIBLE", "NOT_ELIGIBLE")
            confidence_score: Confidence in the eligibility decision
            ambiguity_score: Ambiguity score for the claim
            checks_performed: List of checks performed
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if metadata is None:
            metadata = {}
        if confidence_score is not None:
            metadata["confidence_score"] = confidence_score
        if ambiguity_score is not None:
            metadata["ambiguity_score"] = ambiguity_score
        if checks_performed is not None:
            metadata["checks_performed"] = checks_performed
        
        return self._log_agent_action(
            agent_name="EligibilityAgent",
            policy_number=policy_number,
            action=action,
            inputs=inputs,
            outputs=outputs,
            decision=decision,
            metadata=metadata
        )
    
    def log_fraud_detection_action(
        self,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        fraud_probability: Optional[float] = None,
        fraud_prediction: Optional[int] = None,
        fraud_risk_level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log Fraud Detection Agent actions.
        
        Args:
            policy_number: Policy number for the claim
            action: Action performed (e.g., "fraud_detection", "ml_prediction")
            inputs: Input features sent to ML model
            outputs: Fraud detection results from ML model
            fraud_probability: Probability of fraud (0.0 to 1.0)
            fraud_prediction: Binary fraud prediction (0=Not Fraud, 1=Fraud)
            fraud_risk_level: Risk level (e.g., "Low Risk", "High Risk")
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if metadata is None:
            metadata = {}
        if fraud_probability is not None:
            metadata["fraud_probability"] = fraud_probability
        if fraud_prediction is not None:
            metadata["fraud_prediction"] = fraud_prediction
        if fraud_risk_level is not None:
            metadata["fraud_risk_level"] = fraud_risk_level
        
        return self._log_agent_action(
            agent_name="FraudDetectionAgent",
            policy_number=policy_number,
            action=action,
            inputs=inputs,
            outputs=outputs,
            decision="FRAUD_DETECTED" if fraud_prediction == 1 else "NO_FRAUD",
            metadata=metadata
        )
    
    def log_human_review_action(
        self,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        decision: str,
        reviewer_name: Optional[str] = None,
        review_notes: Optional[str] = None,
        original_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log Human Review Agent actions.
        
        Args:
            policy_number: Policy number for the claim
            action: Action performed (e.g., "manual_review", "decision_override")
            inputs: Input data (claim details, AI recommendation, etc.)
            outputs: Human reviewer's decision and reasoning
            decision: Final decision (e.g., "APPROVED", "REJECTED", "NEEDS_INFO")
            reviewer_name: Name/ID of the human reviewer
            review_notes: Notes provided by the reviewer
            original_confidence: Original AI confidence before review
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        if metadata is None:
            metadata = {}
        if reviewer_name is not None:
            metadata["reviewer_name"] = reviewer_name
        if review_notes is not None:
            metadata["review_notes"] = review_notes
        if original_confidence is not None:
            metadata["original_ai_confidence"] = original_confidence
        
        return self._log_agent_action(
            agent_name="HumanReviewAgent",
            policy_number=policy_number,
            action=action,
            inputs=inputs,
            outputs=outputs,
            decision=decision,
            metadata=metadata
        )
    
    def _log_agent_action(
        self,
        agent_name: str,
        policy_number: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        decision: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Core logging function for all agents.
        
        Args:
            agent_name: Name of the agent (e.g., "DocumentAgent")
            policy_number: Policy number for the claim
            action: Action performed
            inputs: Input data
            outputs: Output data
            decision: Decision made
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully, False otherwise
        """
        try:
            # Generate timestamp
            timestamp = datetime.now()
            timestamp_iso = timestamp.isoformat()
            timestamp_str = timestamp.strftime("%Y%m%dT%H%M%S")
            date_folder = timestamp.strftime("%Y-%m-%d")
            
            # Create audit log structure
            audit_log = {
                "agent_name": agent_name,
                "timestamp": timestamp_iso,
                "policy_number": policy_number,
                "action": action,
                "inputs": inputs,
                "outputs": outputs,
                "decision": decision,
                "metadata": metadata or {},
                "responsible_ai": {
                    "transparency": "All decisions logged for audit trail",
                    "accountability": f"Agent: {agent_name}",
                    "traceability": f"Timestamp: {timestamp_iso}",
                    "compliance": "Full input/output capture for regulatory review"
                }
            }
            
            # Create blob path following structure: date/policy_number/AgentName_timestamp.json
            blob_name = f"{date_folder}/{policy_number}/{agent_name}_{timestamp_str}.json"
            
            # Upload to Azure Blob Storage
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            json_data = json.dumps(audit_log, indent=2, default=str)
            blob_client.upload_blob(json_data, overwrite=True)
            
            print(f"✅ Audit log uploaded: {blob_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to log {agent_name} action: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_audit_trail(
        self,
        policy_number: str,
        date: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> list:
        """
        Retrieve audit trail for a specific policy.
        
        Args:
            policy_number: Policy number to retrieve logs for
            date: Optional date filter (YYYY-MM-DD format)
            agent_name: Optional agent name filter
        
        Returns:
            list: List of audit logs matching the criteria
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # Build prefix for blob listing
            if date:
                prefix = f"{date}/{policy_number}/"
            else:
                prefix = f""
            
            # List all blobs matching the criteria
            blobs = container_client.list_blobs(name_starts_with=prefix)
            
            audit_logs = []
            for blob in blobs:
                # Filter by policy number and optionally by agent name
                if policy_number in blob.name:
                    if agent_name is None or agent_name in blob.name:
                        # Download and parse the blob
                        blob_client = container_client.get_blob_client(blob.name)
                        blob_data = blob_client.download_blob().readall()
                        audit_log = json.loads(blob_data)
                        audit_logs.append(audit_log)
            
            return sorted(audit_logs, key=lambda x: x.get('timestamp', ''))
            
        except Exception as e:
            print(f"❌ Failed to retrieve audit trail: {str(e)}")
            return []
    
    def generate_audit_report(
        self,
        policy_number: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report for a policy.
        
        Args:
            policy_number: Policy number to generate report for
            date: Optional date filter (YYYY-MM-DD format)
        
        Returns:
            dict: Audit report with summary statistics
        """
        audit_logs = self.get_audit_trail(policy_number, date)
        
        if not audit_logs:
            return {
                "policy_number": policy_number,
                "total_logs": 0,
                "message": "No audit logs found"
            }
        
        # Aggregate statistics
        agent_counts = {}
        action_counts = {}
        decision_counts = {}
        
        for log in audit_logs:
            agent = log.get("agent_name", "Unknown")
            action = log.get("action", "Unknown")
            decision = log.get("decision", "Unknown")
            
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        return {
            "policy_number": policy_number,
            "date_range": date or "All dates",
            "total_logs": len(audit_logs),
            "first_log_timestamp": audit_logs[0].get("timestamp") if audit_logs else None,
            "last_log_timestamp": audit_logs[-1].get("timestamp") if audit_logs else None,
            "agent_breakdown": agent_counts,
            "action_breakdown": action_counts,
            "decision_breakdown": decision_counts,
            "logs": audit_logs
        }


# Singleton instance for easy import
_audit_agent_instance = None

def get_audit_agent() -> AuditAgent:
    """
    Get singleton instance of AuditAgent.
    
    Returns:
        AuditAgent: Singleton audit agent instance
    """
    global _audit_agent_instance
    if _audit_agent_instance is None:
        _audit_agent_instance = AuditAgent()
    return _audit_agent_instance


if __name__ == "__main__":
    """Test the Audit Agent with sample data from all agents."""
    print("=" * 70)
    print("AUDIT AGENT TEST - All Agents Logging")
    print("=" * 70)
    
    # Initialize audit agent
    audit_agent = get_audit_agent()
    
    # Test data
    test_policy = "POL2024TEST001"
    test_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nTesting audit logs for policy: {test_policy}")
    print("-" * 70)
    
    # Test 1: Orchestrator Agent
    print("\n[1/5] Testing Orchestrator Agent logging...")
    audit_agent.log_orchestrator_action(
        policy_number=test_policy,
        action="workflow_initiation",
        inputs={
            "claim_document": "claim_form_001.pdf",
            "workflow_type": "standard_claim_processing"
        },
        outputs={
            "workflow_id": "WF_20251112_001",
            "agents_activated": ["DocumentAgent", "DatabricksAgent", "EligibilityAgent"],
            "status": "initiated"
        },
        decision="SUCCESS",
        metadata={"initiated_by": "system", "priority": "normal"}
    )
    
    # Test 2: Document Agent
    print("\n[2/5] Testing Document Agent logging...")
    audit_agent.log_document_agent_action(
        policy_number=test_policy,
        action="document_analysis",
        inputs={
            "document_path": "claim_form_001.pdf",
            "page_count": 3,
            "document_type": "claim_form"
        },
        outputs={
            "extracted_fields": {
                "policy_number": test_policy,
                "claim_amount": 15000.00,
                "claim_date": "2025-11-10",
                "claimant_name": "John Doe",
                "claim_reason": "Medical expenses"
            },
            "pages_analyzed": 3
        },
        decision="SUCCESS",
        confidence_score=0.95,
        metadata={"ocr_engine": "Azure Document Intelligence", "processing_time_ms": 2340}
    )
    
    # Test 3: Databricks Agent
    print("\n[3/5] Testing Databricks Agent logging...")
    audit_agent.log_databricks_agent_action(
        policy_number=test_policy,
        action="policy_validation",
        inputs={
            "policy_number": test_policy,
            "query_type": "policy_details"
        },
        outputs={
            "policy_found": True,
            "policy_status": "Active",
            "policy_type": "Health Insurance",
            "coverage_limit": 50000.00,
            "expiry_date": "2026-12-31"
        },
        decision="POLICY_FOUND",
        query_executed="SELECT * FROM policies WHERE policy_number = 'POL2024TEST001'",
        metadata={"query_time_ms": 145, "records_found": 1}
    )
    
    # Test 4: Eligibility Agent
    print("\n[4/5] Testing Eligibility Agent logging...")
    audit_agent.log_eligibility_agent_action(
        policy_number=test_policy,
        action="eligibility_check",
        inputs={
            "claim_amount": 15000.00,
            "claim_date": "2025-11-10",
            "policy_status": "Active",
            "coverage_limit": 50000.00,
            "expiry_date": "2026-12-31"
        },
        outputs={
            "eligibility_status": "ELIGIBLE",
            "checks_passed": 5,
            "checks_failed": 0,
            "recommendation": "Approve claim - all checks passed"
        },
        decision="ELIGIBLE",
        confidence_score=0.92,
        ambiguity_score=8,
        checks_performed=[
            "Policy status check - PASSED",
            "Policy expiry check - PASSED",
            "Claim amount vs limit check - PASSED",
            "Claim date validation - PASSED",
            "Exclusion check - PASSED"
        ],
        metadata={"processing_time_ms": 1890}
    )
    
    # Test 5: Human Review Agent
    print("\n[5/5] Testing Human Review Agent logging...")
    audit_agent.log_human_review_action(
        policy_number=test_policy,
        action="manual_review",
        inputs={
            "ai_recommendation": "ELIGIBLE",
            "ai_confidence": 0.48,
            "reason_for_review": "Low confidence score - requires human validation"
        },
        outputs={
            "final_decision": "APPROVED",
            "reasoning": "Reviewed claim details and supporting documents. All requirements met.",
            "review_duration_minutes": 15
        },
        decision="APPROVED",
        reviewer_name="Sarah Johnson",
        review_notes="Claim amount justified by medical bills. Policy coverage sufficient.",
        original_confidence=0.48,
        metadata={"review_priority": "high", "escalated": False}
    )
    
    print("\n" + "=" * 70)
    print("✅ All agent logs uploaded successfully!")
    print("=" * 70)
    
    # Retrieve and display audit trail
    print(f"\nRetrieving audit trail for policy: {test_policy}")
    print("-" * 70)
    
    audit_report = audit_agent.generate_audit_report(test_policy, test_date)
    
    print(f"\n📊 AUDIT REPORT:")
    print(f"Policy Number: {audit_report['policy_number']}")
    print(f"Total Logs: {audit_report['total_logs']}")
    print(f"\nAgent Breakdown:")
    for agent, count in audit_report['agent_breakdown'].items():
        print(f"  • {agent}: {count} log(s)")
    print(f"\nDecision Breakdown:")
    for decision, count in audit_report['decision_breakdown'].items():
        print(f"  • {decision}: {count} log(s)")
    
    print("\n" + "=" * 70)
    print("🎉 Audit Agent test completed successfully!")
    print(f"Check Azure Portal: {audit_agent.container_name}/{test_date}/{test_policy}/")
    print("=" * 70)

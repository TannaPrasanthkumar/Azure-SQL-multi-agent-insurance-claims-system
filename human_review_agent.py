"""
Human Review Agent - Manual Verification System
Handles low-confidence claims and flagged cases requiring human oversight
Provides human-in-the-loop governance for insurance claims
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, List, Optional


class HumanReviewAgent:
    """
    Agent for human review of flagged or low-confidence claims
    Threshold: Claims with confidence < 50% require manual review
    """
    
    def __init__(self, confidence_threshold: float = 50.0):
        """
        Initialize Human Review Agent
        
        Args:
            confidence_threshold: Minimum confidence percentage for automatic processing
        """
        self.confidence_threshold = confidence_threshold
        self.review_queue_file = "review_queue.json"
        self.review_history_file = "review_history.json"
        
    def needs_review(self, confidence_score: float, checks_failed: List[str]) -> bool:
        """
        Determine if a claim needs human review
        
        Args:
            confidence_score: AI confidence score (0-100)
            checks_failed: List of failed eligibility checks
            
        Returns:
            True if human review is required
        """
        # Low confidence requires review
        if confidence_score < self.confidence_threshold:
            return True
        
        # Specific edge cases that need review
        edge_cases = [
            "policy_expiry_ambiguous",
            "conflicting_information",
            "high_value_claim",  # Claims over threshold
            "multiple_simultaneous_claims"
        ]
        
        for check in checks_failed:
            if any(edge_case in check.lower() for edge_case in edge_cases):
                return True
                
        return False
    
    def flag_for_review(self, claim_data: Dict, analysis_result: Dict, reason: str) -> Dict:
        """
        Flag a claim for human review
        
        Args:
            claim_data: Original claim information
            analysis_result: AI analysis results
            reason: Reason for flagging
            
        Returns:
            Review record with unique ID
        """
        review_id = f"REV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        review_record = {
            "review_id": review_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "claim_data": claim_data,
            "analysis_result": analysis_result,
            "flag_reason": reason,
            "confidence_score": analysis_result.get('confidence_score', 0),
            "reviewer_notes": "",
            "final_decision": None,
            "reviewed_by": None,
            "review_date": None
        }
        
        # Add to review queue
        self._add_to_queue(review_record)
        
        return review_record
    
    def _add_to_queue(self, review_record: Dict):
        """Add review record to the queue"""
        queue = self._load_queue()
        queue.append(review_record)
        self._save_queue(queue)
    
    def _load_queue(self) -> List[Dict]:
        """Load pending reviews from queue"""
        if os.path.exists(self.review_queue_file):
            with open(self.review_queue_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_queue(self, queue: List[Dict]):
        """Save review queue to file"""
        with open(self.review_queue_file, 'w') as f:
            json.dump(queue, f, indent=2)
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get all pending reviews"""
        queue = self._load_queue()
        return [r for r in queue if r['status'] == 'pending']
    
    def submit_review_decision(self, review_id: str, decision: str, 
                               reviewer_name: str, notes: str) -> Dict:
        """
        Submit human review decision
        
        Args:
            review_id: Unique review identifier
            decision: "APPROVE" or "REJECT"
            reviewer_name: Name of the reviewer
            notes: Reviewer's notes/comments
            
        Returns:
            Updated review record
        """
        queue = self._load_queue()
        
        for record in queue:
            if record['review_id'] == review_id:
                record['status'] = 'reviewed'
                record['final_decision'] = decision
                record['reviewer_notes'] = notes
                record['reviewed_by'] = reviewer_name
                record['review_date'] = datetime.now().isoformat()
                
                # Save updated queue
                self._save_queue(queue)
                
                # Archive to history
                self._add_to_history(record)
                
                # Log to Audit Agent
                try:
                    from audit_agent import get_audit_agent
                    audit_agent = get_audit_agent()
                    
                    claim_data = record.get('claim_data', {})
                    analysis_result = record.get('analysis_result', {})
                    
                    audit_agent.log_human_review_action(
                        policy_number=claim_data.get('policy_number', 'UNKNOWN'),
                        action="manual_review_decision",
                        inputs={
                            "ai_recommendation": analysis_result.get('eligibility_decision', 'UNKNOWN'),
                            "ai_confidence": record.get('confidence_score', 0),
                            "reason_for_review": "Low confidence score - manual validation required",
                            "claim_details": claim_data
                        },
                        outputs={
                            "final_decision": decision,
                            "reasoning": notes,
                            "review_date": record['review_date']
                        },
                        decision=decision,
                        reviewer_name=reviewer_name,
                        review_notes=notes,
                        original_confidence=record.get('confidence_score', 0),
                        metadata={
                            "review_id": review_id,
                            "review_duration": "calculated_at_runtime"
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Failed to log human review to audit: {e}")
                
                return record
        
        raise ValueError(f"Review ID {review_id} not found")
    
    def _add_to_history(self, review_record: Dict):
        """Archive completed review to history"""
        history = []
        if os.path.exists(self.review_history_file):
            with open(self.review_history_file, 'r') as f:
                history = json.load(f)
        
        history.append(review_record)
        
        with open(self.review_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_review_statistics(self) -> Dict:
        """Get statistics about reviews"""
        queue = self._load_queue()
        history = []
        
        if os.path.exists(self.review_history_file):
            with open(self.review_history_file, 'r') as f:
                history = json.load(f)
        
        all_reviews = queue + history
        
        stats = {
            "total_reviews": len(all_reviews),
            "pending": len([r for r in all_reviews if r['status'] == 'pending']),
            "reviewed": len([r for r in all_reviews if r['status'] == 'reviewed']),
            "approved": len([r for r in all_reviews if r.get('final_decision') == 'APPROVE']),
            "rejected": len([r for r in all_reviews if r.get('final_decision') == 'REJECT']),
            "avg_confidence": sum([r['confidence_score'] for r in all_reviews]) / len(all_reviews) if all_reviews else 0
        }
        
        return stats


def render_human_review_ui():
    """
    Render the Human Review Agent UI in Streamlit
    This provides the interface for human reviewers
    """
    st.markdown("""
    <style>
    .review-card {
        background: #f8f9fa;
        border-left: 4px solid #ff6b6b;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .review-card-low {
        border-left-color: #ffa500;
    }
    .review-card-high {
        border-left-color: #ff0000;
    }
    .decision-approve {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .decision-reject {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title is now shown in workflow_visualizer.py tab2 section, no need to duplicate here
    st.markdown("**Human-in-the-Loop Governance for Insurance Claims**")
    
    # Check if there's a flagged claim from Process Claims tab
    if st.session_state.get('needs_human_review', False):
        review_id = st.session_state.get('current_review_id', 'Unknown')
        st.success(f"✅ **New Review Available:** Review ID `{review_id}` - Check pending reviews below.")
        # Clear the flag after showing
        if st.button("✔️ Acknowledged"):
            st.session_state['needs_human_review'] = False
            st.rerun()
    
    # Check if there's a rejected claim to review
    if st.session_state.get('rejected_claim_for_review'):
        rejected = st.session_state['rejected_claim_for_review']
        st.error(f"❌ **Rejected Claim Available for Override:** Policy `{rejected['policy_number']}` - See details below.")
        with st.expander("📋 View Rejection Details", expanded=True):
            st.write(f"**Decision:** {rejected['decision']}")
            st.write(f"**Confidence:** {rejected['confidence']}%")
            st.write(f"**Failed Checks:** {', '.join(rejected['checks_failed']) if rejected['checks_failed'] else 'None'}")
            if st.button("✔️ Clear Notification"):
                del st.session_state['rejected_claim_for_review']
                st.rerun()
    
    # Check if there's a fraud-flagged claim for review
    if st.session_state.get('fraud_claim_for_review'):
        fraud_claim = st.session_state['fraud_claim_for_review']
        st.warning(f"🚨 **FRAUD ALERT - Human Review Required:** Policy `{fraud_claim['policy_number']}`")
        
        with st.expander("🚨 Fraud Detection Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{fraud_claim['fraud_probability']:.2%}", 
                         delta=f"+{(fraud_claim['fraud_probability'] - fraud_claim['threshold']):.2%} above threshold")
            with col2:
                st.metric("Risk Level", fraud_claim['fraud_risk'])
            with col3:
                st.metric("Threshold", f"{fraud_claim['threshold']:.0%}")
            
            st.markdown("---")
            
            # Claim Information
            extracted_data = fraud_claim.get('extracted_data', {})
            claim_info = extracted_data.get('claim_info', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📄 Claim Information")
                st.write(f"**Policy Number:** {claim_info.get('policy_number', 'N/A')}")
                st.write(f"**Policyholder:** {claim_info.get('policyholder_name', 'N/A')}")
                st.write(f"**Claim Amount:** ${claim_info.get('claim_amount', 0):,.2f}")
                st.write(f"**Claim Date:** {claim_info.get('claim_date', 'N/A')}")
                st.write(f"**Reason:** {claim_info.get('reason_for_claim', 'N/A')}")
            
            with col2:
                st.markdown("#### 🎯 Fraud Indicators")
                st.write(f"**Driver Rating:** {claim_info.get('driver_rating', 'N/A')}")
                st.write(f"**Age:** {claim_info.get('age', 'N/A')}")
                st.write(f"**Policy Type:** {claim_info.get('policy_type', 'N/A')}")
                st.write(f"**Accident Area:** {claim_info.get('accident_area', 'N/A')}")
                st.write(f"**Police Report:** {claim_info.get('police_report_filed', 'N/A')}")
            
            st.markdown("---")
            
            # Human Review Decision Form
            st.markdown("#### ✍️ Manual Review Decision")
            
            with st.form(key="fraud_review_form"):
                reviewer_name = st.text_input("Reviewer Name", key="fraud_reviewer_name")
                
                decision = st.radio(
                    "Final Decision",
                    ["APPROVE - Process Claim (Override Fraud Detection)", 
                     "REJECT - Deny Claim (Confirm Fraud)"],
                    key="fraud_decision"
                )
                
                reviewer_notes = st.text_area(
                    "Review Notes & Justification (Required)",
                    placeholder="Explain your decision to override or confirm the fraud detection...",
                    height=150,
                    key="fraud_notes"
                )
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    submit = st.form_submit_button("✅ Submit Decision", type="primary")
                
                if submit:
                    if not reviewer_name:
                        st.error("⚠️ Please enter your name")
                    elif not reviewer_notes or len(reviewer_notes.strip()) < 20:
                        st.error("⚠️ Please provide detailed justification (minimum 20 characters)")
                    else:
                        # Process the decision
                        final_decision = "APPROVE" if "APPROVE" in decision else "REJECT"
                        
                        # Store the review result in session state
                        st.session_state['fraud_review_result'] = {
                            'policy_number': fraud_claim['policy_number'],
                            'decision': final_decision,
                            'reviewer': reviewer_name,
                            'notes': reviewer_notes,
                            'fraud_probability': fraud_claim['fraud_probability'],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Mark processing as completed for results display
                        st.session_state['processing_completed'] = True
                        
                        # Clear fraud detection flag so workflow shows complete
                        if 'fraud_detected' in st.session_state:
                            del st.session_state['fraud_detected']
                        
                        # Log to Audit Agent if available
                        try:
                            from audit_agent import get_audit_agent
                            audit_agent = get_audit_agent()
                            
                            audit_agent.log_human_review_action(
                                policy_number=fraud_claim['policy_number'],
                                action="fraud_manual_review",
                                inputs={
                                    "fraud_probability": fraud_claim['fraud_probability'],
                                    "fraud_risk": fraud_claim['fraud_risk'],
                                    "ml_recommendation": "REJECT",
                                    "claim_details": claim_info
                                },
                                outputs={
                                    "final_decision": final_decision,
                                    "reasoning": reviewer_notes
                                },
                                decision=final_decision,
                                reviewer_name=reviewer_name,
                                review_notes=reviewer_notes,
                                original_confidence=fraud_claim['fraud_probability'] * 100,
                                metadata={
                                    "review_type": "fraud_override",
                                    "fraud_threshold": fraud_claim['threshold']
                                }
                            )
                        except Exception as e:
                            print(f"⚠️ Failed to log to audit: {e}")
                        
                        # Clear the fraud claim from session state
                        del st.session_state['fraud_claim_for_review']
                        
                        # Show success message
                        if final_decision == "APPROVE":
                            st.success(f"✅ **Claim {fraud_claim['policy_number']} APPROVED** by {reviewer_name}")
                            st.info("The claim will proceed to processing despite fraud detection.")
                        else:
                            st.error(f"❌ **Claim {fraud_claim['policy_number']} REJECTED** by {reviewer_name}")
                            st.info("The claim has been denied due to confirmed fraud.")
                        
                        st.balloons()
                        st.rerun()
    
    st.markdown("---")
    
    # Initialize agent
    agent = HumanReviewAgent(confidence_threshold=50.0)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Pending Reviews", "📊 Review Statistics", "📜 Review History"])
    
    with tab1:
        st.header("📋 Claims Requiring Manual Review")
        st.markdown("**Threshold:** Claims with AI confidence < 50% or flagged conditions")
        
        pending_reviews = agent.get_pending_reviews()
        
        if not pending_reviews:
            st.success("✅ No claims pending review. All claims processed automatically!")
            st.info("Claims are flagged for review when:\n- AI confidence score is below 50%\n- Edge cases or ambiguous information detected\n- High-value claims requiring additional verification")
        else:
            st.warning(f"⚠️ {len(pending_reviews)} claim(s) require human review")
            
            for idx, review in enumerate(pending_reviews, 1):
                with st.expander(f"📄 Review #{idx}: {review['review_id']} - Confidence: {review['confidence_score']:.1f}%", expanded=(idx == 1)):
                    
                    # Review Card
                    confidence = review['confidence_score']
                    severity_class = "review-card-high" if confidence < 30 else "review-card-low"
                    
                    st.markdown(f"""
                    <div class="review-card {severity_class}">
                        <h3>🚨 Review Required</h3>
                        <p><strong>Review ID:</strong> {review['review_id']}</p>
                        <p><strong>Flagged:</strong> {datetime.fromisoformat(review['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>AI Confidence:</strong> {confidence:.1f}% (Below 50% threshold)</p>
                        <p><strong>Reason:</strong> {review['flag_reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Claim Information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📄 Claim Information")
                        claim_data = review['claim_data']
                        
                        if 'extracted_data' in claim_data and 'claim_info' in claim_data['extracted_data']:
                            claim_info = claim_data['extracted_data']['claim_info']
                            st.write(f"**Policy Number:** {claim_info.get('policy_number', 'N/A')}")
                            st.write(f"**Policyholder:** {claim_info.get('policyholder_name', 'N/A')}")
                            st.write(f"**Claim Amount:** ${claim_info.get('claim_amount', 0):,.2f}")
                            st.write(f"**Claim Date:** {claim_info.get('claim_date', 'N/A')}")
                            st.write(f"**Reason:** {claim_info.get('reason_for_claim', 'N/A')}")
                    
                    with col2:
                        st.markdown("### 🤖 AI Analysis Summary")
                        analysis = review['analysis_result']
                        
                        st.write(f"**Decision:** {analysis.get('eligibility_decision', 'N/A')}")
                        st.write(f"**Confidence:** {analysis.get('confidence_score', 0):.1f}%")
                        
                        checks_failed = analysis.get('checks_failed', [])
                        if checks_failed:
                            st.write(f"**Failed Checks:** {len(checks_failed)}")
                            for check in checks_failed:
                                st.write(f"  • {check}")
                    
                    st.markdown("---")
                    
                    # AI Reasoning
                    st.markdown("### 🧠 AI Reasoning")
                    if 'detailed_checks' in analysis:
                        for check in analysis['detailed_checks'][:5]:  # Show first 5 checks
                            if check.strip():
                                st.markdown(check)
                    
                    st.markdown("---")
                    
                    # Human Review Form
                    st.markdown("### ✍️ Human Review Decision")
                    
                    with st.form(key=f"review_form_{review['review_id']}"):
                        reviewer_name = st.text_input("Reviewer Name", key=f"name_{review['review_id']}")
                        
                        decision = st.radio(
                            "Final Decision",
                            ["APPROVE", "REJECT"],
                            key=f"decision_{review['review_id']}",
                            horizontal=True
                        )
                        
                        reviewer_notes = st.text_area(
                            "Reviewer Notes & Justification",
                            placeholder="Provide detailed notes explaining your decision...",
                            height=150,
                            key=f"notes_{review['review_id']}"
                        )
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            submit = st.form_submit_button("✅ Submit Review", type="primary")
                        
                        with col2:
                            if st.form_submit_button("🔄 Reset Form"):
                                st.rerun()
                        
                        if submit:
                            if not reviewer_name:
                                st.error("⚠️ Please enter your name")
                            elif not reviewer_notes:
                                st.error("⚠️ Please provide justification notes")
                            else:
                                # Submit the review
                                result = agent.submit_review_decision(
                                    review['review_id'],
                                    decision,
                                    reviewer_name,
                                    reviewer_notes
                                )
                                
                                if decision == "APPROVE":
                                    st.success(f"✅ Claim {review['review_id']} APPROVED by {reviewer_name}")
                                else:
                                    st.error(f"❌ Claim {review['review_id']} REJECTED by {reviewer_name}")
                                
                                st.balloons()
                                st.rerun()
    
    with tab2:
        st.header("📊 Review Statistics Dashboard")
        
        stats = agent.get_review_statistics()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", stats['total_reviews'])
        
        with col2:
            st.metric("Pending", stats['pending'], delta=f"-{stats['reviewed']} reviewed")
        
        with col3:
            approval_rate = (stats['approved'] / stats['reviewed'] * 100) if stats['reviewed'] > 0 else 0
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        with col4:
            st.metric("Avg AI Confidence", f"{stats['avg_confidence']:.1f}%")
        
        st.markdown("---")
        
        # Charts
        if stats['total_reviews'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Review Status Distribution")
                status_data = pd.DataFrame({
                    'Status': ['Pending', 'Approved', 'Rejected'],
                    'Count': [stats['pending'], stats['approved'], stats['rejected']]
                })
                st.bar_chart(status_data.set_index('Status'))
            
            with col2:
                st.markdown("### Review Outcomes")
                if stats['reviewed'] > 0:
                    st.write(f"**Total Reviewed:** {stats['reviewed']}")
                    st.write(f"**Approved:** {stats['approved']} ({stats['approved']/stats['reviewed']*100:.1f}%)")
                    st.write(f"**Rejected:** {stats['rejected']} ({stats['rejected']/stats['reviewed']*100:.1f}%)")
                else:
                    st.info("No reviews completed yet")
        else:
            st.info("No review data available yet")
    
    with tab3:
        st.header("📜 Review History")
        
        if os.path.exists(agent.review_history_file):
            with open(agent.review_history_file, 'r') as f:
                history = json.load(f)
            
            if history:
                st.write(f"**Total Completed Reviews:** {len(history)}")
                
                # Display as table
                history_df = pd.DataFrame([
                    {
                        'Review ID': h['review_id'],
                        'Reviewed Date': datetime.fromisoformat(h['review_date']).strftime('%Y-%m-%d %H:%M'),
                        'Reviewer': h['reviewed_by'],
                        'Decision': h['final_decision'],
                        'AI Confidence': f"{h['confidence_score']:.1f}%",
                        'Flag Reason': h['flag_reason']
                    }
                    for h in sorted(history, key=lambda x: x['review_date'], reverse=True)
                ])
                
                st.dataframe(history_df, use_container_width=True)
                
                # Detailed view
                st.markdown("---")
                selected_review = st.selectbox(
                    "View Detailed Review",
                    options=[h['review_id'] for h in history],
                    format_func=lambda x: f"{x} - {next(h['final_decision'] for h in history if h['review_id'] == x)}"
                )
                
                if selected_review:
                    review_detail = next(h for h in history if h['review_id'] == selected_review)
                    
                    st.markdown(f"### Review Details: {selected_review}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Reviewed By:** {review_detail['reviewed_by']}")
                        st.write(f"**Review Date:** {datetime.fromisoformat(review_detail['review_date']).strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Final Decision:** {review_detail['final_decision']}")
                    
                    with col2:
                        st.write(f"**AI Confidence:** {review_detail['confidence_score']:.1f}%")
                        st.write(f"**Flag Reason:** {review_detail['flag_reason']}")
                    
                    st.markdown("**Reviewer Notes:**")
                    st.info(review_detail['reviewer_notes'])
            else:
                st.info("No completed reviews yet")
        else:
            st.info("No review history available")


if __name__ == "__main__":
    render_human_review_ui()

"""
Fraud Detection Agent - AI-powered fraud detection for insurance claims
Analyzes claims for potential fraud indicators and risk assessment
Combines Rule-based + ML Model + AI Analysis
"""

import os
import json
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv
from fraud_ml_model import get_fraud_ml_model

load_dotenv()

class FraudDetectionAgent:
    """
    Fraud Detection Agent that analyzes insurance claims for fraud indicators
    Uses AI to detect suspicious patterns, anomalies, and high-risk factors
    """
    
    def __init__(self):
        """Initialize the Fraud Detection Agent with Azure OpenAI and ML Model"""
        self.endpoint = os.getenv("AZURE_AISERVICES_ENDPOINT")
        self.api_key = os.getenv("AZURE_AISERVICES_APIKEY")
        self.deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
        
        if self.endpoint and self.api_key:
            self.client = AzureOpenAI(
                api_version="2024-02-15-preview",
                azure_endpoint=self.endpoint,
                api_key=self.api_key
            )
            self.enabled = True
        else:
            self.client = None
            self.enabled = False
        
        # Initialize ML model
        try:
            self.ml_model = get_fraud_ml_model()
            self.ml_enabled = True
            print("✅ ML Fraud Detection Model loaded")
        except Exception as e:
            self.ml_model = None
            self.ml_enabled = False
            print(f"⚠️ ML Model unavailable: {e}")
    
    def analyze_fraud_risk(self, claim_data, policy_data, eligibility_result):
        """
        Analyze claim for fraud indicators
        
        Args:
            claim_data: Extracted claim information
            policy_data: Policy validation data from database
            eligibility_result: Eligibility check results
            
        Returns:
            dict: Fraud risk assessment with score and indicators
        """
        if not self.enabled:
            return {
                "fraud_risk_score": 0,
                "risk_level": "UNKNOWN",
                "fraud_indicators": [],
                "reasoning": "Fraud detection unavailable - AI service not configured",
                "recommendation": "Manual review recommended"
            }
        
        try:
            # Extract relevant data
            claim_info = claim_data.get('claim_info', {})
            policy_info = policy_data.get('policy_info', {})
            validation_details = policy_data.get('validation', {}).get('details', {})
            
            claim_amount = claim_info.get('claim_amount', 0)
            reason = claim_info.get('reason_for_claim', '')
            claim_date = claim_info.get('claim_date', '')
            policy_number = claim_info.get('policy_number', '')
            
            policy_limit = validation_details.get('policy_limit', 0)
            past_claims = validation_details.get('past_claims_amount', 0)
            claim_history = validation_details.get('claim_history_count', 0)
            policy_status = validation_details.get('policy_status', '')
            policy_expiry = validation_details.get('policy_expiry_date', '')
            
            # Rule-based fraud indicators
            fraud_indicators = []
            risk_score = 0
            
            # INDICATOR 1: Claim amount close to policy limit (suspicious timing)
            if policy_limit > 0:
                limit_utilization = (claim_amount / policy_limit) * 100
                if limit_utilization > 95:
                    fraud_indicators.append({
                        "indicator": "High Limit Utilization",
                        "severity": "HIGH",
                        "description": f"Claim amount ({claim_amount:,.2f}) is {limit_utilization:.1f}% of policy limit",
                        "weight": 25
                    })
                    risk_score += 25
                elif limit_utilization > 85:
                    fraud_indicators.append({
                        "indicator": "Suspicious Limit Utilization",
                        "severity": "MEDIUM",
                        "description": f"Claim amount is {limit_utilization:.1f}% of limit - close to maximum",
                        "weight": 15
                    })
                    risk_score += 15
            
            # INDICATOR 2: Multiple claims in short period
            if claim_history >= 3:
                fraud_indicators.append({
                    "indicator": "Frequent Claims",
                    "severity": "MEDIUM",
                    "description": f"{claim_history} previous claims - high claim frequency",
                    "weight": 20
                })
                risk_score += 20
            
            # INDICATOR 3: Round number amounts (often fabricated)
            if claim_amount > 0 and claim_amount % 1000 == 0 and claim_amount >= 10000:
                fraud_indicators.append({
                    "indicator": "Round Number Claim",
                    "severity": "LOW",
                    "description": f"Claim amount is exactly ${claim_amount:,.0f} (suspiciously round)",
                    "weight": 10
                })
                risk_score += 10
            
            # INDICATOR 4: Large claim amount
            if claim_amount > 100000:
                fraud_indicators.append({
                    "indicator": "High Value Claim",
                    "severity": "MEDIUM",
                    "description": f"Large claim amount: ${claim_amount:,.2f}",
                    "weight": 15
                })
                risk_score += 15
            
            # INDICATOR 5: Claim filed close to expiry date
            if claim_date and policy_expiry:
                try:
                    from datetime import datetime
                    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']
                    for fmt in date_formats:
                        try:
                            claim_dt = datetime.strptime(claim_date, fmt)
                            expiry_dt = datetime.strptime(policy_expiry, fmt)
                            days_before_expiry = (expiry_dt - claim_dt).days
                            
                            if 0 <= days_before_expiry <= 30:
                                fraud_indicators.append({
                                    "indicator": "Claim Near Expiry",
                                    "severity": "MEDIUM",
                                    "description": f"Claim filed {days_before_expiry} days before policy expiration",
                                    "weight": 20
                                })
                                risk_score += 20
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # AI-powered fraud analysis using GPT-4
            ai_analysis = self._ai_fraud_analysis(claim_info, validation_details, fraud_indicators)
            
            # ML Model prediction
            ml_result = {}
            if self.ml_enabled and self.ml_model:
                try:
                    ml_result = self.ml_model.predict_fraud(claim_data, policy_data)
                    
                    # Weight ML prediction into final risk score
                    ml_risk = ml_result.get('ml_risk_score', 0)
                    # Combine: 50% rule-based + 30% ML + 20% AI
                    combined_risk = (risk_score * 0.5) + (ml_risk * 0.3) + (ai_analysis.get('ai_risk_score', 0) * 0.2)
                    risk_score = min(int(combined_risk), 100)
                    
                    # Add ML insights to indicators
                    if ml_result.get('ml_prediction') == "FRAUD":
                        fraud_indicators.append({
                            "indicator": "ML Model Prediction",
                            "severity": ml_result.get('ml_risk_level', 'MEDIUM'),
                            "description": f"Machine Learning model predicts {ml_result['ml_fraud_probability']:.1%} fraud probability",
                            "weight": ml_risk
                        })
                    
                except Exception as e:
                    print(f"⚠️ ML prediction error: {e}")
            
            # Combine rule-based and AI analysis
            if ai_analysis.get('additional_risk'):
                # Already weighted in combined score if ML is enabled
                if not self.ml_enabled:
                    risk_score += ai_analysis.get('ai_risk_score', 0)
                fraud_indicators.extend(ai_analysis.get('ai_indicators', []))
            
            # Cap risk score at 100
            risk_score = min(risk_score, 100)
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = "HIGH"
                recommendation = "REJECT - High fraud risk detected. Thorough investigation required."
            elif risk_score >= 40:
                risk_level = "MEDIUM"
                recommendation = "REVIEW - Moderate fraud risk. Manual verification strongly recommended."
            elif risk_score >= 20:
                risk_level = "LOW"
                recommendation = "CAUTION - Minor fraud indicators detected. Standard verification recommended."
            else:
                risk_level = "MINIMAL"
                recommendation = "PROCEED - No significant fraud indicators detected."
            
            return {
                "fraud_risk_score": risk_score,
                "risk_level": risk_level,
                "fraud_indicators": fraud_indicators,
                "indicator_count": len(fraud_indicators),
                "reasoning": ai_analysis.get('reasoning', 'Hybrid fraud detection completed (Rules + ML + AI)'),
                "recommendation": recommendation,
                "ai_confidence": ai_analysis.get('confidence', 85),
                "requires_investigation": risk_score >= 40,
                "ml_prediction": ml_result.get('ml_prediction', 'N/A'),
                "ml_fraud_probability": ml_result.get('ml_fraud_probability', 0),
                "ml_confidence": ml_result.get('ml_confidence', 0),
                "detection_method": "Hybrid (Rules + ML + AI)" if self.ml_enabled else "Hybrid (Rules + AI)"
            }
            
        except Exception as e:
            return {
                "fraud_risk_score": 0,
                "risk_level": "ERROR",
                "fraud_indicators": [],
                "reasoning": f"Fraud detection error: {str(e)}",
                "recommendation": "Manual review required due to analysis error"
            }
    
    def _ai_fraud_analysis(self, claim_info, policy_details, rule_indicators):
        """
        Use AI to detect additional fraud patterns and anomalies
        """
        if not self.client:
            return {"additional_risk": False}
        
        try:
            # Prepare context for AI
            reason = claim_info.get('reason_for_claim', 'Not specified')
            claim_amount = claim_info.get('claim_amount', 0)
            policy_type = policy_details.get('policy_type', 'Unknown')
            
            rule_summary = "\n".join([
                f"- {ind['indicator']}: {ind['description']}"
                for ind in rule_indicators
            ])
            
            prompt = f"""You are an insurance fraud detection specialist. Analyze this claim for additional fraud indicators.

**Claim Details:**
- Amount: ${claim_amount:,.2f}
- Reason: {reason}
- Policy Type: {policy_type}

**Rule-Based Indicators Already Detected:**
{rule_summary if rule_summary else "None"}

**Task:**
Analyze the claim reason for:
1. Vague or generic descriptions
2. Unusual or suspicious wording
3. Inconsistencies or contradictions
4. Patterns common in fraudulent claims

Return JSON:
{{
    "additional_risk": true/false,
    "ai_risk_score": 0-30 (additional risk score),
    "reasoning": "Brief explanation",
    "ai_indicators": [
        {{
            "indicator": "Name",
            "severity": "HIGH/MEDIUM/LOW",
            "description": "Specific finding",
            "weight": 5-30
        }}
    ],
    "confidence": 0-100
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are an expert insurance fraud detection AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            ai_result = json.loads(response.choices[0].message.content)
            return ai_result
            
        except Exception as e:
            return {
                "additional_risk": False,
                "reasoning": f"AI analysis unavailable: {str(e)}"
            }


def get_fraud_detection_agent():
    """Get singleton instance of Fraud Detection Agent"""
    return FraudDetectionAgent()


if __name__ == "__main__":
    # Test the fraud detection agent
    agent = FraudDetectionAgent()
    
    if agent.enabled:
        print("✅ Fraud Detection Agent initialized successfully")
        
        # Test with sample data
        test_claim = {
            'claim_info': {
                'policy_number': 'TEST001',
                'claim_amount': 50000,
                'reason_for_claim': 'Vehicle accident',
                'claim_date': '2025-11-15'
            }
        }
        
        test_policy = {
            'policy_info': {'policy_number': 'TEST001'},
            'validation': {
                'details': {
                    'policy_limit': 100000,
                    'past_claims_amount': 30000,
                    'claim_history_count': 2,
                    'policy_status': 'Active',
                    'policy_expiry_date': '2025-12-31',
                    'policy_type': 'Vehicle'
                }
            }
        }
        
        result = agent.analyze_fraud_risk(test_claim, test_policy, None)
        
        print("\n🔍 Test Fraud Analysis:")
        print(f"Risk Score: {result['fraud_risk_score']}/100")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Indicators: {result['indicator_count']}")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print("❌ Fraud Detection Agent not available - check Azure OpenAI configuration")

"""
Fraud Detector Agent
Uses Azure ML deployed model to detect potential fraud in insurance claims
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FraudDetectorAgent:
    def __init__(self):
        """Initialize Fraud Detector Agent with Azure ML endpoint"""
        self.scoring_uri = os.getenv("AZURE_ML_ENDPOINT")
        self.api_key = os.getenv("AZURE_ML_API_KEY")
        
        if not self.scoring_uri or not self.api_key:
            raise ValueError("Azure ML endpoint credentials not found in .env file")
    
    def detect_fraud(self, claim_data):
        """
        Detect fraud using Azure ML model
        
        Args:
            claim_data (dict): Dictionary containing fraud detection fields:
                - DriverRating: Driver rating (1-4)
                - Age: Age of driver
                - PoliceReportFiled: Whether police report was filed (0/1)
                - WeekOfMonthClaimed: Week of month when claim was made (1-5)
                - PolicyType: Type of policy (encoded)
                - WeekOfMonth: Week of month (1-5)
                - AccidentArea: Area where accident occurred (encoded)
                - Sex: Gender (encoded)
                - Deductible: Insurance deductible amount
        
        Returns:
            dict: Fraud detection result with prediction, probability, and risk level
        """
        try:
            # IMPORTANT: Azure ML scoring.py expects NUMERIC values for categorical fields
            # The label encoders on the server side will handle the encoding
            # Mappings: string to numeric (reverse of what scoring.py has)
            accident_area_map = {"Rural": 0, "Urban": 1}
            sex_map = {"Female": 0, "Male": 1}
            policy_type_map = {
                "Sedan - All Perils": 0,
                "Sedan - Collision": 1, 
                "Sedan - Liability": 2,
                "Sport - All Perils": 3,
                "Sport - Collision": 4,
                "Sport - Liability": 5,
                "Utility - All Perils": 6,
                "Utility - Collision": 7,
                "Utility - Liability": 8
            }
            police_report_map = {"No": 0, "Yes": 1}
            
            # Get values from claim_data
            accident_area = claim_data.get("AccidentArea", 1)
            sex = claim_data.get("Sex", 1)
            policy_type = claim_data.get("PolicyType", 2)  # Default to Sedan - Liability
            police_report = claim_data.get("PoliceReportFiled", 0)
            
            # Convert strings to numeric if needed
            if isinstance(accident_area, str):
                accident_area = accident_area_map.get(accident_area, 1)  # Default to Urban
            else:
                accident_area = int(accident_area)
                
            if isinstance(sex, str):
                sex = sex_map.get(sex, 1)  # Default to Male
            else:
                sex = int(sex)
                
            if isinstance(policy_type, str):
                policy_type = policy_type_map.get(policy_type, 2)  # Default to Sedan - Liability
            else:
                policy_type = int(policy_type)
                
            if isinstance(police_report, str):
                police_report = police_report_map.get(police_report, 0)  # Default to No
            else:
                police_report = int(police_report)
            
            # Prepare required fields for model
            # Send categorical values as NUMBERS (0, 1, 2, etc.)
            # Azure ML's scoring.py expects numeric values
            payload = {
                "DriverRating": int(claim_data.get("DriverRating", 1)),
                "Age": int(claim_data.get("Age", 30)),
                "WeekOfMonthClaimed": int(claim_data.get("WeekOfMonthClaimed", 1)),
                "WeekOfMonth": int(claim_data.get("WeekOfMonth", 1)),
                "Deductible": int(claim_data.get("Deductible", 500)),
                "AccidentArea": accident_area,  # Numeric: 0 (Rural) or 1 (Urban)
                "Sex": sex,  # Numeric: 0 (Female) or 1 (Male)
                "PolicyType": policy_type,  # Numeric: 0-8
                "PoliceReportFiled": police_report  # Numeric: 0 (No) or 1 (Yes)
            }
            
            # DEBUG: Print what we're sending to ML
            print("\n" + "="*70)
            print("🚀 FRAUD DETECTOR - SENDING TO AZURE ML:")
            print(json.dumps(payload, indent=2))
            print("="*70 + "\n")
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Call Azure ML endpoint
            response = requests.post(
                self.scoring_uri,
                data=json.dumps(payload),
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"⚠️ Azure ML ERROR - Status {response.status_code}")
                print(f"Response: {response.text}")
                return {
                    "success": False,
                    "error": f"Azure ML endpoint returned status {response.status_code}",
                    "fraud_prediction": 0,
                    "fraud_probability": 0.0,
                    "fraud_risk": "Unknown"
                }
            
            # Parse response
            result = response.json()
            
            # DEBUG: Print raw ML response
            print("\n" + "="*70)
            print("📥 FRAUD DETECTOR - RAW AZURE ML RESPONSE:")
            print(json.dumps(result, indent=2))
            print("="*70 + "\n")
            
            # Handle double-encoded JSON
            if isinstance(result, str):
                result = json.loads(result)
            
            # Check for error in response
            if "error" in result:
                error_msg = result.get("error", "Unknown error")
                print(f"⚠️ AZURE ML MODEL ERROR: {error_msg}")
                return {
                    "success": False,
                    "error": f"Azure ML model error: {error_msg}",
                    "fraud_prediction": 0,
                    "fraud_probability": 0.0,
                    "fraud_risk": "Error",
                    "threshold_used": 0.5,
                    "is_fraud": False,
                    "message": f"Model error: {error_msg}"
                }
            
            # Extract prediction details
            if "predictions" in result:
                pred = result["predictions"][0]
                fraud_prediction = pred["fraud_prediction"]
                fraud_probability = pred["fraud_probability"]
                fraud_risk = pred.get("fraud_risk", "Unknown")
                threshold = pred.get("threshold_used", 0.5)
            else:
                fraud_prediction = result.get("fraud_prediction", 0)
                fraud_probability = result.get("fraud_probability", 0.0)
                fraud_risk = result.get("fraud_risk", "Unknown")
                threshold = result.get("threshold_used", 0.5)
            
            return {
                "success": True,
                "fraud_prediction": fraud_prediction,
                "fraud_probability": round(fraud_probability, 4),
                "fraud_risk": fraud_risk,
                "threshold_used": threshold,
                "is_fraud": fraud_prediction == 1,
                "message": f"Fraud analysis complete: {fraud_risk}"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error calling Azure ML: {str(e)}",
                "fraud_prediction": 0,
                "fraud_probability": 0.0,
                "fraud_risk": "Error"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fraud detection error: {str(e)}",
                "fraud_prediction": 0,
                "fraud_probability": 0.0,
                "fraud_risk": "Error"
            }
    
    def get_risk_recommendation(self, fraud_result):
        """
        Get recommendation based on fraud risk level
        
        Args:
            fraud_result (dict): Result from detect_fraud()
        
        Returns:
            str: Recommendation message
        """
        if not fraud_result.get("success"):
            return "Unable to assess fraud risk due to technical error"
        
        fraud_probability = fraud_result["fraud_probability"]
        
        if fraud_probability >= 0.7:
            return "⚠️ HIGH FRAUD RISK - Recommend thorough investigation and additional verification"
        elif fraud_probability >= 0.5:
            return "⚠️ MODERATE FRAUD RISK - Recommend detailed review of claim documentation"
        elif fraud_probability >= 0.3:
            return "⚠️ LOW-MODERATE FRAUD RISK - Standard verification procedures recommended"
        else:
            return "✅ LOW FRAUD RISK - Claim appears legitimate, proceed with standard review"


# Test function
if __name__ == "__main__":
    print("Testing Fraud Detector Agent...")
    print("=" * 70)
    
    agent = FraudDetectorAgent()
    
    # Test data
    test_claim = {
        "DriverRating": 1,
        "Age": 35,
        "PoliceReportFiled": 1,
        "WeekOfMonthClaimed": 2,
        "PolicyType": 1,
        "WeekOfMonth": 2,
        "AccidentArea": 1,
        "Sex": 1,
        "Deductible": 500
    }
    
    print("\nTest Claim Data:")
    for key, value in test_claim.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Calling Azure ML Fraud Detection Model...")
    print("=" * 70)
    
    result = agent.detect_fraud(test_claim)
    
    print("\nFraud Detection Result:")
    print(f"  Success: {result['success']}")
    if result['success']:
        print(f"  Fraud Prediction: {'FRAUD' if result['is_fraud'] else 'NOT FRAUD'}")
        print(f"  Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"  Risk Level: {result['fraud_risk']}")
        print(f"  Threshold Used: {result['threshold_used']}")
        print(f"\n  Recommendation:")
        print(f"  {agent.get_risk_recommendation(result)}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")

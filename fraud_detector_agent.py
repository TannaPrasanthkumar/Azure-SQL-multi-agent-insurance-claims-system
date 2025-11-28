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
            # Mapping for categorical features (numeric to string)
            # Azure ML's scoring.py expects strings, not pre-encoded numbers
            accident_area_map = {0: "Rural", 1: "Urban"}
            sex_map = {0: "Female", 1: "Male"}
            policy_type_map = {
                0: "Sedan - All Perils",
                1: "Sedan - Collision", 
                2: "Sedan - Liability",
                3: "Sport - All Perils",
                4: "Sport - Collision",
                5: "Sport - Liability",
                6: "Utility - All Perils",
                7: "Utility - Collision",
                8: "Utility - Liability"
            }
            police_report_map = {0: "No", 1: "Yes"}
            
            # Get values from claim_data
            accident_area = claim_data.get("AccidentArea", 1)
            sex = claim_data.get("Sex", 1)
            policy_type = claim_data.get("PolicyType", 1)
            police_report = claim_data.get("PoliceReportFiled", 0)
            
            # Convert to strings if they're numeric
            if isinstance(accident_area, (int, float)):
                accident_area = accident_area_map.get(int(accident_area), "Urban")
            if isinstance(sex, (int, float)):
                sex = sex_map.get(int(sex), "Male")
            if isinstance(policy_type, (int, float)):
                policy_type = policy_type_map.get(int(policy_type), "Sedan - Liability")
            if isinstance(police_report, (int, float)):
                police_report = police_report_map.get(int(police_report), "No")
            
            # Prepare required fields for model
            # Send categorical values as STRINGS, not encoded numbers
            # Azure ML's scoring.py will encode them using label_encoders
            features = {
                "DriverRating": claim_data.get("DriverRating", 1),
                "Age": claim_data.get("Age", 30),
                "WeekOfMonthClaimed": claim_data.get("WeekOfMonthClaimed", 1),
                "WeekOfMonth": claim_data.get("WeekOfMonth", 1),
                "Deductible": claim_data.get("Deductible", 500),
                "AccidentArea": accident_area,  # String: "Urban" or "Rural"
                "Sex": sex,  # String: "Male" or "Female"
                "PolicyType": policy_type,  # String: "Sedan - All Perils", etc.
                "PoliceReportFiled": police_report  # String: "Yes" or "No"
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Call Azure ML endpoint
            response = requests.post(
                self.scoring_uri,
                data=json.dumps(features),
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Azure ML endpoint returned status {response.status_code}",
                    "fraud_prediction": 0,
                    "fraud_probability": 0.0,
                    "fraud_risk": "Unknown"
                }
            
            # Parse response
            result = response.json()
            
            # Handle double-encoded JSON
            if isinstance(result, str):
                result = json.loads(result)
            
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

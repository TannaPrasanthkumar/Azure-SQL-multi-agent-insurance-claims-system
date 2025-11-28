"""
Fraud Detection ML Model - Machine Learning-based fraud prediction
Uses scikit-learn for training and predicting fraud risk
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class FraudMLModel:
    """
    Machine Learning model for fraud detection
    Trains on historical claim data and predicts fraud probability
    """
    
    def __init__(self, model_path='models/fraud_model.pkl'):
        """Initialize ML model"""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            'claim_amount',
            'policy_limit',
            'limit_utilization',
            'past_claims_amount',
            'claim_history_count',
            'claim_to_limit_ratio',
            'days_to_expiry',
            'claim_frequency_rate',
            'round_amount_flag',
            'high_value_flag'
        ]
        
        # Try to load existing model
        self.load_model()
    
    def extract_features(self, claim_data, policy_data):
        """
        Extract numerical features from claim and policy data
        
        Args:
            claim_data: Dictionary with claim information
            policy_data: Dictionary with policy validation data
            
        Returns:
            numpy array of features
        """
        try:
            claim_info = claim_data.get('claim_info', {})
            validation_details = policy_data.get('validation', {}).get('details', {})
            
            # Extract values
            claim_amount = float(claim_info.get('claim_amount', 0))
            policy_limit = float(validation_details.get('policy_limit', 1))
            past_claims = float(validation_details.get('past_claims_amount', 0))
            claim_history = int(validation_details.get('claim_history_count', 0))
            claim_date = claim_info.get('claim_date', '')
            policy_expiry = validation_details.get('policy_expiry_date', '')
            
            # Calculate derived features
            limit_utilization = (claim_amount / policy_limit * 100) if policy_limit > 0 else 0
            claim_to_limit_ratio = claim_amount / policy_limit if policy_limit > 0 else 0
            
            # Calculate days to expiry
            days_to_expiry = -1
            if claim_date and policy_expiry:
                try:
                    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']
                    for fmt in date_formats:
                        try:
                            claim_dt = datetime.strptime(claim_date, fmt)
                            expiry_dt = datetime.strptime(policy_expiry, fmt)
                            days_to_expiry = (expiry_dt - claim_dt).days
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # Claim frequency rate (claims per policy value)
            claim_frequency_rate = claim_history / (policy_limit / 10000) if policy_limit > 0 else 0
            
            # Binary flags
            round_amount_flag = 1 if (claim_amount > 0 and claim_amount % 1000 == 0 and claim_amount >= 10000) else 0
            high_value_flag = 1 if claim_amount > 100000 else 0
            
            # Build feature vector
            features = np.array([
                claim_amount,
                policy_limit,
                limit_utilization,
                past_claims,
                claim_history,
                claim_to_limit_ratio,
                days_to_expiry,
                claim_frequency_rate,
                round_amount_flag,
                high_value_flag
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            # Return default features (all zeros)
            return np.zeros((1, len(self.feature_names)))
    
    def predict_fraud(self, claim_data, policy_data):
        """
        Predict fraud probability using ML model
        
        Args:
            claim_data: Claim information
            policy_data: Policy validation data
            
        Returns:
            dict with fraud probability and risk level
        """
        try:
            # Extract features
            features = self.extract_features(claim_data, policy_data)
            
            if self.model is None:
                # If no model loaded, use rule-based fallback
                return self._rule_based_fallback(features)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Predict fraud probability
            fraud_prob = self.model.predict_proba(features_scaled)[0][1]  # Probability of fraud (class 1)
            fraud_prediction = self.model.predict(features_scaled)[0]
            
            # Convert probability to risk score (0-100)
            risk_score = int(fraud_prob * 100)
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = "HIGH"
            elif risk_score >= 50:
                risk_level = "MEDIUM"
            elif risk_score >= 30:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for name, importance in zip(self.feature_names, self.model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            return {
                "ml_fraud_probability": fraud_prob,
                "ml_risk_score": risk_score,
                "ml_risk_level": risk_level,
                "ml_prediction": "FRAUD" if fraud_prediction == 1 else "LEGITIMATE",
                "ml_confidence": max(fraud_prob, 1 - fraud_prob) * 100,
                "feature_importance": feature_importance,
                "model_used": type(self.model).__name__ if self.model else "None"
            }
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return {
                "ml_fraud_probability": 0.0,
                "ml_risk_score": 0,
                "ml_risk_level": "UNKNOWN",
                "ml_prediction": "ERROR",
                "ml_confidence": 0,
                "error": str(e)
            }
    
    def _rule_based_fallback(self, features):
        """
        Simple rule-based fallback when ML model is not available
        """
        # Extract key features
        limit_utilization = features[0][2]
        claim_history = features[0][4]
        days_to_expiry = features[0][6]
        round_flag = features[0][8]
        high_value_flag = features[0][9]
        
        # Simple rule-based scoring
        risk_score = 0
        
        if limit_utilization > 90:
            risk_score += 30
        elif limit_utilization > 80:
            risk_score += 20
        
        if claim_history >= 3:
            risk_score += 25
        
        if 0 <= days_to_expiry <= 30:
            risk_score += 20
        
        if round_flag == 1:
            risk_score += 15
        
        if high_value_flag == 1:
            risk_score += 10
        
        fraud_prob = risk_score / 100.0
        
        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 50:
            risk_level = "MEDIUM"
        elif risk_score >= 30:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "ml_fraud_probability": fraud_prob,
            "ml_risk_score": risk_score,
            "ml_risk_level": risk_level,
            "ml_prediction": "FRAUD" if fraud_prob >= 0.5 else "LEGITIMATE",
            "ml_confidence": 70,
            "model_used": "Rule-based fallback"
        }
    
    def train_model(self, training_data_path=None):
        """
        Train ML model on historical fraud data
        
        Args:
            training_data_path: Path to CSV with training data
        """
        print("🔧 Training fraud detection ML model...")
        
        # Generate synthetic training data if no file provided
        if training_data_path is None or not os.path.exists(training_data_path):
            print("📊 Generating synthetic training data...")
            X_train, y_train = self._generate_synthetic_data(n_samples=1000)
        else:
            # Load from CSV
            df = pd.read_csv(training_data_path)
            X_train = df[self.feature_names].values
            y_train = df['is_fraud'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest model
        print("🌲 Training Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        print(f"✅ Model trained with accuracy: {train_accuracy:.2%}")
        
        # Save model
        self.save_model()
        
        return train_accuracy
    
    def _generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic fraud training data"""
        np.random.seed(42)
        
        # Generate legitimate claims (70%)
        n_legit = int(n_samples * 0.7)
        n_fraud = n_samples - n_legit
        
        # Legitimate claims
        legit_features = []
        for _ in range(n_legit):
            claim_amount = np.random.uniform(5000, 80000)
            policy_limit = np.random.uniform(100000, 500000)
            limit_util = (claim_amount / policy_limit) * 100
            past_claims = np.random.uniform(0, policy_limit * 0.3)
            claim_history = np.random.randint(0, 3)
            claim_ratio = claim_amount / policy_limit
            days_to_expiry = np.random.randint(30, 365)
            freq_rate = claim_history / (policy_limit / 10000)
            round_flag = 0
            high_value = 1 if claim_amount > 100000 else 0
            
            legit_features.append([
                claim_amount, policy_limit, limit_util, past_claims,
                claim_history, claim_ratio, days_to_expiry, freq_rate,
                round_flag, high_value
            ])
        
        # Fraudulent claims (30%)
        fraud_features = []
        for _ in range(n_fraud):
            claim_amount = np.random.uniform(80000, 200000)  # Higher amounts
            policy_limit = np.random.uniform(100000, 300000)
            limit_util = np.random.uniform(85, 100)  # Near limit
            past_claims = np.random.uniform(policy_limit * 0.4, policy_limit * 0.8)
            claim_history = np.random.randint(2, 5)  # More claims
            claim_ratio = claim_amount / policy_limit
            days_to_expiry = np.random.randint(0, 45)  # Near expiry
            freq_rate = claim_history / (policy_limit / 10000)
            round_flag = 1 if np.random.random() > 0.5 else 0
            high_value = 1 if claim_amount > 100000 else 0
            
            fraud_features.append([
                claim_amount, policy_limit, limit_util, past_claims,
                claim_history, claim_ratio, days_to_expiry, freq_rate,
                round_flag, high_value
            ])
        
        X = np.array(legit_features + fraud_features)
        y = np.array([0] * n_legit + [1] * n_fraud)
        
        # Shuffle
        indices = np.random.permutation(len(y))
        return X[indices], y[indices]
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Model saved to {self.model_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', self.feature_names)
                
                print(f"✅ Model loaded from {self.model_path}")
                return True
            else:
                print("⚠️ No pre-trained model found. Will train on first use.")
                # Auto-train with synthetic data
                self.train_model()
                return False
                
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            return False


# Singleton instance
_ml_model_instance = None

def get_fraud_ml_model():
    """Get singleton instance of ML model"""
    global _ml_model_instance
    if _ml_model_instance is None:
        _ml_model_instance = FraudMLModel()
    return _ml_model_instance


if __name__ == "__main__":
    # Test the ML model
    print("🧪 Testing Fraud Detection ML Model\n")
    
    ml_model = FraudMLModel()
    
    # Test prediction with sample data
    test_claim = {
        'claim_info': {
            'claim_amount': 95000,
            'claim_date': '2025-11-15'
        }
    }
    
    test_policy = {
        'validation': {
            'details': {
                'policy_limit': 100000,
                'past_claims_amount': 30000,
                'claim_history_count': 3,
                'policy_expiry_date': '2025-12-01'
            }
        }
    }
    
    result = ml_model.predict_fraud(test_claim, test_policy)
    
    print("\n📊 ML Prediction Results:")
    print(f"Fraud Probability: {result['ml_fraud_probability']:.2%}")
    print(f"Risk Score: {result['ml_risk_score']}/100")
    print(f"Risk Level: {result['ml_risk_level']}")
    print(f"Prediction: {result['ml_prediction']}")
    print(f"Confidence: {result['ml_confidence']:.1f}%")
    print(f"Model: {result['model_used']}")
    
    if result.get('feature_importance'):
        print("\n🔍 Top Feature Importance:")
        sorted_features = sorted(
            result['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for feature, importance in sorted_features:
            print(f"  • {feature}: {importance:.3f}")

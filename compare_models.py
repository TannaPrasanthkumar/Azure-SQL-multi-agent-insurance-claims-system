"""
Compare different model versions to find which one gives 93.3% accuracy
"""
import os
import joblib
import pandas as pd
import numpy as np

# Load test dataset
df = pd.read_csv('ML/data/test_dataset_15_samples.csv')

# Encode categorical columns
category_mapping = {
    'AccidentArea': {'Urban': 'Urban', 'Rural': 'Rural'},
    'Sex': {'Male': 'Male', 'Female': 'Female'},
    'PolicyType': {
        'Sedan - Collision': 'Sedan - Collision',
        'Sedan - All Perils': 'Sedan - All Perils',
        'Sport - Collision': 'Sport - Collision',
        'Sedan - Liability': 'Sedan - Liability',
        'Utility - All Perils': 'Utility - All Perils',
        'Sport - All Perils': 'Sport - All Perils',
        'Utility - Collision': 'Utility - Collision'
    },
    'PoliceReportFiled': {'Yes': 'Yes', 'No': 'No'}
}

for col, mapping in category_mapping.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

print("=" * 80)
print("COMPARING MODEL VERSIONS")
print("=" * 80)

# Model locations to test
model_locations = [
    {
        'name': 'ML/New folder (Deployed)',
        'model': 'ML/New folder/balanced_random_forest_fraud_detector_0.pkl',
        'encoders': 'ML/New folder/label_encoders_0.pkl',
        'scaler': 'ML/New folder/scaler_0.pkl',
        'metadata': 'ML/New folder/model_metadata_0.pkl'
    },
    {
        'name': 'ML/actual_model',
        'model': 'ML/actual_model/balanced_random_forest_fraud_detector_2025-11-21.pkl',
        'encoders': 'ML/actual_model/label_encoders.pkl',
        'scaler': 'ML/actual_model/scaler.pkl',
        'metadata': 'ML/actual_model/model_metadata.pkl'
    },
    {
        'name': 'ML/models',
        'model': 'ML/models/balanced_random_forest_fraud_detector.pkl',
        'encoders': 'ML/models/label_encoders.pkl',
        'scaler': 'ML/models/scaler.pkl',
        'metadata': 'ML/models/model_metadata.pkl'
    },
    {
        'name': 'ML/new_model',
        'model': 'ML/new_model/balanced_random_forest_fraud_detector.pkl',
        'encoders': 'ML/new_model/label_encoders.pkl',
        'scaler': 'ML/new_model/scaler.pkl',
        'metadata': None
    }
]

def test_model(location):
    """Test a model and return accuracy"""
    try:
        # Load model files
        model = joblib.load(location['model'])
        encoders = joblib.load(location['encoders'])
        scaler = joblib.load(location['scaler'])
        
        # Load metadata if available
        threshold = 0.5
        feature_order = None
        if location['metadata'] and os.path.exists(location['metadata']):
            metadata = joblib.load(location['metadata'])
            threshold = metadata.get('optimal_threshold', 0.5)
            feature_order = metadata.get('features')
        
        # Prepare data
        df_test = df.copy()
        
        # Encode categorical features
        categorical_cols = ['AccidentArea', 'Sex', 'PolicyType', 'PoliceReportFiled']
        for col in categorical_cols:
            if col in encoders and col in df_test.columns:
                encoder = encoders[col]
                df_test[col] = df_test[col].astype(str)
                known_classes = set(encoder.classes_)
                df_test[col] = df_test[col].apply(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )
                df_test[col] = encoder.transform(df_test[col])
        
        # Get features
        if feature_order:
            features = df_test[feature_order]
        else:
            # Default order
            feature_cols = ['DriverRating', 'Age', 'WeekOfMonthClaimed', 'WeekOfMonth', 
                          'Deductible', 'AccidentArea', 'Sex', 'PolicyType', 'PoliceReportFiled']
            features = df_test[feature_cols]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        probabilities = model.predict_proba(features_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate accuracy
        actual = df['FraudFound_P'].values
        correct = (predictions == actual).sum()
        accuracy = correct / len(actual) * 100
        
        return {
            'success': True,
            'accuracy': accuracy,
            'threshold': threshold,
            'correct': correct,
            'total': len(actual),
            'predictions': predictions,
            'probabilities': probabilities,
            'feature_order': feature_order
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Test each model
results = []
for loc in model_locations:
    print(f"\n{'=' * 80}")
    print(f"Testing: {loc['name']}")
    print(f"{'=' * 80}")
    
    if not os.path.exists(loc['model']):
        print(f"❌ Model file not found: {loc['model']}")
        continue
    
    result = test_model(loc)
    
    if result['success']:
        accuracy = result['accuracy']
        print(f"✅ Model loaded successfully")
        print(f"   Threshold: {result['threshold']}")
        print(f"   Feature order: {result['feature_order']}")
        print(f"   Accuracy: {result['correct']}/{result['total']} = {accuracy:.1f}%")
        
        if accuracy >= 90:
            print(f"\n🎯 THIS IS THE HIGH-ACCURACY MODEL! ({accuracy:.1f}%)")
            
            # Show sample predictions
            print(f"\n   Sample predictions:")
            for i in [3, 8, 12]:  # Samples 4, 9, 13 (problem samples)
                idx = i
                prob = result['probabilities'][idx]
                pred = result['predictions'][idx]
                actual = df['FraudFound_P'].values[idx]
                status = "✅" if pred == actual else "❌"
                print(f"   {status} Sample {idx+1}: Prob={prob*100:.1f}%, Pred={pred}, Actual={actual}")
        
        results.append({
            'name': loc['name'],
            'accuracy': accuracy,
            'threshold': result['threshold']
        })
    else:
        print(f"❌ Error: {result['error']}")

# Summary
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

if results:
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\nModels ranked by accuracy:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}: {r['accuracy']:.1f}% (threshold={r['threshold']})")
    
    best = results[0]
    if best['accuracy'] >= 90:
        print(f"\n🎯 FOUND: '{best['name']}' gives {best['accuracy']:.1f}% accuracy")
        print(f"   This is the model that should be deployed to Azure ML!")
    else:
        print(f"\n⚠️  No model achieved 93.3% accuracy")
        print(f"   Best model: '{best['name']}' with {best['accuracy']:.1f}%")
else:
    print("No models could be loaded successfully")

print(f"\n{'=' * 80}")

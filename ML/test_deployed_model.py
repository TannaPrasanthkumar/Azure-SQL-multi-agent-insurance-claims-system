"""
Test the deployed model with test_dataset_15_samples.csv
This script loads the model artifacts and runs predictions
"""
import pandas as pd
import json
from test import init, run

print("="*60)
print("TESTING DEPLOYED MODEL WITH TEST DATASET")
print("="*60)

# Initialize the model (loads from actual_model/)
print("\n🔄 Initializing model...")
init()

# Load test dataset
test_data_path = "data/test_dataset_15_samples.csv"
df_test = pd.read_csv(test_data_path)
print(f"\n✅ Test data loaded: {df_test.shape}")
print(f"   Fraud cases: {df_test['FraudFound_P'].sum()} out of {len(df_test)}")

# Get the features needed for the model
features = ['DriverRating', 'Age', 'PoliceReportFiled', 'WeekOfMonthClaimed', 
            'PolicyType', 'WeekOfMonth', 'AccidentArea', 'Sex', 'Deductible']

# Prepare test data (exclude target column)
X_test = df_test[features].copy()
y_test = df_test['FraudFound_P']

print("\n" + "="*60)
print("RUNNING PREDICTIONS ON 15 SAMPLES")
print("="*60)

correct_predictions = 0
total_samples = len(df_test)

# Run predictions for each sample
for i in range(len(df_test)):
    # Get sample as dictionary
    sample = X_test.iloc[i].to_dict()
    
    # Convert to JSON string (as the scoring script expects)
    sample_json = json.dumps(sample)
    
    # Get prediction
    result = run(sample_json)
    result_dict = json.loads(result)
    
    # Extract prediction and actual
    pred = result_dict['predictions'][0]
    actual = y_test.iloc[i]
    predicted = pred['fraud_prediction']
    probability = pred['fraud_probability']
    risk = pred['fraud_risk']
    
    # Check if correct
    is_correct = (predicted == actual)
    if is_correct:
        correct_predictions += 1
    
    match_symbol = "✅" if is_correct else "❌"
    actual_label = "FRAUD" if actual == 1 else "NOT FRAUD"
    pred_label = "FRAUD" if predicted == 1 else "NOT FRAUD"
    
    print(f"\nSample {i+1:2d}: {match_symbol}")
    print(f"  Actual:      {actual_label}")
    print(f"  Predicted:   {pred_label}")
    print(f"  Probability: {probability:.4f}")
    print(f"  Risk Level:  {risk}")

# Calculate metrics
accuracy = correct_predictions / total_samples
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Re-run to calculate confusion matrix
for i in range(len(df_test)):
    sample = X_test.iloc[i].to_dict()
    sample_json = json.dumps(sample)
    result = run(sample_json)
    result_dict = json.loads(result)
    
    actual = y_test.iloc[i]
    predicted = result_dict['predictions'][0]['fraud_prediction']
    
    if actual == 1 and predicted == 1:
        true_positives += 1
    elif actual == 0 and predicted == 1:
        false_positives += 1
    elif actual == 0 and predicted == 0:
        true_negatives += 1
    elif actual == 1 and predicted == 0:
        false_negatives += 1

# Calculate metrics
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*60)
print("DEPLOYED MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"Total Samples:      {total_samples}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  True Negatives:   {true_negatives}")
print(f"  False Positives:  {false_positives}")
print(f"  False Negatives:  {false_negatives}")
print(f"  True Positives:   {true_positives}")
print(f"\nMetrics:")
print(f"  Precision:        {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:           {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:         {f1:.4f}")
print("="*60)

# Summary
print("\n✅ Testing completed!")
print(f"   Model accuracy: {accuracy*100:.2f}%")
print(f"   Threshold used: {result_dict['predictions'][0]['threshold_used']}")

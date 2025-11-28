"""
Test Azure ML Deployed Endpoint with test_dataset_15_samples.csv
Calls the real-time inference endpoint in Azure ML
"""
import pandas as pd
import json
import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

print("="*60)
print("TESTING AZURE ML DEPLOYED ENDPOINT")
print("="*60)

# Azure ML Configuration
subscription_id = "94a0d91a-896b-4e4d-a3c7-df06074221d2"
resource_group = "AgenticAI-ML"
workspace_name = "AgenticAI_ML"

# Initialize Azure ML Client
print("\n🔄 Connecting to Azure ML Workspace...")
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
print("✅ Connected to Azure ML Workspace")

# Get the endpoint (you need to replace with your endpoint name)
endpoint_name = input("\nEnter your endpoint name (e.g., fraud-detection-endpoint): ").strip()

try:
    # Get endpoint details
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    print(f"\n✅ Endpoint found: {endpoint.name}")
    print(f"   Scoring URI: {endpoint.scoring_uri}")
    print(f"   State: {endpoint.provisioning_state}")
    
    # Get the key for authentication
    keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
    api_key = keys.primary_key
    print(f"   Authentication: API Key retrieved")
    
except Exception as e:
    print(f"\n❌ Error getting endpoint: {e}")
    print("\nTip: List your endpoints with:")
    print("   from azure.ai.ml import MLClient")
    print("   endpoints = ml_client.online_endpoints.list()")
    print("   for ep in endpoints: print(ep.name)")
    exit(1)

# Load test dataset
print("\n📊 Loading test dataset...")
test_data_path = "data/test_dataset_15_samples.csv"
df_test = pd.read_csv(test_data_path)
print(f"✅ Test data loaded: {df_test.shape}")
print(f"   Fraud cases: {df_test['FraudFound_P'].sum()} out of {len(df_test)}")

# Prepare features
features = ['DriverRating', 'Age', 'PoliceReportFiled', 'WeekOfMonthClaimed', 
            'PolicyType', 'WeekOfMonth', 'AccidentArea', 'Sex', 'Deductible']

X_test = df_test[features].copy()
y_test = df_test['FraudFound_P']

# Prepare headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

print("\n" + "="*60)
print("RUNNING PREDICTIONS ON 15 SAMPLES")
print("="*60)

correct_predictions = 0
predictions_list = []

# Call endpoint for each sample
for i in range(len(df_test)):
    sample = X_test.iloc[i].to_dict()
    
    # Format request body (adjust based on your scoring script format)
    request_body = json.dumps(sample)
    
    try:
        # Make API call
        response = requests.post(
            endpoint.scoring_uri,
            data=request_body,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract prediction (adjust based on your scoring script output format)
            if 'predictions' in result:
                pred_data = result['predictions'][0]
                predicted = pred_data['fraud_prediction']
                probability = pred_data['fraud_probability']
                risk = pred_data.get('fraud_risk', 'N/A')
                threshold = pred_data.get('threshold_used', 'N/A')
            else:
                # Fallback parsing
                predicted = result.get('fraud_prediction', 0)
                probability = result.get('fraud_probability', 0.0)
                risk = result.get('fraud_risk', 'N/A')
                threshold = result.get('threshold_used', 'N/A')
            
            actual = y_test.iloc[i]
            is_correct = (predicted == actual)
            if is_correct:
                correct_predictions += 1
            
            predictions_list.append({
                'sample': i+1,
                'actual': actual,
                'predicted': predicted,
                'probability': probability,
                'correct': is_correct
            })
            
            match_symbol = "✅" if is_correct else "❌"
            actual_label = "FRAUD" if actual == 1 else "NOT FRAUD"
            pred_label = "FRAUD" if predicted == 1 else "NOT FRAUD"
            
            print(f"\nSample {i+1:2d}: {match_symbol}")
            print(f"  Actual:      {actual_label}")
            print(f"  Predicted:   {pred_label}")
            print(f"  Probability: {probability:.4f}")
            print(f"  Risk Level:  {risk}")
            if i == 0:  # Show threshold on first sample
                print(f"  Threshold:   {threshold}")
        else:
            print(f"\n❌ Sample {i+1}: API Error {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"\n❌ Sample {i+1}: Error - {str(e)}")

# Calculate metrics
accuracy = correct_predictions / len(df_test)
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for pred in predictions_list:
    actual = pred['actual']
    predicted = pred['predicted']
    
    if actual == 1 and predicted == 1:
        true_positives += 1
    elif actual == 0 and predicted == 1:
        false_positives += 1
    elif actual == 0 and predicted == 0:
        true_negatives += 1
    elif actual == 1 and predicted == 0:
        false_negatives += 1

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*60)
print("AZURE ML ENDPOINT PERFORMANCE SUMMARY")
print("="*60)
print(f"Endpoint Name:      {endpoint_name}")
print(f"Total Samples:      {len(df_test)}")
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

print("\n✅ Azure ML Endpoint Testing completed!")
print(f"   Endpoint: {endpoint_name}")
print(f"   Model accuracy: {accuracy*100:.2f}%")

"""
Test all 15 samples from test_dataset_15_samples.csv with threshold 0.7
"""
import pandas as pd
import json
import sys
from pathlib import Path

# Add New folder to path
sys.path.insert(0, str(Path(__file__).parent / "New folder"))

from scoring import init, run

# Initialize model
print("Initializing model...")
init()
print()

# Load test dataset
df = pd.read_csv('data/test_dataset_15_samples.csv')

print(f"Testing {len(df)} samples with threshold 0.7")
print("="*80)

correct = 0
total = len(df)
results = []

for idx, row in df.iterrows():
    # Prepare input data
    data = {
        "DriverRating": row["DriverRating"],
        "Age": row["Age"],
        "PoliceReportFiled": row["PoliceReportFiled"],
        "WeekOfMonthClaimed": row["WeekOfMonthClaimed"],
        "PolicyType": row["PolicyType"],
        "WeekOfMonth": row["WeekOfMonth"],
        "AccidentArea": row["AccidentArea"],
        "Sex": row["Sex"],
        "Deductible": row["Deductible"]
    }
    
    # Get prediction
    result = json.loads(run(json.dumps(data)))
    pred = result['predictions'][0]
    
    # Get actual value
    actual = row.get('FraudFound_P', row.get('FraudFound', 'Unknown'))
    
    # Check if correct
    is_correct = str(pred['fraud_prediction']) == str(actual)
    if is_correct:
        correct += 1
    
    # Display result
    status = "✅" if is_correct else "❌"
    print(f"{status} Sample {idx+1:2d}: Prob={pred['fraud_probability']:5.1%} | "
          f"Pred={pred['fraud_prediction']} | Actual={actual} | {pred['fraud_risk']}")
    
    results.append({
        'sample': idx + 1,
        'probability': pred['fraud_probability'],
        'prediction': pred['fraud_prediction'],
        'actual': actual,
        'correct': is_correct
    })

print("="*80)
print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")
print(f"\nWith threshold 0.7:")
print(f"  - Fraud detected (≥70%): {sum(1 for r in results if r['prediction'] == 1)} samples")
print(f"  - Not fraud (<70%): {sum(1 for r in results if r['prediction'] == 0)} samples")

"""
Simple batch fraud check using the test dataset
The test dataset corresponds to the 15 PDFs (rows 1-15 map to 1.pdf-15.pdf)
"""
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_API_KEY = os.getenv("AZURE_ML_API_KEY")

# Load test dataset
df = pd.read_csv('ML/data/test_dataset_15_samples.csv')

print("=" * 80)
print("FRAUD DETECTION FOR 15 PDF FILES")
print("=" * 80)
print("(Using test dataset - each row corresponds to a PDF file)")
print()

fraud_files = []
no_fraud_files = []

for idx, row in df.iterrows():
    pdf_num = idx + 1
    pdf_name = f"{pdf_num}.pdf"
    
    print(f"{'=' * 80}")
    print(f"File {pdf_num}/15: {pdf_name}")
    print(f"{'=' * 80}")
    
    # Prepare data - send categorical as strings
    data = {
        "DriverRating": int(row["DriverRating"]),
        "Age": int(row["Age"]),
        "WeekOfMonthClaimed": int(row["WeekOfMonthClaimed"]),
        "WeekOfMonth": int(row["WeekOfMonth"]),
        "Deductible": int(row["Deductible"]),
        "AccidentArea": str(row["AccidentArea"]),
        "Sex": str(row["Sex"]),
        "PolicyType": str(row["PolicyType"]),
        "PoliceReportFiled": str(row["PoliceReportFiled"])
    }
    
    actual_fraud = int(row["FraudFound_P"])
    
    print(f"\nClaim Details:")
    print(f"  Driver Rating: {data['DriverRating']} (1=Poor, 4=Excellent)")
    print(f"  Age: {data['Age']}")
    print(f"  Police Report: {data['PoliceReportFiled']}")
    print(f"  Policy Type: {data['PolicyType']}")
    print(f"  Accident Area: {data['AccidentArea']}")
    print(f"  Sex: {data['Sex']}")
    print(f"  Deductible: ${data['Deductible']}")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AZURE_ML_API_KEY}"
        }
        
        response = requests.post(
            AZURE_ML_ENDPOINT,
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, str):
                result = json.loads(result)
            
            pred_data = result["predictions"][0]
            fraud_prob = pred_data["fraud_probability"]
            fraud_pred = pred_data["fraud_prediction"]
            threshold = pred_data["threshold_used"]
            risk_level = pred_data["fraud_risk"]
            
            print(f"\nFraud Detection:")
            print(f"  Probability: {fraud_prob * 100:.1f}%")
            print(f"  Threshold: {threshold}")
            print(f"  Risk Level: {risk_level}")
            print(f"  Actual: {'FRAUD' if actual_fraud == 1 else 'NOT FRAUD'}")
            
            if fraud_pred == 1:
                print(f"\n  ** FRAUD DETECTED! **")
                fraud_files.append({
                    'file': pdf_name,
                    'number': pdf_num,
                    'probability': fraud_prob * 100,
                    'risk': risk_level,
                    'driver_rating': data['DriverRating'],
                    'age': data['Age'],
                    'actual': actual_fraud
                })
            else:
                print(f"\n  No fraud detected")
                no_fraud_files.append({
                    'file': pdf_name,
                    'number': pdf_num,
                    'probability': fraud_prob * 100,
                    'risk': risk_level,
                    'driver_rating': data['DriverRating'],
                    'age': data['Age'],
                    'actual': actual_fraud
                })
        else:
            print(f"  ERROR: Status {response.status_code}")
            
    except Exception as e:
        print(f"  ERROR: {e}")

# Summary
print("\n" + "=" * 80)
print("FRAUD DETECTION SUMMARY")
print("=" * 80)

print(f"\n** FRAUD DETECTED ({len(fraud_files)} files):")
print("=" * 80)
if fraud_files:
    for record in sorted(fraud_files, key=lambda x: x['probability'], reverse=True):
        match = " (CORRECT)" if record['actual'] == 1 else " (FALSE POSITIVE)"
        print(f"\n  {record['file']}{match}")
        print(f"    Probability: {record['probability']:.1f}%")
        print(f"    Risk Level: {record['risk']}")
        print(f"    Driver Rating: {record['driver_rating']}, Age: {record['age']}")
else:
    print("  None")

print(f"\n** NO FRAUD ({len(no_fraud_files)} files):")
print("=" * 80)
if no_fraud_files:
    for record in sorted(no_fraud_files, key=lambda x: x['probability'], reverse=True):
        match = " (CORRECT)" if record['actual'] == 0 else " (MISSED FRAUD)"
        print(f"\n  {record['file']}{match}")
        print(f"    Probability: {record['probability']:.1f}%")
        print(f"    Risk Level: {record['risk']}")
        print(f"    Driver Rating: {record['driver_rating']}, Age: {record['age']}")

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
total = len(fraud_files) + len(no_fraud_files)
print(f"Total Files: {total}")
print(f"Fraud Detected: {len(fraud_files)} ({len(fraud_files)/total*100:.1f}%)")
print(f"No Fraud: {len(no_fraud_files)} ({len(no_fraud_files)/total*100:.1f}%)")

if fraud_files or no_fraud_files:
    all_probs = [r['probability'] for r in fraud_files + no_fraud_files]
    print(f"\nProbability Range:")
    print(f"  Highest: {max(all_probs):.1f}%")
    print(f"  Lowest: {min(all_probs):.1f}%")
    print(f"  Average: {sum(all_probs)/len(all_probs):.1f}%")

# Calculate accuracy
correct = 0
for r in fraud_files:
    if r['actual'] == 1:
        correct += 1
for r in no_fraud_files:
    if r['actual'] == 0:
        correct += 1

accuracy = correct / total * 100
print(f"\nModel Accuracy: {correct}/{total} = {accuracy:.1f}%")

print("\n" + "=" * 80)
print("FRAUD FILES (for easy reference):")
print("=" * 80)
if fraud_files:
    fraud_nums = [r['number'] for r in fraud_files]
    fraud_names = [r['file'] for r in fraud_files]
    print(f"File numbers: {', '.join(map(str, fraud_nums))}")
    print(f"File names: {', '.join(fraud_names)}")
else:
    print("None detected")

print("=" * 80)

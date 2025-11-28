"""
Process all 15 PDF files and check which ones are flagged as fraud by the ML model
"""
import os
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from fraud_detector_agent import FraudDetectorAgent

load_dotenv()

# Initialize clients
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
document_client = DocumentAnalysisClient(
    endpoint=doc_intelligence_endpoint,
    credential=AzureKeyCredential(doc_intelligence_key)
)

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_AISERVICES_APIKEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_AISERVICES_ENDPOINT")
)

fraud_agent = FraudDetectorAgent()

def extract_claim_data(pdf_path):
    """Extract claim data from PDF"""
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    
    # Analyze document
    poller = document_client.begin_analyze_document("prebuilt-document", file_bytes)
    result = poller.result()
    
    # Extract text
    extracted_text = ""
    for page in result.pages:
        for line in page.lines:
            extracted_text += line.content + "\n"
    
    # Extract key-value pairs
    key_value_pairs = {}
    if result.key_value_pairs:
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_text = kv_pair.key.content if hasattr(kv_pair.key, 'content') else str(kv_pair.key)
                value_text = kv_pair.value.content if hasattr(kv_pair.value, 'content') else str(kv_pair.value)
                key_value_pairs[key_text] = value_text
    
    # Use AI to extract structured data
    deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
    prompt = f"""Extract the following information from this insurance claim document:

1. Policy Number
2. Policyholder Name
3. Claim Amount (numeric value only)
4. Driver Rating (1-4)
5. Age
6. Police Report Filed (0=No, 1=Yes)
7. Week of Month Claimed (1-5)
8. Policy Type (1=Sedan, 2=Sport-Utility, etc.)
9. Accident Area (1=Urban, 2=Rural)
10. Sex (0=Female, 1=Male)
11. Deductible
12. Week of Month (1-5)

Text:
{extracted_text[:2000]}

Return ONLY a JSON object with keys: policy_number, policyholder_name, claim_amount, driver_rating, age, police_report_filed, week_of_month_claimed, policy_type, accident_area, sex, deductible, week_of_month
"""
    
    response = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.1
    )
    
    result_text = response.choices[0].message.content.strip()
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    if result_text.startswith("```"):
        result_text = result_text[3:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]
    
    return json.loads(result_text.strip())

print("=" * 80)
print("PROCESSING 15 PDF FILES FOR FRAUD DETECTION")
print("=" * 80)

print("=" * 80)
print("PROCESSING 15 PDF FILES FOR FRAUD DETECTION")
print("=" * 80)

pdf_folder = "c:/Projects/DEMO/data"
pdf_files = [f"{i}.pdf" for i in range(1, 16)]

fraud_results = []
no_fraud_results = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ File not found: {pdf_file}")
        continue
    
    print(f"\n{'=' * 80}")
    print(f"Processing: {pdf_file}")
    print(f"{'=' * 80}")
    
    try:
        # Extract data from PDF
        print(f"   Extracting data...")
        claim_info = extract_claim_data(pdf_path)
        
        # Get key info
        policy_number = claim_info.get("policy_number", "Unknown")
        claim_amount = claim_info.get("claim_amount", 0)
        policyholder = claim_info.get("policyholder_name", "Unknown")
        
        print(f"\n📄 Claim Details:")
        print(f"   Policy: {policy_number}")
        print(f"   Holder: {policyholder}")
        print(f"   Amount: ${claim_amount:,}")
        
        # Prepare fraud detection data
        fraud_data = {
            "DriverRating": claim_info.get("driver_rating", 1),
            "Age": claim_info.get("age", 30),
            "PoliceReportFiled": claim_info.get("police_report_filed", 0),
            "WeekOfMonthClaimed": claim_info.get("week_of_month_claimed", 1),
            "PolicyType": claim_info.get("policy_type", 1),
            "WeekOfMonth": claim_info.get("week_of_month", 1),
            "AccidentArea": claim_info.get("accident_area", 1),
            "Sex": claim_info.get("sex", 1),
            "Deductible": claim_info.get("deductible", 400)
        }
        
        print(f"\n🔍 Fraud Detection Data:")
        for key, value in fraud_data.items():
            print(f"   {key}: {value}")
        
        # Run fraud detection
        fraud_result = fraud_agent.detect_fraud(fraud_data)
        
        fraud_prob = fraud_result.get("fraud_probability", 0)
        is_fraud = fraud_result.get("fraud_detected", False)
        risk_level = fraud_result.get("fraud_risk", "Unknown")
        threshold = fraud_result.get("threshold_used", 0.65)
        
        print(f"\n📊 Fraud Detection Result:")
        print(f"   Probability: {fraud_prob * 100:.1f}%")
        print(f"   Threshold: {threshold}")
        print(f"   Risk Level: {risk_level}")
        
        record = {
            "file": pdf_file,
            "policy_number": policy_number,
            "policyholder": policyholder,
            "claim_amount": claim_amount,
            "fraud_probability": fraud_prob * 100,
            "is_fraud": is_fraud,
            "risk_level": risk_level
        }
        
        if is_fraud:
            print(f"\n🚨 FRAUD DETECTED!")
            print(f"   ⚠️  Probability {fraud_prob * 100:.1f}% >= Threshold {threshold}")
            fraud_results.append(record)
        else:
            print(f"\n✅ NO FRAUD")
            print(f"   ✓  Probability {fraud_prob * 100:.1f}% < Threshold {threshold}")
            no_fraud_results.append(record)
            
    except Exception as e:
        print(f"❌ Error processing {pdf_file}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("FRAUD DETECTION SUMMARY")
print("=" * 80)

print(f"\n🚨 FRAUD DETECTED ({len(fraud_results)} files):")
print("=" * 80)
if fraud_results:
    for record in sorted(fraud_results, key=lambda x: x['fraud_probability'], reverse=True):
        print(f"\n📄 {record['file']}")
        print(f"   Policy: {record['policy_number']}")
        print(f"   Holder: {record['policyholder']}")
        print(f"   Amount: ${record['claim_amount']:,}")
        print(f"   Probability: {record['fraud_probability']:.1f}%")
        print(f"   Risk: {record['risk_level']}")
else:
    print("   None")

print(f"\n✅ NO FRAUD DETECTED ({len(no_fraud_results)} files):")
print("=" * 80)
if no_fraud_results:
    for record in sorted(no_fraud_results, key=lambda x: x['fraud_probability'], reverse=True):
        print(f"\n📄 {record['file']}")
        print(f"   Policy: {record['policy_number']}")
        print(f"   Holder: {record['policyholder']}")
        print(f"   Amount: ${record['claim_amount']:,}")
        print(f"   Probability: {record['fraud_probability']:.1f}%")
        print(f"   Risk: {record['risk_level']}")

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print(f"Total PDFs Processed: {len(fraud_results) + len(no_fraud_results)}")
print(f"Fraud Detected: {len(fraud_results)} ({len(fraud_results)/(len(fraud_results)+len(no_fraud_results))*100:.1f}%)")
print(f"No Fraud: {len(no_fraud_results)} ({len(no_fraud_results)/(len(fraud_results)+len(no_fraud_results))*100:.1f}%)")

if fraud_results or no_fraud_results:
    all_probs = [r['fraud_probability'] for r in fraud_results + no_fraud_results]
    print(f"\nFraud Probability Range:")
    print(f"   Highest: {max(all_probs):.1f}%")
    print(f"   Lowest: {min(all_probs):.1f}%")
    print(f"   Average: {sum(all_probs)/len(all_probs):.1f}%")

print("\n" + "=" * 80)

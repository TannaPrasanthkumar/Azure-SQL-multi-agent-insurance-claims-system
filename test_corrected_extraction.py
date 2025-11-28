"""Test AI extraction with corrected prompt"""
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
import json

load_dotenv()

# Initialize clients
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
document_client = DocumentAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_AISERVICES_APIKEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_AISERVICES_ENDPOINT")
)

# Read 4.pdf
pdf_path = "c:/Projects/DEMO/data/4.pdf"
print(f"Testing AI extraction for: {pdf_path}\n")

with open(pdf_path, "rb") as f:
    file_bytes = f.read()

# Extract text
poller = document_client.begin_analyze_document("prebuilt-layout", file_bytes)
result = poller.result()

extracted_text = ""
for page in result.pages:
    for line in page.lines:
        extracted_text += line.content + "\n"

# Use CORRECTED prompt (with text for categorical fields)
deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
prompt = f"""Extract the following information from this insurance claim document:

1. Policy Number
2. Policyholder Name
3. Claim Amount (numeric value only, no currency symbols)
4. Reason for Claim
5. Policy Type (exact text as in document, e.g., "Sedan - Liability", "Utility - All Perils")
6. Claim Date (preserve exact format as found, e.g., DD-MM-YYYY or YYYY-MM-DD)
7. Driver Rating (1-4, where 1=poor, 4=excellent)
8. Age (age of driver/policyholder)
9. Police Report Filed (text: "Yes" or "No")
10. Week of Month Claimed (1-5, week number when claim was filed)
11. Accident Area (text: "Urban" or "Rural")
12. Sex (text: "Male" or "Female")
13. Deductible (insurance deductible amount)
14. Week of Month (1-5, current week of month)

Important Instructions:
- For Policy Type: Extract EXACT text as it appears (e.g., "Sedan - Liability", "Utility - All Perils") - do NOT convert to numbers
- For Accident Area: Extract as text ("Urban" or "Rural") - do NOT convert to numbers
- For Sex: Extract as text ("Male" or "Female") - do NOT convert to numbers
- For Police Report Filed: Extract as text ("Yes" or "No") - do NOT convert to numbers
- For numeric fields (driver_rating, age, week_of_month_claimed, week_of_month, deductible): Extract as numbers

Extracted Text:
{extracted_text[:2000]}

Return ONLY a JSON object with these exact keys: policy_number, policyholder_name, claim_amount, reason_for_claim, policy_type, claim_date, driver_rating, age, police_report_filed, week_of_month_claimed, accident_area, sex, deductible, week_of_month
"""

response = openai_client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a data extraction expert. Extract information and return only valid JSON."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0.1,
    response_format={"type": "json_object"}
)

claim_info = json.loads(response.choices[0].message.content)

print("=" * 80)
print("AI EXTRACTED DATA (WITH CORRECTED PROMPT):")
print("=" * 80)
for key, value in claim_info.items():
    value_type = type(value).__name__
    print(f"  {key}: {value} ({value_type})")

print("\n" + "=" * 80)
print("VERIFICATION:")
print("=" * 80)
print(f"✓ policy_type is STRING: {isinstance(claim_info.get('policy_type'), str)}")
print(f"  Value: '{claim_info.get('policy_type')}'")
print(f"  Expected: 'Utility - All Perils'")
print(f"  Match: {claim_info.get('policy_type') == 'Utility - All Perils'}")

print(f"\n✓ accident_area is STRING: {isinstance(claim_info.get('accident_area'), str)}")
print(f"  Value: '{claim_info.get('accident_area')}'")
print(f"  Expected: 'Urban'")
print(f"  Match: {claim_info.get('accident_area') == 'Urban'}")

print(f"\n✓ sex is STRING: {isinstance(claim_info.get('sex'), str)}")
print(f"  Value: '{claim_info.get('sex')}'")
print(f"  Expected: 'Male'")
print(f"  Match: {claim_info.get('sex') == 'Male'}")

print(f"\n✓ police_report_filed is STRING: {isinstance(claim_info.get('police_report_filed'), str)}")
print(f"  Value: '{claim_info.get('police_report_filed')}'")
print(f"  Expected: 'No'")
print(f"  Match: {claim_info.get('police_report_filed') == 'No'}")

print(f"\n✓ week_of_month is NUMBER: {isinstance(claim_info.get('week_of_month'), int)}")
print(f"  Value: {claim_info.get('week_of_month')}")
print(f"  Expected: 5")
print(f"  Match: {claim_info.get('week_of_month') == 5}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
if (claim_info.get('policy_type') == 'Utility - All Perils' and
    claim_info.get('accident_area') == 'Urban' and
    claim_info.get('sex') == 'Male' and
    claim_info.get('week_of_month') == 5):
    print("✅ SUCCESS! All fields extracted correctly as strings/numbers")
    print("✅ This should now give ~75.9% fraud probability")
else:
    print("❌ FAILED! Some fields still incorrect")
    print("   Check the values above")

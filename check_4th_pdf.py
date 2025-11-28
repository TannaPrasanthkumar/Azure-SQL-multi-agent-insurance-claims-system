"""Check what data is in 4.pdf"""
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
print(f"Reading: {pdf_path}\n")

with open(pdf_path, "rb") as f:
    file_bytes = f.read()

# Extract text
poller = document_client.begin_analyze_document("prebuilt-layout", file_bytes)
result = poller.result()

extracted_text = ""
for page in result.pages:
    for line in page.lines:
        extracted_text += line.content + "\n"

print("="*80)
print("EXTRACTED TEXT FROM 4.PDF:")
print("="*80)
print(extracted_text[:1000])
print("="*80)

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
8. Policy Type (exact text as appears in document)
9. Accident Area (Urban or Rural)
10. Sex (Male or Female)
11. Deductible
12. Week of Month (1-5)

Extracted Text:
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
    temperature=0.1,
    response_format={"type": "json_object"}
)

claim_info = json.loads(response.choices[0].message.content)

print("\n" + "="*80)
print("AI EXTRACTED DATA:")
print("="*80)
for key, value in claim_info.items():
    print(f"  {key}: {value}")

print("\n" + "="*80)
print("COMPARISON WITH CSV ROW 4:")
print("="*80)
print("CSV Row 4 (Expected for 75.9% fraud):")
print("  DriverRating: 4")
print("  Age: 41")
print("  PoliceReportFiled: No")
print("  WeekOfMonthClaimed: 4")
print("  PolicyType: Utility - All Perils")
print("  WeekOfMonth: 5")
print("  AccidentArea: Urban")
print("  Sex: Male")
print("  Deductible: 400")

print("\n" + "="*80)
print("DIFFERENCES:")
print("="*80)

csv_data = {
    "driver_rating": 4,
    "age": 41,
    "police_report_filed": "No",
    "week_of_month_claimed": 4,
    "policy_type": "Utility - All Perils",
    "week_of_month": 5,
    "accident_area": "Urban",
    "sex": "Male",
    "deductible": 400
}

for key in ["driver_rating", "age", "policy_type", "week_of_month_claimed", "week_of_month", "accident_area", "sex", "deductible"]:
    pdf_value = claim_info.get(key, "NOT FOUND")
    csv_value = csv_data.get(key, "NOT FOUND")
    
    match = "✓" if str(pdf_value) == str(csv_value) else "✗"
    print(f"  {match} {key}: PDF={pdf_value}, CSV={csv_value}")

"""Test Azure Document Intelligence API directly"""
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

print(f"Endpoint: {endpoint}")
print(f"Key length: {len(key) if key else 0}")
print(f"Key (first 10 chars): {key[:10] if key else 'None'}")

# Create client
document_client = DocumentAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

print("\nClient created successfully")

# Test with a PDF file
pdf_path = "c:/Projects/DEMO/data/1.pdf"
print(f"\nReading file: {pdf_path}")

with open(pdf_path, "rb") as f:
    file_bytes = f.read()

print(f"File size: {len(file_bytes)} bytes")

try:
    print("\nStarting document analysis...")
    poller = document_client.begin_analyze_document("prebuilt-document", file_bytes)
    print("Poller created, waiting for result...")
    result = poller.result()
    print(f"\n✅ SUCCESS! Pages found: {len(result.pages)}")
    
    # Extract some text
    if result.pages:
        print("\nFirst few lines:")
        for i, line in enumerate(result.pages[0].lines[:5]):
            print(f"  {i+1}. {line.content}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

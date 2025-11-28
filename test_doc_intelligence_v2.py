"""Test Azure Document Intelligence with different models"""
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

document_client = DocumentAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

pdf_path = "c:/Projects/DEMO/data/1.pdf"

with open(pdf_path, "rb") as f:
    file_bytes = f.read()

# Try different models
models_to_try = [
    "prebuilt-layout",
    "prebuilt-read",
    "prebuilt-document"
]

for model_id in models_to_try:
    print(f"\n{'='*60}")
    print(f"Testing model: {model_id}")
    print(f"{'='*60}")
    
    try:
        poller = document_client.begin_analyze_document(model_id, file_bytes)
        result = poller.result()
        print(f"✅ SUCCESS with {model_id}!")
        print(f"   Pages: {len(result.pages)}")
        
        if result.pages and result.pages[0].lines:
            print(f"   First line: {result.pages[0].lines[0].content[:50]}...")
        
        break  # Stop on first success
        
    except Exception as e:
        print(f"❌ FAILED with {model_id}")
        print(f"   Error: {str(e)[:100]}")

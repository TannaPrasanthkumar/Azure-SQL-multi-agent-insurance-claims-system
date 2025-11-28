# 📄 Insurance Claims Document Intelligence Agent

An intelligent document processing application that uses Azure Document Intelligence and Azure OpenAI to analyze insurance claims and extract key information.

## 🚀 Features

- **Document Upload**: Support for PDF, PNG, JPG, JPEG files
- **Text Extraction**: Automatically extracts text from documents using OCR
- **Key-Value Pair Detection**: Identifies important fields and their values
- **Table Extraction**: Detects and extracts tabular data
- **AI-Powered Summary**: Generates comprehensive summaries using GPT-4
- **Interactive UI**: Clean, user-friendly Streamlit interface
- **Multi-Document Support**: Insurance claims, medical records, policies, invoices

## 📋 Prerequisites

- Python 3.8+
- Azure Document Intelligence resource
- Azure OpenAI resource
- Active Azure subscription

## 🔧 Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd C:\Projects\DEMO
   ```

2. **Activate virtual environment**
   ```powershell
   .\myenv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install streamlit azure-ai-formrecognizer openai python-dotenv
   ```

## ⚙️ Configuration

Update your `.env` file with Azure credentials:

```properties
# Azure OpenAI Configuration
AZURE_AISERVICES_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_AISERVICES_APIKEY=your-api-key
MODEL_DEPLOYMENT_NAME=gpt-4.1-mini

# Azure Document Intelligence Configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-document-intelligence.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-document-intelligence-key
```

## 🎯 Usage

1. **Start the application**
   ```bash
   python -m streamlit run document_agent.py
   ```

2. **Open your browser**
   - Local URL: http://localhost:8501
   - Network URL: http://172.16.4.182:8501

3. **Upload and analyze documents**
   - Click "Browse files" or drag & drop your document
   - Select document type (Insurance Claim, Medical Record, etc.)
   - Click "🚀 Analyze Document"
   - View extracted data and AI-generated summary

## 📊 What the Agent Extracts

### From Azure Document Intelligence:
- ✅ Full text content (OCR)
- ✅ Key-value pairs (fields and values)
- ✅ Tables and structured data
- ✅ Page count and layout information

### From Azure OpenAI:
- ✅ Comprehensive document summary
- ✅ Key information highlights (dates, amounts, parties)
- ✅ Important findings
- ✅ Potential issues or areas of concern

## 🎨 UI Features

- **Two-Column Layout**: Upload on left, results on right
- **Document Preview**: Visual preview for image files
- **Expandable Sections**: Detailed extracted data in collapsible panels
- **Download Summary**: Export AI-generated summaries as text files
- **Full Text View**: Access complete extracted text content
- **Progress Indicators**: Real-time processing status

## 📁 File Structure

```
C:\Projects\DEMO\
├── document_agent.py      # Main Streamlit application
├── main.py               # Interactive chat bot
├── .env                  # Environment variables (credentials)
├── myenv/               # Virtual environment
└── README.md            # This file
```

## 🔒 Security Notes

- Never commit `.env` files to version control
- Keep your API keys secure
- Use environment variables for sensitive data
- Regenerate keys if accidentally exposed

## 🛠️ Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
pip install streamlit azure-ai-formrecognizer openai python-dotenv
```

**"Authentication Error"**
- Verify your API keys in `.env`
- Check endpoint URLs are correct
- Ensure your Azure resources are active

**"Deployment Not Found"**
- Verify the model deployment name in Azure Portal
- Update `MODEL_DEPLOYMENT_NAME` in `.env`

**"Streamlit not found"**
```bash
python -m streamlit run document_agent.py
```

## 📝 Supported Document Types

- Insurance Claims
- Medical Records  
- Policy Documents
- Invoices & Receipts
- General Documents

## 🔄 Updates & Maintenance

To update packages:
```bash
pip install --upgrade streamlit azure-ai-formrecognizer openai
```

## 📞 Support

For issues or questions:
1. Check Azure Portal for resource status
2. Verify API keys and endpoints
3. Review error messages in the Streamlit UI
4. Check terminal output for detailed logs

## 🎓 Next Steps

- Add support for batch document processing
- Implement document comparison features
- Add export to PDF/Excel functionality
- Create custom document templates
- Add multi-language support

---

**Built with:**
- 🔷 Azure Document Intelligence
- 🤖 Azure OpenAI GPT-4
- 🎨 Streamlit
- 🐍 Python 3.x

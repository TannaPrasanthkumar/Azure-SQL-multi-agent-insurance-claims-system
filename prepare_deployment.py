"""
Prepare Model Deployment Package for Azure ML
Creates a deployment folder ready to upload to Azure ML portal
"""

import os
import shutil
from pathlib import Path

print("="*70)
print("      PREPARING AZURE ML DEPLOYMENT PACKAGE")
print("="*70)

# Define paths
model_dir = Path("ML/actual_model")
scoring_script = Path("ML/test.py")
deployment_dir = Path("azure_ml_deployment")

# Create deployment directory
if deployment_dir.exists():
    print(f"\n🗑️  Removing existing deployment directory...")
    shutil.rmtree(deployment_dir)

deployment_dir.mkdir(exist_ok=True)
print(f"\n✅ Created deployment directory: {deployment_dir}")

# Copy model files
print("\n📦 Copying model files...")
if model_dir.exists():
    for file in model_dir.glob("*.pkl"):
        dest = deployment_dir / file.name
        shutil.copy2(file, dest)
        print(f"   ✓ Copied: {file.name}")
else:
    print(f"   ❌ Model directory not found: {model_dir}")
    exit(1)

# Copy and rename scoring script
print("\n📝 Copying scoring script...")
if scoring_script.exists():
    dest = deployment_dir / "score.py"
    shutil.copy2(scoring_script, dest)
    print(f"   ✓ Copied: test.py → score.py")
else:
    print(f"   ❌ Scoring script not found: {scoring_script}")
    exit(1)

# Create conda environment file
print("\n📋 Creating conda environment file...")
conda_env = """name: fraud-detection-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - pip:
    - azureml-defaults
    - scikit-learn==1.0.2
    - pandas==1.3.5
    - numpy==1.21.5
    - joblib==1.1.0
    - imbalanced-learn==0.9.0
"""

with open(deployment_dir / "conda.yml", "w") as f:
    f.write(conda_env)
print("   ✓ Created: conda.yml")

# Create README with deployment instructions
readme = """# Azure ML Deployment Package

This folder contains everything needed to deploy the fraud detection model to Azure ML.

## Files Included:
- **score.py**: Scoring script (copied from test.py)
- **balanced_random_forest_fraud_detector_2025-11-21.pkl**: Trained model
- **label_encoders.pkl**: Label encoders for categorical features
- **scaler.pkl**: Standard scaler for feature normalization
- **model_metadata.pkl**: Model metadata and configuration
- **conda.yml**: Python environment dependencies

## Deployment Steps (Azure ML Portal):

### Option 1: Update Existing Endpoint

1. Go to Azure ML Studio (https://ml.azure.com)
2. Navigate to **Endpoints** → **Real-time endpoints**
3. Select your endpoint: `fraud-detection-endpoint`
4. Click **Update deployment**
5. Under **Model**, upload all .pkl files from this folder
6. Under **Scoring script**, upload `score.py`
7. Under **Environment**, select "Custom environment" and upload `conda.yml`
8. Click **Update**
9. Wait for deployment to complete (~5-10 minutes)

### Option 2: Create New Endpoint

1. Go to Azure ML Studio (https://ml.azure.com)
2. Navigate to **Endpoints** → **Create** → **Real-time endpoint**
3. Fill in:
   - Name: `fraud-detection-endpoint-v2`
   - Compute type: Managed
   - Virtual machine: Standard_DS2_v2
4. Upload all .pkl files from this folder
5. Set scoring script to `score.py`
6. Set environment using `conda.yml`
7. Click **Create**

### Testing After Deployment

Run this command to test the updated endpoint:

```bash
python test_sample2.py
```

Expected result after fix:
- Fraud Probability: ~0.29 (29%)
- Prediction: NOT FRAUD
- Risk Level: Low Risk

(Currently shows ~0.53 (53%) which is incorrect)

## What Changed?

The scoring script now includes:
- Proper preprocessing pipeline (scaling + encoding)
- Label encoders for categorical variables
- Consistent feature ordering
- Optimal threshold (0.5)

This ensures predictions match local testing results.
"""

with open(deployment_dir / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("   ✓ Created: README.md")

# List all files in deployment directory
print("\n" + "="*70)
print("DEPLOYMENT PACKAGE READY!")
print("="*70)
print(f"\nLocation: {deployment_dir.absolute()}")
print("\nFiles included:")
for file in sorted(deployment_dir.iterdir()):
    size = file.stat().st_size
    if size > 1024*1024:
        size_str = f"{size/(1024*1024):.2f} MB"
    elif size > 1024:
        size_str = f"{size/1024:.2f} KB"
    else:
        size_str = f"{size} bytes"
    print(f"  📄 {file.name:50s} {size_str:>12s}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("""
1. Open the 'azure_ml_deployment' folder
2. Read the README.md file for detailed instructions
3. Go to Azure ML Studio portal (https://ml.azure.com)
4. Follow the steps to update your endpoint
5. Test with: python test_sample2.py

After deployment, the model will give consistent results!
""")

print("✅ Deployment package created successfully!\n")

"""
Update Azure ML Endpoint with Correct Scoring Script
This script will redeploy the model with the test.py scoring script
to ensure consistency between local testing and production.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("      UPDATING AZURE ML ENDPOINT WITH CORRECT SCORING SCRIPT")
print("="*70)

# Check required files
model_dir = Path("ML/actual_model")
scoring_script = Path("ML/test.py")

if not model_dir.exists():
    print(f"\n❌ Model directory not found: {model_dir}")
    print("   Please ensure the model files are in ML/actual_model/")
    exit(1)

if not scoring_script.exists():
    print(f"\n❌ Scoring script not found: {scoring_script}")
    exit(1)

print(f"\n✅ Model directory: {model_dir}")
print(f"✅ Scoring script: {scoring_script}")

# List model files
model_files = list(model_dir.glob("*.pkl"))
print(f"\n📦 Model files found ({len(model_files)}):")
for f in model_files:
    print(f"   - {f.name}")

print("\n" + "="*70)
print("DEPLOYMENT INSTRUCTIONS")
print("="*70)

print("""
To update your Azure ML endpoint with the correct scoring script:

1. **Using Azure ML CLI:**
   
   # Navigate to ML directory
   cd ML
   
   # Update the deployment (replace <endpoint-name> and <deployment-name>)
   az ml online-deployment update \\
     --name <deployment-name> \\
     --endpoint <endpoint-name> \\
     --model actual_model \\
     --code-configuration code=. scoring_script=test.py \\
     --instance-type Standard_DS2_v2

2. **Using Azure ML SDK (Python):**

   Run the following Python code:
""")

print('''
   from azure.ai.ml import MLClient
   from azure.ai.ml.entities import Model, ManagedOnlineDeployment, CodeConfiguration, Environment
   from azure.identity import DefaultAzureCredential
   
   # Initialize ML Client
   credential = DefaultAzureCredential()
   ml_client = MLClient(
       credential=credential,
       subscription_id="<your-subscription-id>",
       resource_group_name="<your-resource-group>",
       workspace_name="<your-workspace-name>"
   )
   
   # Register the model with all artifacts
   model = Model(
       path="actual_model",
       name="fraud-detection-model",
       description="Fraud Detection Model with all artifacts"
   )
   registered_model = ml_client.models.create_or_update(model)
   
   # Create deployment with test.py as scoring script
   deployment = ManagedOnlineDeployment(
       name="fraud-detection-deployment",
       endpoint_name="fraud-detection-endpoint",
       model=registered_model.id,
       code_configuration=CodeConfiguration(
           code=".",
           scoring_script="test.py"
       ),
       environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
       instance_type="Standard_DS2_v2",
       instance_count=1
   )
   
   ml_client.online_deployments.begin_create_or_update(deployment).result()
   print("✅ Deployment updated successfully!")
''')

print("\n" + "="*70)
print("IMPORTANT NOTES")
print("="*70)
print("""
• The test.py file contains the CORRECT scoring logic with:
  - Proper preprocessing (scaling, encoding)
  - Optimal threshold (0.5)
  - All required artifacts (model, scaler, label_encoders, metadata)

• After redeployment, test with test_sample2.py to verify:
  - Sample 2 should give ~29% fraud probability (NOT FRAUD)
  - Predictions should match the local test.py results

• Current issue: Azure ML endpoint uses a different/older model
  Solution: Redeploy with the actual_model/ artifacts and test.py script
""")

print("\n✅ Ready to update Azure ML deployment!")
print("   Follow the instructions above to update your endpoint.\n")

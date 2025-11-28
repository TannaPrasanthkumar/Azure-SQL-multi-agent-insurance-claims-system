import re

# Read the file
with open('workflow_visualizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all Databricks UI references
replacements = [
    (r'\*\*💾 Databricks Agent\*\*', '**🗄️ Azure SQL Agent**'),
    (r'Databricks Agent" is querying', 'Azure SQL Agent" is querying'),
    (r'Routing to Databricks Agent', 'Routing to Azure SQL Agent'),
    (r'STEP 4: Databricks', 'STEP 4: Azure SQL'),
    (r'Document Agent completed - now showing Databricks', 'Document Agent completed - now showing Azure SQL'),
    (r'log_databricks_agent_action', 'log_policy_validation_action'),
    (r'"database": "Databricks"', '"database": "Azure SQL"'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open('workflow_visualizer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Updated all Databricks references to Azure SQL in workflow_visualizer.py')

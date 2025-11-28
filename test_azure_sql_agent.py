"""
Test script for Azure SQL Agent
"""
from azure_sql_agent import get_azure_sql_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the agent instance
sql_agent = get_azure_sql_agent()

print("=" * 60)
print("Testing Azure SQL Agent")
print("=" * 60)

# Test 1: Connect to database
print("\n✅ Test 1: Connection Test")
if sql_agent.connect():
    print("   ✓ Successfully connected to Azure SQL Database")
else:
    print("   ✗ Failed to connect")
    exit(1)

# Test 2: Validate existing policy
print("\n✅ Test 2: Validate Existing Policy (POL90927)")
result = sql_agent.validate_policy("POL90927")
print(f"   Result: {result}")
if result['success'] and result['policy_exists']:
    print("   ✓ Policy found successfully")
else:
    print("   ✗ Policy validation failed")

# Test 3: Validate non-existent policy
print("\n✅ Test 3: Validate Non-Existent Policy (POL99999)")
result = sql_agent.validate_policy("POL99999")
print(f"   Result: {result}")
if result['success'] and not result['policy_exists']:
    print("   ✓ Correctly identified non-existent policy")
else:
    print("   ✗ Test failed")

# Test 4: Get policy details for existing policy
print("\n✅ Test 4: Get Policy Details (POL52837)")
result = sql_agent.get_policy_details("POL52837")
print(f"   Success: {result['success']}")
if result['success']:
    policy_info = result['policy_info']
    print(f"   Policy Info: {policy_info}")
    print("   ✓ Policy details retrieved successfully")
else:
    print(f"   ✗ Failed: {result.get('message', 'Unknown error')}")

# Test 5: Get policy details for non-existent policy
print("\n✅ Test 5: Get Details for Non-Existent Policy (POL99999)")
result = sql_agent.get_policy_details("POL99999")
print(f"   Result: {result}")
if not result['success']:
    print("   ✓ Correctly handled non-existent policy")
else:
    print("   ✗ Test failed")

print("\n" + "=" * 60)
print("All Tests Completed!")
print("=" * 60)

"""
Quick test to verify Azure SQL integration in PolicyValidator
"""
from policy_validator import PolicyValidator

print("="*60)
print("Testing Azure SQL Integration in PolicyValidator")
print("="*60)

# Initialize validator
validator = PolicyValidator()
print(f"\n✅ PolicyValidator initialized")
print(f"   - Enabled: {validator.enabled}")
print(f"   - SQL Agent: {validator.sql_agent}")

if validator.enabled:
    print("\n🧪 Test 1: Validate existing policy POL90927")
    result = validator.validate_policy("POL90927")
    print(f"   Result: {result.get('found')}")
    print(f"   Message: {result.get('message')}")
    print(f"   Status: {result.get('policy_status')}")
    
    print("\n🧪 Test 2: Validate non-existent policy POL99999")
    result = validator.validate_policy("POL99999")
    print(f"   Result: {result.get('found')}")
    print(f"   Message: {result.get('message')}")
    
    print("\n✅ Integration test completed successfully!")
else:
    print("\n❌ PolicyValidator not enabled - check Azure SQL configuration")

print("="*60)

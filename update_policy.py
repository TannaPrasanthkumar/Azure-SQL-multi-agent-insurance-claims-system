from azure_sql_agent import get_azure_sql_agent

agent = get_azure_sql_agent()
if agent.connect():
    cursor = agent.connection.cursor()
    
    # Show before update
    cursor.execute("SELECT policy_number, policy_limit FROM policy_data WHERE policy_number = 'POL90927'")
    before = cursor.fetchone()
    print(f"\nBEFORE: Policy {before[0]} - Limit: {before[1]}")
    
    # Update the policy limit
    cursor.execute("""
        UPDATE policy_data 
        SET policy_limit = 527000 
        WHERE policy_number = 'POL90927'
    """)
    agent.connection.commit()
    
    # Show after update
    cursor.execute("SELECT policy_number, policy_limit FROM policy_data WHERE policy_number = 'POL90927'")
    after = cursor.fetchone()
    print(f"AFTER:  Policy {after[0]} - Limit: {after[1]}")
    print(f"\n✅ Successfully updated policy limit from {before[1]} to {after[1]}")
    
    cursor.close()
else:
    print("Failed to connect to database")

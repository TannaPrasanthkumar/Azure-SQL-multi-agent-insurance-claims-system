from azure_sql_agent import get_azure_sql_agent

agent = get_azure_sql_agent()
if agent.connect():
    cursor = agent.connection.cursor()
    cursor.execute('SELECT TOP 1 * FROM policy_data')
    row = cursor.fetchone()
    
    print("\n" + "="*60)
    print("FIRST ROW DATA:")
    print("="*60)
    print(f"Policy Number: {row[0]}")
    print(f"Policyholder Name: {row[1]}")
    print(f"Policyholder ID: {row[2]}")
    print(f"Claim History Count: {row[3]}")
    print(f"Past Claims Amount: {row[4]}")
    print(f"Policy Status: {row[5]}")
    print(f"Policy Limit: {row[6]}")
    print("="*60)
    
    cursor.close()

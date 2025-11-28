import pyodbc
from dotenv import load_dotenv
import os

load_dotenv()

server = os.getenv('AZURE_SQL_SERVER')
database = os.getenv('AZURE_SQL_DATABASE')
username = os.getenv('AZURE_SQL_USERNAME')
password = os.getenv('AZURE_SQL_PASSWORD')

connection_string = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=yes;"
)

try:
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    # Show before
    cursor.execute("SELECT policy_number, policy_limit FROM policy_data WHERE policy_number = 'POL90927'")
    before = cursor.fetchone()
    print(f"\nBEFORE: Policy {before[0]} - Limit: ${before[1]:,}")
    
    # Update
    cursor.execute("UPDATE policy_data SET policy_limit = 527000 WHERE policy_number = 'POL90927'")
    conn.commit()
    
    # Show after
    cursor.execute("SELECT policy_number, policy_limit FROM policy_data WHERE policy_number = 'POL90927'")
    after = cursor.fetchone()
    print(f"AFTER:  Policy {after[0]} - Limit: ${after[1]:,}")
    print(f"\n✅ Successfully updated! Changed from ${before[1]:,} to ${after[1]:,}")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")

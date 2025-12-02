"""
Script to update SQL Server with data from Policy_data.csv
"""

import os
import pandas as pd
import pyodbc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection details
server = os.getenv("AZURE_SQL_SERVER")
database = os.getenv("AZURE_SQL_DATABASE")
username = os.getenv("AZURE_SQL_USERNAME")
password = os.getenv("AZURE_SQL_PASSWORD")

print(f"🔌 Connecting to SQL Server...")
print(f"   Server: {server}")
print(f"   Database: {database}")

# Create connection string
connection_string = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
)

try:
    # Connect to database
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print("✅ Connected to SQL Server successfully!")
    
    # Check existing tables
    print("\n🔍 Checking existing tables...")
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    tables = cursor.fetchall()
    print(f"   Available tables:")
    for table in tables:
        print(f"   - {table[0]}")
    
    # Read CSV file
    print("\n📄 Reading Policy_data.csv...")
    df = pd.read_csv('Policy_data.csv')
    print(f"   Found {len(df)} records")
    
    # Clear existing data (optional - comment out if you want to keep existing data)
    print("\n🗑️ Clearing existing policy data...")
    cursor.execute("DELETE FROM policy_data")
    conn.commit()
    print("   Existing data cleared")
    
    # Insert new data
    print("\n📝 Inserting new policy data...")
    insert_query = """
    INSERT INTO policy_data (
        policy_number, 
        policyholder_Name, 
        policyholder_id, 
        claim_history_count, 
        past_claims_amount, 
        policy_status, 
        policy_limit
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    inserted_count = 0
    for index, row in df.iterrows():
        cursor.execute(
            insert_query,
            row['policy_number'],
            row['policyholder_Name'],
            row['policyholder_id'],
            row['claim_history_count'],
            row['past_claims_amount'],
            row['policy_status'],
            row['policy_limit']
        )
        inserted_count += 1
        if inserted_count % 5 == 0:
            print(f"   Inserted {inserted_count}/{len(df)} records...")
    
    conn.commit()
    print(f"✅ Successfully inserted {inserted_count} records!")
    
    # Verify data
    print("\n🔍 Verifying inserted data...")
    cursor.execute("SELECT COUNT(*) FROM policy_data")
    count = cursor.fetchone()[0]
    print(f"   Total records in database: {count}")
    
    # Show sample data
    print("\n📊 Sample data from database:")
    cursor.execute("SELECT TOP 5 policy_number, policyholder_Name, policy_status, policy_limit FROM policy_data")
    for row in cursor.fetchall():
        print(f"   {row[0]} | {row[1]} | {row[2]} | ${row[3]:,.0f}")
    
    cursor.close()
    conn.close()
    print("\n🎉 Database update completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error updating database: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

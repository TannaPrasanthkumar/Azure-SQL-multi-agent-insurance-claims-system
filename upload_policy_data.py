import pyodbc
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection details
server = os.getenv('AZURE_SQL_SERVER')
database = os.getenv('AZURE_SQL_DATABASE')
username = os.getenv('AZURE_SQL_USERNAME')
password = os.getenv('AZURE_SQL_PASSWORD')

# Read CSV file
csv_file = r'C:\Projects\DEMO\Policy_data.csv'
df = pd.read_csv(csv_file)

print(f"Found {len(df)} records in CSV file")
print(f"Columns: {list(df.columns)}")

# Create connection string using SQL Server driver
connection_string = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=yes;"
)

print(f"Using driver: SQL Server")

try:
    # Connect to database
    print("\nConnecting to Azure SQL Database...")
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print("Connected successfully!")
    
    # Insert data
    print(f"\nInserting {len(df)} records...")
    inserted_count = 0
    skipped_count = 0
    
    for index, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO policy_data 
                (policy_number, policyholder_Name, policyholder_id, claim_history_count, 
                 past_claims_amount, policy_status, policy_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, 
            row['policy_number'], 
            row['policyholder_Name'], 
            row['policyholder_id'], 
            row['claim_history_count'], 
            row['past_claims_amount'], 
            row['policy_status'], 
            row['policy_limit'])
            inserted_count += 1
        except pyodbc.IntegrityError:
            # Skip duplicate policy numbers
            skipped_count += 1
            print(f"Skipped duplicate: {row['policy_number']}")
        except Exception as e:
            print(f"Error inserting {row['policy_number']}: {e}")
    
    # Commit changes
    conn.commit()
    print(f"\n✅ Upload complete!")
    print(f"   - Inserted: {inserted_count} records")
    print(f"   - Skipped: {skipped_count} duplicates")
    
    # Verify data
    cursor.execute("SELECT COUNT(*) FROM policy_data")
    total = cursor.fetchone()[0]
    print(f"   - Total records in database: {total}")
    
    # Close connection
    cursor.close()
    conn.close()
    print("\nConnection closed.")

except Exception as e:
    print(f"❌ Error: {e}")

import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure SQL Database credentials
server = os.getenv('AZURE_SQL_SERVER')
database = os.getenv('AZURE_SQL_DATABASE')
username = os.getenv('AZURE_SQL_USERNAME')
password = os.getenv('AZURE_SQL_PASSWORD')

# Connection string
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Connect to Azure SQL Database
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print("✅ Connected to Azure SQL Database")
    
    # Create table
    create_table_query = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='policy_data' AND xtype='U')
    CREATE TABLE policy_data (
        policy_number VARCHAR(20) PRIMARY KEY,
        policyholder_Name NVARCHAR(100),
        policyholder_id VARCHAR(50),
        claim_history_count INT,
        past_claims_amount DECIMAL(18, 2),
        policy_status VARCHAR(20),
        policy_limit DECIMAL(18, 2)
    )
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    print("✅ Table 'policy_data' created successfully!")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")

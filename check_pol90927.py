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

conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

cursor.execute("SELECT * FROM policy_data WHERE policy_number = 'POL90927'")
row = cursor.fetchone()

print("\n" + "="*60)
print("Policy POL90927 Details:")
print("="*60)
print(f'Policy Number: {row[0]}')
print(f'Policyholder Name: {row[1]}')
print(f'Policyholder ID: {row[2]}')
print(f'Claim History Count: {row[3]}')
print(f'Past Claims Amount: ${row[4]:,}')
print(f'Policy Status: {row[5]}')
print(f'Policy Limit: ${row[6]:,}')
print("="*60)

cursor.close()
conn.close()

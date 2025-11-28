"""
Azure Databricks Connection using REST API
This method uses Databricks REST API to execute queries
No SQL Warehouse required - works with regular compute clusters
"""

import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Databricks Configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_SERVER_HOSTNAME")  # e.g., https://adb-xxxxx.azuredatabricks.net
DATABRICKS_TOKEN = os.getenv("DATABRICKS_ACCESS_TOKEN")
DATABRICKS_CLUSTER_ID = os.getenv("DATABRICKS_CLUSTER_ID")  # Your cluster ID from the JSON you shared


class DatabricksAgent:
    def __init__(self, host, token, cluster_id):
        self.host = host.rstrip('/')
        self.token = token
        self.cluster_id = cluster_id
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    def get_cluster_status(self):
        """Check if cluster is running"""
        url = f"{self.host}/api/2.0/clusters/get"
        params = {"cluster_id": self.cluster_id}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        cluster_info = response.json()
        state = cluster_info.get('state', 'UNKNOWN')
        print(f"🖥️  Cluster Status: {state}")
        return state

    def start_cluster(self):
        """Start the cluster if it's not running"""
        url = f"{self.host}/api/2.0/clusters/start"
        data = {"cluster_id": self.cluster_id}
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        print("⏳ Starting cluster... This may take a few minutes...")
        # Wait for cluster to start
        while True:
            state = self.get_cluster_status()
            if state == 'RUNNING':
                print("✅ Cluster is now running!")
                break
            time.sleep(10)

    def execute_sql(self, query):
        """Execute a SQL query on Databricks cluster via REST API"""
        # Step 1: Create execution context
        print(f"🔄 Creating execution context...")
        context_url = f"{self.host}/api/1.2/contexts/create"
        context_data = {
            "clusterId": self.cluster_id,
            "language": "sql"
        }
        response = requests.post(context_url, headers=self.headers, json=context_data)
        response.raise_for_status()
        context_id = response.json()['id']
        print(f"✅ Context created: {context_id}")

        # Step 2: Execute command
        print(f"📊 Executing SQL query...")
        execute_url = f"{self.host}/api/1.2/commands/execute"
        execute_data = {
            "clusterId": self.cluster_id,
            "contextId": context_id,
            "language": "sql",
            "command": query
        }
        response = requests.post(execute_url, headers=self.headers, json=execute_data)
        response.raise_for_status()
        command_id = response.json()['id']

        # Step 3: Poll for results
        print(f"⏳ Waiting for results...")
        status_url = f"{self.host}/api/1.2/commands/status"
        while True:
            status_params = {
                "clusterId": self.cluster_id,
                "contextId": context_id,
                "commandId": command_id
            }
            response = requests.get(status_url, headers=self.headers, params=status_params)
            response.raise_for_status()
            result = response.json()
            status = result['status']
            if status == 'Finished':
                print("✅ Query executed successfully!")
                return result['results']
            elif status == 'Error':
                error_msg = result.get('results', {}).get('cause', 'Unknown error')
                raise Exception(f"Query failed: {error_msg}")
            elif status in ['Cancelled', 'Cancelling']:
                raise Exception("Query was cancelled")
            time.sleep(2)

    def get_policy_dataset(self, limit=100):
        """Get policy_data as pandas DataFrame"""
        query = f"SELECT * FROM policy_data LIMIT {limit}"
        result = self.execute_sql(query)
        if 'data' in result:
            df = pd.DataFrame(result['data'])
            print(f"\n✅ Retrieved {len(df)} rows")
            return df
        else:
            print("Result:", result)
            return None




def agent_cli():
    """Simple CLI for Databricks Agent"""
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN or not DATABRICKS_CLUSTER_ID:
        print("❌ Missing configuration. Please set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID in your .env file.")
        return
    agent = DatabricksAgent(DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID)
    print("\n🤖 Databricks Agent Ready!")
    while True:
        print("\nOptions:")
        print("1. Get cluster status")
        print("2. Start cluster")
        print("3. Query policy_dataset")
        print("4. Run custom SQL query")
        print("5. Exit")
        choice = input("Select an option: ").strip()
        if choice == "1":
            agent.get_cluster_status()
        elif choice == "2":
            agent.start_cluster()
        elif choice == "3":
            df = agent.get_policy_dataset()
            if df is not None:
                print(df.head())
        elif choice == "4":
            query = input("Enter SQL query: ")
            try:
                result = agent.execute_sql(query)
                print(result)
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    agent_cli()

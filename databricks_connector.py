from databricks.connect import DatabricksSession
import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DATABRICKS_SERVER_HOSTNAME")            # e.g. adb-3668550085637732.12.azuredatabricks.net
cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")           # Your cluster ID
access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")       # Your personal access token

# Databricks Connect URI format (prepend "https://" to host)
data = DatabricksSession.builder.remote(
    host = f"https://{host}",
    token = access_token,
    cluster_id = cluster_id
).getOrCreate()

print("hai")
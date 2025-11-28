"""
Azure SQL Database Agent
Handles policy validation and data retrieval from Azure SQL Database
"""

import pyodbc
import os
from typing import Dict, Any, Optional
from datetime import datetime


class AzureSQLAgent:
    """Agent for interacting with Azure SQL Database for policy validation"""
    
    def __init__(self):
        """Initialize Azure SQL Database connection"""
        self.server = os.getenv('AZURE_SQL_SERVER')  # e.g., 'your-server.database.windows.net'
        self.database = os.getenv('AZURE_SQL_DATABASE')  # e.g., 'insurance-db'
        self.username = os.getenv('AZURE_SQL_USERNAME')
        self.password = os.getenv('AZURE_SQL_PASSWORD')
        self.connection = None
        
    def connect(self) -> bool:
        """Establish connection to Azure SQL Database"""
        try:
            connection_string = (
                f"DRIVER={{SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"Encrypt=yes;"
                f"TrustServerCertificate=yes;"
            )
            
            self.connection = pyodbc.connect(connection_string)
            print(f"✅ Connected to Azure SQL Database: {self.database}")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to Azure SQL Database: {e}")
            return False
    
    def validate_policy(self, policy_number: str) -> Dict[str, Any]:
        """
        Validate if policy exists in Azure SQL Database
        
        Args:
            policy_number: Policy number to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not self.connection:
                if not self.connect():
                    return {
                        'success': False,
                        'error': 'Failed to connect to Azure SQL Database',
                        'policy_exists': False
                    }
            
            cursor = self.connection.cursor()
            
            # Check if policy exists
            query = "SELECT policy_number FROM policy_data WHERE policy_number = ?"
            cursor.execute(query, (policy_number,))
            result = cursor.fetchone()
            
            if result:
                print(f"✅ Policy {policy_number} found in Azure SQL Database")
                return {
                    'success': True,
                    'policy_exists': True,
                    'policy_number': policy_number
                }
            else:
                print(f"❌ Policy {policy_number} not found in Azure SQL Database")
                return {
                    'success': True,
                    'policy_exists': False,
                    'policy_number': policy_number,
                    'error': f'Policy {policy_number} not found in database'
                }
                
        except Exception as e:
            print(f"❌ Error validating policy: {e}")
            return {
                'success': False,
                'error': str(e),
                'policy_exists': False
            }
    
    def get_policy_details(self, policy_number: str) -> Dict[str, Any]:
        """
        Retrieve complete policy details from Azure SQL Database
        
        Args:
            policy_number: Policy number to retrieve
            
        Returns:
            Dictionary with policy information
        """
        try:
            if not self.connection:
                if not self.connect():
                    return {
                        'success': False,
                        'error': 'Failed to connect to Azure SQL Database'
                    }
            
            cursor = self.connection.cursor()
            
            # Get all policy details
            query = "SELECT * FROM policy_data WHERE policy_number = ?"
            cursor.execute(query, (policy_number,))
            
            # Get column names
            columns = [column[0] for column in cursor.description]
            
            # Fetch data
            row = cursor.fetchone()
            
            if row:
                # Convert row to dictionary
                policy_data = dict(zip(columns, row))
                
                print(f"✅ Retrieved policy details for {policy_number}")
                
                return {
                    'success': True,
                    'policy_info': {
                        'policy_number': policy_data.get('policy_number'),
                        'policyholder_name': policy_data.get('policyholder_Name'),
                        'policyholder_id': policy_data.get('policyholder_id'),
                        'claim_history_count': policy_data.get('claim_history_count'),
                        'past_claims_amount': policy_data.get('past_claims_amount'),
                        'policy_status': policy_data.get('policy_status'),
                        'policy_limit': policy_data.get('policy_limit')
                    },
                    'validation': {
                        'is_valid': policy_data.get('policy_status', '').lower() == 'active',
                        'details': {
                            'policy_status': policy_data.get('policy_status'),
                            'policy_limit': policy_data.get('policy_limit'),
                            'past_claims_amount': policy_data.get('past_claims_amount'),
                            'claim_history_count': policy_data.get('claim_history_count')
                        }
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f'Policy {policy_number} not found'
                }
                
        except Exception as e:
            print(f"❌ Error retrieving policy details: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Azure SQL Database connection closed")


# Singleton instance
_azure_sql_agent = None

def get_azure_sql_agent() -> AzureSQLAgent:
    """Get or create Azure SQL Agent instance"""
    global _azure_sql_agent
    if _azure_sql_agent is None:
        _azure_sql_agent = AzureSQLAgent()
    return _azure_sql_agent

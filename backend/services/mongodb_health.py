"""
MongoDB Health Check and Connection Management
"""

import os
from typing import Optional
from dotenv import load_dotenv
import time

load_dotenv()

class MongoDBHealthCheck:
    """Check MongoDB connection health and manage fallback strategies"""
    
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "victor_rag")
        self.is_healthy = False
        self.last_check = 0
        self.check_interval = 30  # Check every 30 seconds
    
    def check_connection(self) -> bool:
        """Check if MongoDB is accessible"""
        current_time = time.time()
        
        # Only check if enough time has passed
        if current_time - self.last_check < self.check_interval:
            return self.is_healthy
        
        try:
            from pymongo import MongoClient
            
            print(f"🔍 Checking MongoDB connection: {self.mongodb_uri}")
            
            # Create client with short timeout
            client = MongoClient(
                self.mongodb_uri, 
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000
            )
            
            # Test connection
            client.admin.command('ping')
            
            # Test database access
            db = client[self.database_name]
            collections = db.list_collection_names()
            
            client.close()
            
            self.is_healthy = True
            self.last_check = current_time
            print(f"✅ MongoDB connection healthy - Collections: {len(collections)}")
            
            return True
            
        except Exception as e:
            self.is_healthy = False
            self.last_check = current_time
            print(f"❌ MongoDB connection failed: {str(e)}")
            print("💡 Suggestion: Check if MongoDB is running or update MONGODB_URI in .env")
            
            return False
    
    def get_fallback_strategy(self) -> str:
        """Get fallback strategy when MongoDB is unavailable"""
        if not self.is_healthy:
            return "in_memory"
        return "mongodb"

# Global health checker
_mongo_health = None

def get_mongodb_health() -> MongoDBHealthCheck:
    """Get MongoDB health checker singleton"""
    global _mongo_health
    if _mongo_health is None:
        _mongo_health = MongoDBHealthCheck()
    return _mongo_health

def is_mongodb_available() -> bool:
    """Quick check if MongoDB is available"""
    return get_mongodb_health().check_connection()
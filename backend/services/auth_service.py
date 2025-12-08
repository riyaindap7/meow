import os
import jwt
from typing import Optional, Dict
from fastapi import HTTPException, status
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

class AuthService:
    def __init__(self):
        self.jwt_secret = os.getenv("BETTER_AUTH_SECRET", "your-secret-key")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        
        # MongoDB connection for fetching user role
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo_client.victor_rag
        self.users_collection = self.db.users
    
    def get_user_with_role(self, user_id: str) -> Dict:
        """Fetch complete user data including role from MongoDB"""
        try:
            # Convert string ID to ObjectId for MongoDB query
            user_doc = self.users_collection.find_one({"_id": ObjectId(user_id)})
            
            if not user_doc:
                print(f"⚠️ User not found in database: {user_id}")
                return None
                
            # Extract role - check both direct field and additionalFields
            role = "user"  # default
            
            # First check if role is stored directly (Better Auth v2 style)
            if "role" in user_doc:
                role = user_doc.get("role", "user")
                print(f"🔍 Found role in direct field: {role}")
            # Then check additionalFields (Better Auth v1 style)
            elif "additionalFields" in user_doc and user_doc["additionalFields"]:
                role = user_doc["additionalFields"].get("role", "user")
                print(f"🔍 Found role in additionalFields: {role}")
            
            user_data = {
                "user_id": str(user_doc["_id"]),
                "email": user_doc.get("email"),
                "name": user_doc.get("name"),
                "role": role
            }
            
            print(f"🔍 ROLE FETCHED: {user_data['email']} -> ROLE: {role}")
            return user_data
            
        except Exception as e:
            print(f"❌ Error fetching user role: {str(e)}")
            return None
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token and return user data with role"""
        try:
            # Try to verify JWT locally first
            try:
                decoded = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                user_id = decoded.get('userId') or decoded.get('id')
                
                # Fetch complete user data including role from database
                user_data = self.get_user_with_role(str(user_id))
                if user_data:
                    return user_data
                    
                # Fallback to basic data if database fetch fails
                return {
                    "user_id": str(user_id),
                    "email": decoded.get('email'),
                    "name": decoded.get('name'),
                    "role": "user"  # default role
                }
                
            except jwt.InvalidTokenError:
                # If JWT fails, check if token looks like a MongoDB ObjectId (for development)
                if len(token) == 24:  # MongoDB ObjectId length
                    print(f"🔧 Development mode: Using token as user_id: {token}")
                    
                    # Try to get user data from database in dev mode
                    user_data = self.get_user_with_role(token)
                    if user_data:
                        return user_data
                        
                    return {
                        "user_id": token,
                        "email": "dev@example.com",
                        "name": "Development User",
                        "role": "user"
                    }
                
                # If local verification fails, try Better Auth API
                response = requests.get(
                    f"{self.frontend_url}/api/auth/get-session",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=5
                )
                
                if response.status_code == 200:
                    session_data = response.json()
                    if session_data.get('user'):
                        user = session_data['user']
                        user_id = user.get('id') or user.get('_id')
                        
                        # Fetch complete user data including role from database
                        user_data = self.get_user_with_role(str(user_id))
                        if user_data:
                            return user_data
                            
                        # Fallback to session data
                        return {
                            "user_id": str(user_id),
                            "email": user.get('email'),
                            "name": user.get('name'),
                            "role": "user"  # default role
                        }
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
        
        except Exception as e:
            print(f"❌ Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )

# Global instance
_auth_service = None

def get_auth_service() -> AuthService:
    """Get or create AuthService singleton"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service

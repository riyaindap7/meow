import os
import jwt
from typing import Optional, Dict
from fastapi import HTTPException, status
import requests
from dotenv import load_dotenv

load_dotenv()

class AuthService:
    def __init__(self):
        self.jwt_secret = os.getenv("BETTER_AUTH_SECRET", "your-secret-key")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token and return user data"""
        try:
            # Try to verify JWT locally first
            try:
                decoded = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
                user_id = decoded.get('userId') or decoded.get('id')
                return {
                    "user_id": str(user_id),  # This will be the MongoDB ObjectId
                    "email": decoded.get('email'),
                    "name": decoded.get('name')
                }
            except jwt.InvalidTokenError:
                # If JWT fails, check if token looks like a MongoDB ObjectId (for development)
                if len(token) == 24:  # MongoDB ObjectId length
                    print(f"🔧 Development mode: Using token as user_id: {token}")
                    return {
                        "user_id": token,
                        "email": "dev@example.com",
                        "name": "Development User"
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
                        return {
                            "user_id": str(user_id),  # MongoDB ObjectId as string
                            "email": user.get('email'),
                            "name": user.get('name')
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

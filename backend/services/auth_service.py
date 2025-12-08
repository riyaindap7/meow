import os
from typing import Optional, Dict
from fastapi import HTTPException, status
from dotenv import load_dotenv
from backend.services.user_service import verify_session, get_user_by_id

load_dotenv()

class AuthService:
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key")
    
    def verify_token(self, token: str) -> Dict:
        """
        Verify session token and return user data with role
        Now uses the custom auth system instead of Better Auth
        """
        try:
            print(f"🔍 Verifying token: {token[:20]}...")
            
            # Use the new verify_session function from user_service
            user_data = verify_session(token)
            
            if not user_data:
                print("❌ Invalid or expired session token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Return user data in the expected format
            result = {
                "user_id": str(user_data["_id"]),
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "role": user_data.get("role", "user")
            }
            
            print(f"✅ Token verified for user: {result['email']} (role: {result['role']})")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    def get_user_with_role(self, user_id: str) -> Dict:
        """Fetch complete user data including role from MongoDB"""
        try:
            print(f"🔍 Fetching user data for ID: {user_id}")
            
            # Use the new get_user_by_id function from user_service
            user_data = get_user_by_id(user_id)
            
            if not user_data:
                print(f"⚠️ User not found: {user_id}")
                return None
            
            result = {
                "user_id": str(user_data["_id"]),
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "role": user_data.get("role", "user")
            }
            
            print(f"✅ User data fetched: {result['email']} (role: {result['role']})")
            return result
            
        except Exception as e:
            print(f"❌ Error fetching user data: {str(e)}")
            return None


# Global instance
_auth_service = None

def get_auth_service() -> AuthService:
    """Get or create AuthService singleton"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service

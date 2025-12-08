from fastapi import HTTPException, status, Request, Depends
from services.auth_service import get_auth_service, AuthService
from typing import Dict

async def verify_auth_token(request: Request, auth_service: AuthService = Depends(get_auth_service)) -> Dict:
    """FastAPI dependency to verify authentication token"""
    
    # Get token from Authorization header
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    # Extract Bearer token
    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization scheme"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    # Verify token and get user data
    user_data = auth_service.verify_token(token)
    print(f"🔐 Authenticated user: {user_data['user_id']} ({user_data['email']})")
    return user_data

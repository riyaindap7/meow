from fastapi import APIRouter, HTTPException, Response, Cookie
from pydantic import BaseModel, EmailStr
from typing import Optional
from backend.services.user_service import (
    create_user,
    authenticate_user,
    create_session,
    verify_session,
    delete_session
)

router = APIRouter()


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class SigninRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[dict] = None
    token: Optional[str] = None


@router.post("/signup", response_model=AuthResponse)
async def signup(request: SignupRequest, response: Response):
    """User signup endpoint"""
    try:
        if len(request.password) < 6:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 6 characters long"
            )
        
        user = create_user(
            email=request.email,
            password=request.password,
            name=request.name
        )
        
        token = create_session(user["_id"])
        
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=7 * 24 * 60 * 60
        )
        
        return AuthResponse(
            success=True,
            message="User created successfully",
            user=user,
            token=token
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/signin", response_model=AuthResponse)
async def signin(request: SigninRequest, response: Response):
    """User signin endpoint"""
    try:
        user = authenticate_user(request.email, request.password)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        token = create_session(user["_id"])
        
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=7 * 24 * 60 * 60
        )
        
        return AuthResponse(
            success=True,
            message="Signed in successfully",
            user=user,
            token=token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Signin error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/signout")
async def signout(
    response: Response,
    session_token: Optional[str] = Cookie(None)
):
    """User signout endpoint"""
    try:
        if session_token:
            delete_session(session_token)
        
        response.delete_cookie(key="session_token")
        
        return {"success": True, "message": "Signed out successfully"}
    
    except Exception as e:
        print(f"❌ Signout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session")
async def get_session(session_token: Optional[str] = Cookie(None)):
    """Get current session - THIS IS THE ENDPOINT YOU'RE CALLING"""
    try:
        print(f"🔍 Session check - Token: {session_token[:20] if session_token else 'None'}...")
        
        if not session_token:
            return {"user": None, "authenticated": False}
        
        user = verify_session(session_token)
        
        if not user:
            return {"user": None, "authenticated": False}
        
        print(f"✅ Session valid for user: {user.get('email')}")
        return {
            "user": user,
            "authenticated": True
        }
    
    except Exception as e:
        print(f"❌ Session error: {str(e)}")
        return {"user": None, "authenticated": False}
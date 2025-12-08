from typing import Optional, Dict, Any
from datetime import datetime
import hashlib
import secrets
from backend.services.mongodb_service import get_mongo_db, serialize_doc

def get_ist_now():
    """Get current time in IST"""
    return datetime.utcnow()


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash password with salt"""
    if not salt:
        salt = secrets.token_hex(16)
    
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return pwd_hash.hex(), salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify password against hash"""
    pwd_hash, _ = hash_password(password, salt)
    return pwd_hash == hashed_password


def create_user(email: str, password: str, name: str, role: str = "user") -> Dict[str, Any]:
    """Create a new user"""
    db = get_mongo_db()
    users = db.users
    
    if users.find_one({"email": email}):
        raise ValueError("User with this email already exists")
    
    pwd_hash, salt = hash_password(password)
    
    now = get_ist_now()
    user_data = {
        "email": email,
        "name": name,
        "password": pwd_hash,
        "salt": salt,
        "role": role,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
        "last_login": None
    }
    
    result = users.insert_one(user_data)
    user_data["_id"] = result.inserted_id
    
    user_data.pop("password")
    user_data.pop("salt")
    
    print(f"✅ User created: {email}")
    return serialize_doc(user_data)


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user and return user data"""
    db = get_mongo_db()
    users = db.users
    
    user = users.find_one({"email": email})
    
    if not user:
        return None
    
    if not verify_password(password, user["password"], user["salt"]):
        return None
    
    users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": get_ist_now()}}
    )
    
    user.pop("password")
    user.pop("salt")
    
    print(f"✅ User authenticated: {email}")
    return serialize_doc(user)


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    from bson import ObjectId
    db = get_mongo_db()
    users = db.users
    
    user = users.find_one({"_id": ObjectId(user_id)})
    
    if user:
        user.pop("password", None)
        user.pop("salt", None)
    
    return serialize_doc(user) if user else None


def create_session(user_id: str) -> str:
    """Create session token for user"""
    db = get_mongo_db()
    sessions = db.sessions
    
    token = secrets.token_urlsafe(32)
    now = get_ist_now()
    
    session_data = {
        "user_id": user_id,
        "token": token,
        "created_at": now,
        "expires_at": datetime.fromtimestamp(now.timestamp() + 7 * 24 * 60 * 60)
    }
    
    sessions.insert_one(session_data)
    print(f"✅ Session created for user: {user_id}")
    return token


def verify_session(token: str) -> Optional[Dict[str, Any]]:
    """Verify session token and return user data"""
    db = get_mongo_db()
    sessions = db.sessions
    
    session = sessions.find_one({"token": token})
    
    if not session:
        print("❌ Session not found")
        return None
    
    if session["expires_at"] < get_ist_now():
        sessions.delete_one({"_id": session["_id"]})
        print("❌ Session expired")
        return None
    
    return get_user_by_id(str(session["user_id"]))


def delete_session(token: str) -> bool:
    """Delete session (logout)"""
    db = get_mongo_db()
    sessions = db.sessions
    
    result = sessions.delete_one({"token": token})
    return result.deleted_count > 0
import requests
import json

# Test the backend authentication
BASE_URL = "http://localhost:8000"

# Use Durva's MongoDB ObjectId as token (from migration)
AUTH_TOKEN = "693426160444cd216cc768fa"

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

print("🔍 Testing Backend Authentication")
print(f"Using auth token: {AUTH_TOKEN}")
print("=" * 50)

# Test 1: Health check (no auth required)
print("1. Testing health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: List conversations (auth required)
print("\n2. Testing conversations list...")
try:
    response = requests.get(f"{BASE_URL}/conversations", headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Found {data['count']} conversations")
        for conv in data['conversations'][:3]:
            print(f"      - {conv['title']} ({conv['message_count']} messages)")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Create conversation (auth required)
print("\n3. Testing conversation creation...")
try:
    create_data = {
        "title": "Test Auth Conversation",
        "metadata": {"test": True}
    }
    response = requests.post(f"{BASE_URL}/conversations", headers=headers, json=create_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        conv_data = response.json()
        print(f"   Created conversation: {conv_data['conversation_id']}")
        print(f"   Title: {conv_data['title']}")
        print(f"   User ID: {conv_data['user_id']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
print("✅ Backend authentication test complete!")
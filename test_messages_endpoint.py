import requests
import json

BASE_URL = "http://localhost:8000"

# Test getting messages from conversation with 0 messages
conv_id = "7906e36a-47f1-45f4-9d93-c74d0559862d"  # This one has 0 messages
print(f"Testing GET /conversations/{conv_id}/messages")

try:
    response = requests.get(f"{BASE_URL}/conversations/{conv_id}/messages")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, default=str)}")
except Exception as e:
    print(f"Error: {e}")

# Test getting messages from conversation with 2 messages
conv_id = "3ec100e0-a5c9-4aee-b7b0-6ad8ca3571b0"  # This one has 2 messages
print(f"\nTesting GET /conversations/{conv_id}/messages")

try:
    response = requests.get(f"{BASE_URL}/conversations/{conv_id}/messages")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Got {len(data.get('messages', []))} messages")
    if 'messages' in data:
        for msg in data['messages']:
            print(f"  - {msg['role']}: {msg['content'][:50]}...")
except Exception as e:
    print(f"Error: {e}")

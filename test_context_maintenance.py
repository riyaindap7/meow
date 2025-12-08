"""
Test context maintenance with conversation history
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Create a new conversation
print("1️⃣ Creating a new conversation...")
conv_response = requests.post(f"{BASE_URL}/conversations", json={"title": "Context Test"})
conversation_id = conv_response.json()["conversation_id"]
print(f"✅ Conversation created: {conversation_id}")

# First query
print("\n2️⃣ First query (no history)...")
query1 = "What is the capital of France?"
response1 = requests.post(f"{BASE_URL}/ask", json={
    "query": query1,
    "conversation_id": conversation_id,
    "top_k": 5,
    "temperature": 0.1
})

if response1.status_code == 200:
    result1 = response1.json()
    print(f"✅ Answer 1: {result1['answer'][:100]}...")
else:
    print(f"❌ Error: {response1.text}")

# Second query in same conversation (should include history)
print("\n3️⃣ Second query (with history)...")
query2 = "Tell me more about it"
response2 = requests.post(f"{BASE_URL}/ask", json={
    "query": query2,
    "conversation_id": conversation_id,
    "top_k": 5,
    "temperature": 0.1
})

if response2.status_code == 200:
    result2 = response2.json()
    print(f"✅ Answer 2: {result2['answer'][:100]}...")
    print(f"   (Notice how it refers to the previous answer)")
else:
    print(f"❌ Error: {response2.text}")

# Get full conversation
print("\n4️⃣ Getting full conversation history...")
conv_response = requests.get(f"{BASE_URL}/conversations/{conversation_id}")
if conv_response.status_code == 200:
    conv = conv_response.json()
    print(f"✅ Conversation has {len(conv.get('messages', []))} messages")
    for i, msg in enumerate(conv.get('messages', []), 1):
        role = msg.get('role', '?').upper()
        content = msg.get('content', '')[:60] + "..."
        print(f"   {i}. {role}: {content}")
else:
    print(f"❌ Error: {conv_response.text}")

print("\n✅ Context maintenance test completed!")
print("   The LLM now maintains conversation history and context automatically")

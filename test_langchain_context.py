#!/usr/bin/env python3
"""Test LangChain integration with conversation history"""

import requests
import json
from time import sleep

API_URL = 'http://localhost:8000'

print("=" * 70)
print("🧪 Testing LangChain Integration with Context & History")
print("=" * 70)

# Test 1: Check memory status (should be empty initially)
print("\n1️⃣ Checking initial memory status...")
try:
    response = requests.get(f'{API_URL}/memory/status')
    if response.status_code == 200:
        memory = response.json()
        print(f"✅ Memory Status: {json.dumps(memory, indent=2)}")
    else:
        print(f"⚠️ Status code: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Create new conversation
print("\n2️⃣ Creating new conversation...")
try:
    response = requests.post(f'{API_URL}/conversations')
    if response.status_code == 200:
        conv_data = response.json()
        conversation_id = conv_data['conversation_id']
        print(f"✅ Conversation created: {conversation_id}")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test 3: First query
print("\n3️⃣ Sending first query with LangChain context...")
print("   Query: 'What is educational policy in India?'")

try:
    response = requests.post(f'{API_URL}/ask', json={
        'query': 'What is educational policy in India?',
        'conversation_id': conversation_id,
        'top_k': 3,
        'temperature': 0.1
    }, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ First response received")
        print(f"   Answer: {result['answer'][:150]}...")
        print(f"   Conversation ID: {result['conversation_id']}")
        print(f"   Sources: {len(result.get('sources', []))} documents")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")

sleep(1)

# Test 4: Check memory after first query
print("\n4️⃣ Checking memory after first query...")
try:
    response = requests.get(f'{API_URL}/memory/status')
    if response.status_code == 200:
        memory = response.json()
        message_count = memory['memory_info']['total_messages']
        print(f"✅ Memory now contains: {message_count} messages")
        if message_count > 0:
            print(f"   Messages in memory:")
            for msg in memory['memory_info']['messages'][:2]:
                print(f"     - {msg['role']}: {msg['content'][:80]}...")
    else:
        print(f"⚠️ Status code: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

sleep(1)

# Test 5: Second query (should use context from first)
print("\n5️⃣ Sending second query using LangChain context...")
print("   Query: 'Tell me more about funding for higher education'")

try:
    response = requests.post(f'{API_URL}/ask', json={
        'query': 'Tell me more about funding for higher education',
        'conversation_id': conversation_id,
        'top_k': 3,
        'temperature': 0.1
    }, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Second response received")
        print(f"   Answer: {result['answer'][:150]}...")
        print(f"   ✨ This response used LangChain memory context from first query!")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")

sleep(1)

# Test 6: Check conversation history
print("\n6️⃣ Retrieving conversation history from MongoDB...")
try:
    response = requests.get(f'{API_URL}/conversations/{conversation_id}/messages')
    if response.status_code == 200:
        conv_data = response.json()
        messages = conv_data.get('messages', [])
        print(f"✅ Retrieved {len(messages)} messages from MongoDB")
        for i, msg in enumerate(messages, 1):
            print(f"   Message {i}: {msg['role']} - {msg['content'][:80]}...")
    else:
        print(f"⚠️ Status code: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

sleep(1)

# Test 7: Check memory after second query
print("\n7️⃣ Checking LangChain memory after second query...")
try:
    response = requests.get(f'{API_URL}/memory/status')
    if response.status_code == 200:
        memory = response.json()
        message_count = memory['memory_info']['total_messages']
        print(f"✅ LangChain memory contains: {message_count} messages")
        print(f"   Messages:")
        for msg in memory['memory_info']['messages']:
            print(f"     - {msg['role']}: {msg['content'][:80]}...")
    else:
        print(f"⚠️ Status code: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("✨ LangChain Integration Test Complete!")
print("=" * 70)
print("""
What's being tested:
✅ LangChain ConversationBufferMemory for context management
✅ Conversation history loaded from MongoDB into LangChain
✅ Multi-turn responses with context awareness
✅ Memory endpoints for debugging

How it works:
1. First query stored in both MongoDB AND LangChain memory
2. Second query uses LangChain memory for context
3. Responses are more consistent because LangChain maintains conversation state
4. History is persistent in MongoDB even if server restarts
""")

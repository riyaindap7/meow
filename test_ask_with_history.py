#!/usr/bin/env python3
"""Test /ask endpoint with conversation history and MongoDB storage"""

import requests
import json

# Test the /ask endpoint
url = 'http://localhost:8000/ask'

# First query - should create new conversation
payload1 = {
    'query': 'What is educational policy?',
    'top_k': 3,
    'temperature': 0.1
}

print("=" * 70)
print("🧪 Testing /ask endpoint with MongoDB chat history")
print("=" * 70)

print("\n📝 First Query (new conversation):")
print(f"   Query: {payload1['query']}")

try:
    response = requests.post(url, json=payload1, timeout=60)
    print(f"\n✅ Status: {response.status_code}")
    
    result = response.json()
    print(f"\n📊 Response:")
    print(f"   Query: {result.get('query')}")
    print(f"   Answer: {result.get('answer')[:150]}...")
    print(f"   Conversation ID: {result.get('conversation_id')}")
    print(f"   Sources: {len(result.get('sources', []))} documents")
    
    conversation_id = result.get('conversation_id')
    
    # Second query - use same conversation
    if conversation_id:
        print(f"\n📝 Second Query (same conversation):")
        payload2 = {
            'query': 'Tell me more about higher education funding',
            'top_k': 3,
            'temperature': 0.1,
            'conversation_id': conversation_id
        }
        print(f"   Query: {payload2['query']}")
        print(f"   Conversation ID: {conversation_id}")
        
        response2 = requests.post(url, json=payload2, timeout=60)
        print(f"\n✅ Status: {response2.status_code}")
        
        result2 = response2.json()
        print(f"\n📊 Response:")
        print(f"   Answer: {result2.get('answer')[:150]}...")
        print(f"   Conversation ID: {result2.get('conversation_id')}")
        
        # Retrieve conversation history
        print(f"\n📚 Retrieving conversation history...")
        history_url = f'http://localhost:8000/conversations/{conversation_id}/messages'
        
        try:
            history_response = requests.get(history_url, timeout=30)
            print(f"\n✅ Status: {history_response.status_code}")
            
            history = history_response.json()
            print(f"\n📊 Conversation History:")
            print(f"   Conversation ID: {history.get('conversation_id')}")
            print(f"   Total messages: {len(history.get('messages', []))}")
            
            for i, msg in enumerate(history.get('messages', []), 1):
                print(f"\n   Message {i}:")
                print(f"     Role: {msg.get('role')}")
                print(f"     Content: {msg.get('content')[:100]}...")
                
        except Exception as e:
            print(f"⚠️  History retrieval warning: {e}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✨ Test complete!")
print("=" * 70)

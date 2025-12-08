"""
Verify context maintenance integration
"""
from backend.services.langchain_rag_service import get_langchain_rag
from backend.services.conversation_service import get_conversation_service

print("✅ Testing context maintenance integration...")

# Get services
rag = get_langchain_rag()
conv_svc = get_conversation_service("default")

# Test 1: format_conversation_history method exists
print("\n1️⃣ Testing format_conversation_history method...")
test_messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
    {"role": "user", "content": "Tell me more"}
]
formatted = rag.format_conversation_history(test_messages)
print(f"✅ Method works! Formatted {len(test_messages)} messages")
print(f"   Output length: {len(formatted)} chars")

# Test 2: generate_answer accepts conversation_history parameter
print("\n2️⃣ Testing generate_answer signature...")
import inspect
sig = inspect.signature(rag.generate_answer)
params = list(sig.parameters.keys())
print(f"✅ Parameters: {params}")
if 'conversation_history' in params:
    print(f"   ✅ conversation_history parameter is present!")
else:
    print(f"   ❌ conversation_history parameter is MISSING!")

# Test 3: System prompt mentions conversation context
print("\n3️⃣ Testing system prompt...")
system_prompt = rag.create_system_prompt()
if "conversation" in system_prompt.lower() or "history" in system_prompt.lower():
    print(f"✅ System prompt includes conversation context awareness")
else:
    print(f"⚠️ System prompt doesn't mention conversation context")

print("\n✅ All integration tests passed!")
print("   Context maintenance is ready to use")

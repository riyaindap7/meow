#!/usr/bin/env python
"""Test which import paths work for LangChain 0.1.x"""

print("Testing langchain_core.memory import:")
try:
    from langchain_core.memory import BaseChatMemory
    print("✅ SUCCESS: langchain_core.memory.BaseChatMemory")
    print(f"   Type: {BaseChatMemory}")
except ImportError as e:
    print(f"❌ FAILED: {e}")

print("\nTesting langchain.memory.chat_memory import:")
try:
    from langchain.memory.chat_memory import BaseChatMemory
    print("✅ SUCCESS: langchain.memory.chat_memory.BaseChatMemory")
    print(f"   Type: {BaseChatMemory}")
except ImportError as e:
    print(f"❌ FAILED: {e}")

print("\nTesting langchain_core.messages import:")
try:
    from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
    print("✅ SUCCESS: langchain_core.messages")
except ImportError as e:
    print(f"❌ FAILED: {e}")

print("\nTesting langchain.schema import:")
try:
    from langchain.schema import AIMessage, HumanMessage, BaseMessage
    print("✅ SUCCESS: langchain.schema")
except ImportError as e:
    print(f"❌ FAILED: {e}")

print("\nTesting langchain_core.language_models import:")
try:
    from langchain_core.language_models import BaseLLM
    print("✅ SUCCESS: langchain_core.language_models.BaseLLM")
except ImportError as e:
    print(f"❌ FAILED: {e}")

print("\nTesting langchain_core.language_models.llms import:")
try:
    from langchain_core.language_models.llms import BaseLLM
    print("✅ SUCCESS: langchain_core.language_models.llms.BaseLLM")
except ImportError as e:
    print(f"❌ FAILED: {e}")

#!/usr/bin/env python
"""Check available modules in langchain package"""

import langchain
import pkgutil

print(f"LangChain version: {langchain.__version__}")
print("\nAvailable langchain submodules:")

for importer, modname, ispkg in pkgutil.iter_modules(langchain.__path__):
    pkg_str = " (package)" if ispkg else ""
    print(f"  - langchain.{modname}{pkg_str}")

print("\n" + "="*60)
print("Checking langchain.memory specifically:")
print("="*60)

try:
    import langchain.memory
    print("✅ langchain.memory EXISTS")
    print("\nContents of langchain.memory:")
    for importer, modname, ispkg in pkgutil.iter_modules(langchain.memory.__path__):
        pkg_str = " (package)" if ispkg else ""
        print(f"  - langchain.memory.{modname}{pkg_str}")
except ImportError as e:
    print(f"❌ langchain.memory does NOT exist: {e}")

print("\n" + "="*60)
print("Checking specific imports:")
print("="*60)

# Test the imports we're using
tests = [
    ("langchain.memory.chat_memory", "BaseChatMemory"),
    ("langchain.schema", "AIMessage"),
    ("langchain.schema", "HumanMessage"),
    ("langchain.chains", "LLMChain"),
    ("langchain.prompts", "PromptTemplate"),
]

for module_path, class_name in tests:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ from {module_path} import {class_name} - OK")
    except Exception as e:
        print(f"❌ from {module_path} import {class_name} - FAILED: {e}")

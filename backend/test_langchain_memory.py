#!/usr/bin/env python
"""Test LangChain memory integration patterns"""

import langchain
print(f"LangChain version: {langchain.__version__}")

# Test 1: Check LLMChain signature
print("\n" + "="*60)
print("Test 1: LLMChain memory parameter")
print("="*60)

from langchain.chains import LLMChain
import inspect

sig = inspect.signature(LLMChain.__init__)
print(f"LLMChain.__init__ signature:\n{sig}")

# Test 2: Check what ConversationChain expects
print("\n" + "="*60)
print("Test 2: ConversationChain (recommended for memory)")
print("="*60)

try:
    from langchain.chains import ConversationChain
    sig = inspect.signature(ConversationChain.__init__)
    print(f"✅ ConversationChain exists")
    print(f"ConversationChain.__init__ signature:\n{sig}")
except ImportError as e:
    print(f"❌ ConversationChain not available: {e}")

# Test 3: Check if RunnableSequence exists
print("\n" + "="*60)
print("Test 3: RunnableSequence (LCEL)")
print("="*60)

try:
    from langchain.runnables import RunnableSequence
    print(f"✅ RunnableSequence exists")
    print(f"   Module: {RunnableSequence.__module__}")
except ImportError as e:
    print(f"❌ RunnableSequence not available: {e}")

# Test 4: Test BaseChatMemory interface
print("\n" + "="*60)
print("Test 4: BaseChatMemory interface")
print("="*60)

from langchain.memory.chat_memory import BaseChatMemory
print(f"BaseChatMemory methods:")
for attr in dir(BaseChatMemory):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Test 5: Check ConversationBufferMemory as reference
print("\n" + "="*60)
print("Test 5: ConversationBufferMemory structure")
print("="*60)

from langchain.memory import ConversationBufferMemory
mem = ConversationBufferMemory()
print(f"ConversationBufferMemory type: {type(mem)}")
print(f"Is BaseChatMemory: {isinstance(mem, BaseChatMemory)}")
print(f"memory_variables: {mem.memory_variables}")

# Test 6: Try creating a simple LLMChain with memory
print("\n" + "="*60)
print("Test 6: Create LLMChain with ConversationBufferMemory")
print("="*60)

try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain_core.language_models.llms import BaseLLM
    
    class DummyLLM(BaseLLM):
        @property
        def _llm_type(self):
            return "dummy"
        
        def _call(self, prompt, stop=None):
            return "test response"
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template="{chat_history}\nUser: {input}\nAI:"
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = DummyLLM()
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    print("✅ LLMChain with memory created successfully!")
    print(f"   Chain type: {type(chain)}")
    
except Exception as e:
    print(f"❌ Failed to create LLMChain with memory: {e}")
    import traceback
    traceback.print_exc()

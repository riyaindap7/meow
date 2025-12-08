"""
Full LangChain Integration Service using OpenRouter
TRUE LangChain Memory Implementation with BaseChatMemory
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx
import json
import time
# LangChain imports
try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_core.language_models.llms import BaseLLM

    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain imports successful - TRUE LangChain mode enabled!")
except ImportError as e:
    print(f"⚠️ LangChain imports failed - using manual fallback mode: {repr(e)}")
    LANGCHAIN_AVAILABLE = False
    BaseMemory = object
    AIMessage = dict
    HumanMessage = dict
    BaseLLM = object

# ========== TRUE LANGCHAIN MEMORY ==========
class MongoConversationMemory(BaseMemory):
    """
    LangChain-compatible Memory implementation for MongoDB
    - Inherits from BaseMemory for proper LangChain integration
    - Returns LangChain Message objects (HumanMessage, AIMessage)
    - Works with LangChain 0.1.x
    """
    
    # Pydantic v1 field declarations (required for BaseMemory)
    collection: Any = None
    conversation_id: str = ""
    user_id: str = ""
    chat_memory: List = []
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, mongo_collection, conversation_id: str, user_id: str, **kwargs):
        super().__init__(
            collection=mongo_collection,
            conversation_id=conversation_id,
            user_id=user_id,
            chat_memory=[],
            **kwargs
        )
    
    @property
    def memory_variables(self):
        """Define what variables this memory provides"""
        return ["chat_history"]
    
    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """
        Load conversation history from MongoDB
        Returns LangChain Message objects (HumanMessage, AIMessage)
        """
        try:
            # Find conversation in MongoDB
            convo = self.collection.find_one({
                "conversation_id": self.conversation_id,
                "user_id": self.user_id
            })
            
            if not convo or "messages" not in convo:
                return {"chat_history": []}
            
            # Convert to LangChain Message objects
            messages = []
            if LANGCHAIN_AVAILABLE:
                for msg in convo["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
            else:
                # Fallback for when LangChain not available
                for msg in convo["messages"]:
                    messages.append({
                        "role": msg.get("role", ""),
                        "content": msg.get("content", "")
                    })
            
            self.chat_memory = messages  # Cache in memory
            return {"chat_history": messages}
            
        except Exception as e:
            print(f"⚠️ Error loading memory: {str(e)}")
            return {"chat_history": []}
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """
        Save conversation context to MongoDB
        Called automatically by LangChain after chain execution
        """
        try:
            from datetime import datetime
            
            print("\n" + "="*80)
            print("💾 SAVE_CONTEXT CALLED - Saving to MongoDB")
            print("="*80)
            print(f"   Conversation ID: {self.conversation_id}")
            print(f"   User ID: {self.user_id}")
            print(f"   Input: {inputs.get('input', '')[:100]}...")
            print(f"   Output keys: {list(outputs.keys())}")
            
            # Extract output - LangChain might use 'output', 'text', or 'response'
            output_content = outputs.get("output") or outputs.get("text") or outputs.get("response") or ""
            print(f"   Output content (first 100 chars): {output_content[:100]}...")
            
            user_msg = {
                "role": "user",
                "content": inputs.get("input", ""),
                "timestamp": time.time(),
                "created_at": datetime.utcnow()
            }
            
            ai_msg = {
                "role": "assistant", 
                "content": output_content,
                "timestamp": time.time(),
                "created_at": datetime.utcnow(),
                "sources": outputs.get("sources", [])
            }
            
            # Check if conversation exists
            print(f"🔍 Checking if conversation exists...")
            existing = self.collection.find_one({
                "conversation_id": self.conversation_id,
                "user_id": self.user_id
            })
            
            if existing:
                print(f"   ✅ Conversation exists with {len(existing.get('messages', []))} messages")
            else:
                print(f"   ⚠️ Conversation does NOT exist - will create with upsert")
            
            update_fields = {
                "$push": {
                    "messages": {
                        "$each": [user_msg, ai_msg]
                    }
                },
                "$set": {
                    "updated_at": datetime.utcnow()
                }
            }
            
            # If conversation doesn't exist, set initial fields
            if not existing:
                update_fields["$setOnInsert"] = {
                    "created_at": datetime.utcnow(),
                    "title": inputs.get("input", "")[:50],  # Use first query as title
                    "archived": False
                }
                print(f"   📝 Will create conversation with title: '{inputs.get('input', '')[:50]}'")
            # If this is the first message (only has 0 messages), update title
            elif existing and len(existing.get("messages", [])) == 0:
                update_fields["$set"]["title"] = inputs.get("input", "")[:50]
                print(f"   📝 Updating title: '{inputs.get('input', '')[:50]}'")
            
            # Update MongoDB with new messages
            print(f"💾 Executing MongoDB update_one...")
            result = self.collection.update_one(
                {
                    "conversation_id": self.conversation_id,
                    "user_id": self.user_id
                },
                update_fields,
                upsert=True
            )
            
            print(f"✅ MongoDB update completed!")
            print(f"   Matched: {result.matched_count}")
            print(f"   Modified: {result.modified_count}")
            print(f"   Upserted ID: {result.upserted_id}")
            
            # Verify it was saved
            verify = self.collection.find_one({"conversation_id": self.conversation_id})
            if verify:
                print(f"✅ VERIFIED: Conversation now has {len(verify.get('messages', []))} messages")
            else:
                print(f"❌ VERIFICATION FAILED: Conversation not found after save!")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"⚠️ Error saving memory: {str(e)}")
    
    def clear(self):
        """Clear conversation history"""
        try:
            self.collection.delete_one({
                "conversation_id": self.conversation_id,
                "user_id": self.user_id
            })
            self.chat_memory = []
            print(f"✅ Memory cleared for conversation: {self.conversation_id}")
        except Exception as e:
            print(f"⚠️ Error clearing memory: {str(e)}")

class OpenRouterLLM:
    """OpenRouter LLM wrapper for RAG operations"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "alibaba/tongyi-deepresearch-30b-a3b")
        self.site_url = os.getenv("SITE_URL", "http://localhost:3000")
        self.site_name = os.getenv("SITE_NAME", "VICTOR")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.temperature = 0.1
        self.max_tokens = 2000
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0)
        )
    
    def generate(self, prompt: str, temperature: float = None) -> str:
        """Generate response using OpenRouter"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature or self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = self.client.post(self.base_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"OpenRouter LLM call failed: {str(e)}")

# ========== TRUE LANGCHAIN CHAIN WITH AUTO-MEMORY ==========
class TrueLangChainRAG:
    """
    TRUE LangChain Chain Implementation
    - Uses LLMChain with memory=... parameter
    - Memory automatically loads, injects, and saves
    - No manual memory.load() or memory.save() calls needed
    """
    
    def __init__(self, llm: OpenRouterLLM, retriever, memory: MongoConversationMemory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        
        # Create LangChain prompt template
        print(f"🔍 LANGCHAIN_AVAILABLE: {LANGCHAIN_AVAILABLE}")
        if LANGCHAIN_AVAILABLE:
            try:
                print("🔧 Creating PromptTemplate...")
                self.prompt_template = PromptTemplate(
                    input_variables=["chat_history", "context", "input"],
                    template="""You are an intelligent assistant that answers questions based on provided documents and conversation history.

Instructions:
- Answer ONLY using information from the provided context and conversation history
- If information is not available, say "I cannot answer this based on the provided documents"
- Always cite source documents and page numbers when referencing them
- Be aware of the conversation history and refer to previous points when relevant
- When user asks "elaborate more" or similar, use the conversation history to understand what to elaborate on
- Be accurate, concise, and maintain conversation continuity

{chat_history}

Document Context:
{context}

Current Question: {input}

Answer:"""
                )
                
                print("🔧 Creating LLM wrapper...")
                llm_wrapper = self._create_langchain_llm_wrapper()
                print(f"✅ LLM wrapper created: {type(llm_wrapper)}")
                
                print("🔧 Creating LLMChain with memory...")
                # Create LangChain LLMChain with automatic memory
                # LLMChain supports custom input variables (like 'context' for RAG)
                self.chain = LLMChain(
                    llm=llm_wrapper,
                    prompt=self.prompt_template,
                    memory=self.memory,  # ← Memory auto-injects chat_history!
                    verbose=True
                )
                print("✅ TRUE LangChain LLMChain created with automatic memory")
            except Exception as e:
                print(f"❌ Error creating LangChain LLMChain: {str(e)}")
                import traceback
                traceback.print_exc()
                self.chain = None
                print("⚠️ Falling back to manual mode")
        else:
            self.chain = None
            print("⚠️ LangChain not available - using manual mode")
    
    def _create_langchain_llm_wrapper(self):
        """Create a proper LangChain BaseLLM wrapper"""
        
        if not LANGCHAIN_AVAILABLE:
            return None
        
        from langchain_core.outputs import Generation, LLMResult
        
        class OpenRouterLLMWrapper(BaseLLM):
            """Proper LangChain BaseLLM implementation for OpenRouter"""
            
            llm: Any = None
            
            class Config:
                arbitrary_types_allowed = True
            
            def __init__(self, llm_instance):
                super().__init__()
                object.__setattr__(self, 'llm', llm_instance)
            
            @property
            def _llm_type(self) -> str:
                return "openrouter"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                """Call OpenRouter API"""
                return self.llm.generate(prompt)
            
            def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
                """Generate responses for multiple prompts"""
                generations = []
                for prompt in prompts:
                    text = self.llm.generate(prompt)
                    generations.append([Generation(text=text)])
                return LLMResult(generations=generations)
        
        return OpenRouterLLMWrapper(self.llm)
    
    def _format_chat_history(self, chat_history: List) -> str:
        """Format chat history for prompt"""
        if not chat_history:
            return ""
        
        history_text = "\n=== Conversation History ===\n"
        for msg in chat_history[-10:]:
            if LANGCHAIN_AVAILABLE and hasattr(msg, 'content'):
                # LangChain Message object
                role = "USER" if isinstance(msg, HumanMessage) else "ASSISTANT"
                content = msg.content
            else:
                # Dict format
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
            
            if len(content) > 200:
                content = content[:200] + "..."
            history_text += f"{role}: {content}\n"
        
        return history_text
    
    def _format_contexts(self, contexts: List[Dict]) -> str:
        """Format document contexts"""
        if not contexts:
            return "No relevant documents found."
        
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"\n[Source {i}]\n"
                f"Document: {ctx.get('source_file', 'Unknown')}\n"
                f"Page: {ctx.get('page_idx', 'N/A')}\n"
                f"Content: {ctx.get('text', '')}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def invoke(self, inputs: Dict) -> Dict:
        """
        Execute TRUE LangChain pipeline
        - memory.load() → automatic
        - prompt injection → automatic
        - llm.generate() → automatic
        - memory.save() → automatic
        """
        try:
            query = inputs.get("input", "")
            
            print(f"🔵 TRUE LangChain RAG Chain executing: {query}")
            
            # 1. Retrieve documents (still manual - retrieval happens before chain)
            contexts = self.retriever.retrieve(query)
            print(f"📚 Retrieved {len(contexts)} documents")
            
            # 2. Format context
            context_text = self._format_contexts(contexts)
            
            if self.chain and LANGCHAIN_AVAILABLE:
                # TRUE LangChain automatic flow:
                # chain.invoke() → memory loads → injects history → LLM generates → memory saves
                print("🔄 Executing LLMChain with automatic memory...")
                
                result = self.chain.invoke({
                    "input": query,
                    "context": context_text
                })
                
                answer = result.get("text", result.get("output", ""))
                
                print("✅ LangChain auto-memory completed")
            
            else:
                # Fallback manual mode
                print("🔄 Manual mode (LangChain not available)...")
                
                memory_vars = self.memory.load_memory_variables({})
                chat_history = memory_vars.get("chat_history", [])
                
                history_text = self._format_chat_history(chat_history)
                
                prompt = f"""{history_text}

Document Context:
{context_text}

Current Question: {query}

Answer:"""
                
                answer = self.llm.generate(prompt, temperature=inputs.get("temperature", 0.1))
                
                self.memory.save_context(
                    inputs={"input": query},
                    outputs={"output": answer, "sources": contexts}
                )
            
            return {
                "output": answer,
                "contexts": contexts
            }
            
        except Exception as e:
            print(f"❌ Chain invoke error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

class DocumentRetriever:
    """Document retriever for Milvus"""
    
    def __init__(self, milvus_client):
        self.milvus_client = milvus_client
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        try:
            if self.milvus_client:
                return self.milvus_client.search(query=query, top_k=top_k)
            return []
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []

class FullLangChainRAG:
    """
    Complete RAG implementation with TRUE LangChain Memory
    Uses MongoConversationMemory (BaseChatMemory) with automatic memory handling
    """
    
    def __init__(self):
        print("🔧 Initializing Full LangChain RAG with TRUE Memory...")
        
        # Initialize OpenRouter LLM
        self.llm = OpenRouterLLM()
        self.model_name = self.llm.model_name
        
        # Initialize Milvus client
        self.milvus_client = None
        try:
            from api.milvus_client import get_milvus_client
            self.milvus_client = get_milvus_client()
            print("✅ Milvus client initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize Milvus: {str(e)}")
        
        # Initialize document retriever
        self.retriever = DocumentRetriever(self.milvus_client)
        
        # Initialize MongoDB connection for TRUE LangChain memory
        self.mongodb_service = None
        self.conversations_collection = None
        print("\n" + "="*80)
        print("🔵 INITIALIZING MONGODB FOR CONVERSATION PERSISTENCE")
        print("="*80)
        try:
            from services.mongodb_health import is_mongodb_available
            
            print("🔍 Checking MongoDB availability...")
            if is_mongodb_available():
                print("✅ MongoDB is available!")
                # Use the function-based MongoDB service
                from services.mongodb_service import get_mongo_db
                db = get_mongo_db()
                self.conversations_collection = db.conversations
                self.mongodb_service = True  # Flag to indicate MongoDB is available
                print("✅ MongoDB conversations collection initialized for TRUE LangChain memory")
                print(f"   Database: {db.name}")
                print(f"   Collection: conversations")
                print(f"   Current conversations count: {self.conversations_collection.count_documents({})}")
            else:
                print("❌ MongoDB is NOT available - memory will not persist")
                print("💡 Check:")
                print("   1. MongoDB is running: 'net start MongoDB' or docker ps")
                print("   2. MONGODB_URI in .env is correct")
                print("   3. MongoDB is accessible at localhost:27017")
        except Exception as e:
            print(f"❌ Could not initialize MongoDB: {str(e)}")
            import traceback
            traceback.print_exc()
        print("="*80 + "\n")
        
        print("✅ Full LangChain RAG with TRUE Memory initialized successfully")
    
    def ask(self, 
           query: str, 
           conversation_id: str = None, 
           user_id: str = None,
           temperature: float = 0.1) -> Dict[str, Any]:
        """
        Ask a question using TRUE LangChain pipeline
        - Memory automatically loads from MongoDB
        - Chain automatically injects history into prompt
        - Memory automatically saves after response
        """
        
        try:
            print(f"🔵 TRUE LangChain RAG Query: {query}")
            print(f"   Conversation ID: {conversation_id}")
            print(f"   User ID: {user_id}")
            
            # Create TRUE LangChain Memory (BaseChatMemory)
            if conversation_id and user_id and self.conversations_collection is not None:
                memory = MongoConversationMemory(
                    self.conversations_collection,
                    conversation_id,
                    user_id
                )
                print("✅ TRUE LangChain Memory (BaseChatMemory) created")
            else:
                # No-op memory for sessions without persistence
                memory = type('obj', (object,), {
                    'memory_variables': ["chat_history"],
                    'load_memory_variables': lambda self, inputs: {"chat_history": []},
                    'save_context': lambda self, inputs, outputs: None,
                    'clear': lambda self: None
                })()
                print("⚠️ Using no-op memory (no persistence)")
            
            # Create TRUE LangChain Chain with automatic memory
            chain = TrueLangChainRAG(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory
            )
            
            # Execute chain - memory loads, injects, and saves automatically!
            print("🔄 Executing TRUE LangChain chain with auto-memory...")
            result = chain.invoke({
                "input": query,
                "temperature": temperature
            })
            
            # Extract results
            answer = result.get("output", "")
            contexts = result.get("contexts", [])
            
            # Format sources
            sources = []
            for ctx in contexts:
                sources.append({
                    "text": ctx.get('text', '')[:200] + "..." if len(ctx.get('text', '')) > 200 else ctx.get('text', ''),
                    "source_file": ctx.get("source_file", ""),
                    "page_idx": ctx.get("page_idx", 0),
                    "score": ctx.get("score", 0.0),
                    "global_chunk_id": ctx.get("global_chunk_id"),
                    "document_id": ctx.get("document_id"),
                    "chunk_index": ctx.get("chunk_index"),
                    "section_hierarchy": ctx.get("section_hierarchy"),
                    "char_count": ctx.get("char_count"),
                    "word_count": ctx.get("word_count")
                })
            
            print("✅ TRUE LangChain RAG completed (memory auto-saved)")
            
            return {
                "answer": answer,
                "sources": sources,
                "conversation_id": conversation_id,
                "model_used": self.model_name
            }
            
        except Exception as e:
            print(f"❌ Error in TRUE LangChain RAG: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_new_conversation(self, title: str, user_id: str, metadata: Dict = None) -> str:
        """Create a new conversation"""
        try:
            if self.mongodb_service:
                # Use ConversationService for creating conversations
                from services.conversation_service import get_conversation_service
                conv_service = get_conversation_service()
                result = conv_service.create_conversation(
                    user_id=user_id,
                    title=title,
                    metadata=metadata or {}
                )
                return result.get("conversation_id")
            else:
                # Generate conversation ID for in-memory storage
                import uuid
                conversation_id = str(uuid.uuid4())
                return conversation_id
        except Exception as e:
            print(f"❌ Error creating conversation: {str(e)}")
            raise
    
    def get_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for a user"""
        try:
            if self.mongodb_service:
                # Use ConversationService for getting conversations
                from services.conversation_service import get_conversation_service
                conv_service = get_conversation_service()
                conversations = conv_service.get_user_conversations(user_id)
                
                # Format the conversations to ensure datetime fields are strings
                formatted_conversations = []
                for conv in conversations:
                    try:
                        # Convert created_at to string if it's datetime
                        created_at = conv.get("created_at")
                        if created_at:
                            if hasattr(created_at, 'isoformat'):
                                created_at = created_at.isoformat()
                            elif not isinstance(created_at, str):
                                created_at = str(created_at)
                        else:
                            created_at = "2023-01-01T00:00:00"
                        
                        # Convert updated_at to string if it's datetime
                        updated_at = conv.get("updated_at", created_at)
                        if updated_at:
                            if hasattr(updated_at, 'isoformat'):
                                updated_at = updated_at.isoformat()
                            elif not isinstance(updated_at, str):
                                updated_at = str(updated_at)
                        else:
                            updated_at = created_at
                        
                        formatted_conversations.append({
                            "conversation_id": conv.get("conversation_id", ""),
                            "user_id": conv.get("user_id", user_id),
                            "title": conv.get("title", "Untitled"),
                            "created_at": created_at,
                            "updated_at": updated_at,
                            "messages": conv.get("messages", [])
                        })
                    except Exception as e:
                        print(f"⚠️ Error formatting conversation: {str(e)}")
                        continue
                
                return formatted_conversations
            else:
                # Return empty list for in-memory storage
                return []
        except Exception as e:
            print(f"❌ Error getting conversations: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

# Global instance
_full_langchain_rag = None

def get_full_langchain_rag() -> FullLangChainRAG:
    """Get or create FullLangChainRAG singleton"""
    global _full_langchain_rag
    if _full_langchain_rag is None:
        _full_langchain_rag = FullLangChainRAG()
    return _full_langchain_rag
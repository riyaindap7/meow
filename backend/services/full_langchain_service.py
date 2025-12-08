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
        Load conversation history from separate messages collection
        Returns LangChain Message objects (HumanMessage, AIMessage)
        """
        try:
            from .mongodb_service import mongodb_service
            
            # Get messages from separate collection
            messages_data = mongodb_service.get_last_messages(self.conversation_id, limit=10)
            
            if not messages_data:
                return {"chat_history": []}
            
            # Convert to LangChain Message objects
            messages = []
            if LANGCHAIN_AVAILABLE:
                for msg in messages_data:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
            else:
                # Fallback for when LangChain not available
                messages = messages_data
            
            self.chat_memory = messages  # Cache in memory
            return {"chat_history": messages}
            
        except Exception as e:
            print(f"⚠️ Error loading memory: {str(e)}")
            return {"chat_history": []}
    
    def save_context(self, inputs: Dict, outputs: Dict):
        """
        Save conversation context to separate messages collection
        Called automatically by LangChain after chain execution
        """
        try:
            from .mongodb_service import mongodb_service
            
            print("\n" + "="*80)
            print("💾 SAVE_CONTEXT CALLED - Saving to Messages Collection")
            print("="*80)
            print(f"   Conversation ID: {self.conversation_id}")
            print(f"   User ID: {self.user_id}")
            print(f"   Input: {inputs.get('input', '')[:100]}...")
            print(f"   Output keys: {list(outputs.keys())}")
            
            # Extract output - LangChain might use 'output', 'text', or 'response'
            output_content = outputs.get("output") or outputs.get("text") or outputs.get("response") or ""
            print(f"   Output content (first 100 chars): {output_content[:100]}...")
            
            # Ensure conversation exists
            existing_conversation = mongodb_service.get_conversation(self.conversation_id, self.user_id)
            if not existing_conversation:
                # Create conversation with title from first query
                user_query = inputs.get("input", "")
                if user_query:
                    # Generate smart title from first query
                    title = user_query.strip()
                    if len(title) > 50:
                        title = title[:47] + "..."
                else:
                    title = "New Conversation"
                
                mongodb_service.create_conversation(self.conversation_id, self.user_id, title)
                print(f"   📝 Created conversation with title: '{title}'")
            else:
                # If this is an existing conversation with "New Conversation" title, update it
                current_title = existing_conversation.get("title", "")
                if current_title == "New Conversation" and inputs.get("input", ""):
                    user_query = inputs.get("input", "")
                    new_title = user_query.strip()
                    if len(new_title) > 50:
                        new_title = new_title[:47] + "..."
                    
                    mongodb_service.update_conversation_title(self.conversation_id, self.user_id, new_title)
                    print(f"   📝 Updated conversation title: '{new_title}'")
                    
                    mongodb_service.update_conversation_title(self.conversation_id, self.user_id, new_title)
                    print(f"   📝 Updated conversation title to: '{new_title}'")
            
            # Add user message
            user_success = mongodb_service.add_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                role="user",
                content=inputs.get("input", ""),
                metadata={}
            )
            
            # Add assistant message  
            ai_success = mongodb_service.add_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                role="assistant",
                content=output_content,
                metadata={"sources": outputs.get("sources", [])}
            )
            
            if user_success and ai_success:
                print(f"✅ Messages saved to separate collection!")
                
                # Verify messages were saved
                messages = mongodb_service.get_last_messages(self.conversation_id, limit=2)
                print(f"   ✅ VERIFIED: Found {len(messages)} messages in conversation")
            else:
                print(f"❌ FAILED to save messages!")
            
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
    
    def invoke(self, query: str, top_k: int = 5, temperature: float = 0.1):
        """
        Invoke the TRUE LangChain RAG with role-based parameters
        """
        try:
            print(f"🎯 Using role-based top_k: {top_k}")
            print(f"🔍 Retrieving {top_k} documents")
            
            # Retrieve documents with role-based top_k
            contexts = self.retriever.similarity_search(query, k=top_k)
            print(f"📚 Retrieved {len(contexts)} documents")
            
            if not contexts:
                return [], "I couldn't find relevant documents to answer your question."
            
            # Format contexts for LLM
            formatted_contexts = []
            for i, doc in enumerate(contexts):
                context_dict = {
                    "text": doc.page_content,
                    "source": getattr(doc, 'metadata', {}).get('source', f'Document {i+1}'),
                    "page": getattr(doc, 'metadata', {}).get('page', 1),
                    "score": getattr(doc, 'metadata', {}).get('score', 0.0)
                }
                formatted_contexts.append(context_dict)
            
            # Generate answer using LLM with role-based temperature
            prompt = self._create_enhanced_prompt(query, formatted_contexts)
            
            # Use the LLM generate method with role-based temperature
            answer = self.llm.generate(prompt, temperature=temperature)
            
            return formatted_contexts, answer
            
        except Exception as e:
            print(f"❌ TRUE LangChain invoke error: {str(e)}")
            return [], f"Error in retrieval: {str(e)}"

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
    
    def _update_conversation_context(self, conversation_id: str, user_id: str, query: str, answer: str, contexts: List[Dict[str, Any]]):
        """
        Extract and update conversation context using LLM analysis
        """
        if self.conversations_collection is None or not conversation_id:
            return
        
        import json  # Import at the top of the method
        
        try:
            # Get conversation history for context
            conversation = self.conversations_collection.find_one({
                "conversation_id": conversation_id,
                "user_id": user_id
            })
            
            if not conversation:
                return
            
            # Get last few messages for analysis
            from .mongodb_service import mongodb_service
            recent_messages = mongodb_service.get_last_messages(conversation_id, limit=6)
            
            # Format conversation for analysis
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages[-6:]  # Last 3 exchanges
            ])
            
            # Format contexts for analysis
            context_info = "\n".join([
                f"- {ctx.get('source_file', 'Unknown')}: {ctx.get('text', '')[:100]}..."
                for ctx in contexts[:5]
            ])
            
            # LLM prompt for context extraction
            context_prompt = f"""
Analyze this conversation and extract key context information in JSON format.

Conversation:
{conversation_text}

Relevant Documents:
{context_info}

Extract and return a JSON object with:
{{
  "summary": "Brief 2-sentence summary of the conversation",
  "topics": ["list", "of", "key", "topics", "discussed"],
  "main_entities": ["important", "entities", "mentioned"],
  "user_goal": "What the user seems to be trying to achieve"
}}

Return only valid JSON, no extra text:"""
            
            # Get context extraction with retry logic
            for attempt in range(3):
                try:
                    context_response = self.llm.generate(context_prompt)
                    context_text = str(context_response)
                    
                    # Clean up response
                    context_text = context_text.strip()
                    if context_text.startswith('```json'):
                        context_text = context_text[7:]
                    if context_text.endswith('```'):
                        context_text = context_text[:-3]
                    context_text = context_text.strip()
                    
                    # Parse JSON
                    context_data = json.loads(context_text)
                    
                    # Update conversation with extracted context
                    mongodb_service.update_conversation_context(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        context_data=context_data
                    )
                    
                    print(f"✅ Context extracted and saved for conversation {conversation_id}")
                    break
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ Context extraction JSON error (attempt {attempt + 1}): {e}")
                    if attempt == 2:
                        print("❌ Failed to extract context after 3 attempts")
                except Exception as e:
                    print(f"⚠️ Context extraction error (attempt {attempt + 1}): {e}")
                    if attempt == 2:
                        print(f"❌ Failed to extract context: {e}")
                        
        except Exception as e:
            print(f"❌ Error updating conversation context: {e}")
    
    def ask(self, query: str, user_id: str, conversation_id: str = None, temperature: float = None, top_k: int = None, user: dict = None, dense_weight: float = 0.6, sparse_weight: float = 0.4, method: str = "hybrid"):
        """
        Ask a question using TRUE LangChain pipeline with role-based parameters
        - Memory automatically loads from MongoDB
        - Chain automatically injects history into prompt
        - Memory automatically saves after response
        - Applies role-based RAG parameters
        - Supports hybrid search with dense/sparse weights
        """
        
        # Apply role-based parameters if user is provided
        if user:
            from .role_config import build_chain_params
            user_role = user.get("role", "user")
            role_params = build_chain_params(user)
            print(f"📊 ROLE: {user_role} | PARAMS: temp={role_params.get('temperature')}, docs={role_params.get('top_k')}")
            
            # Use role-based parameters with override capability
            if temperature is None:
                temperature = role_params.get('temperature', 0.1)
            if top_k is None:
                top_k = role_params.get('top_k', 5)
            if dense_weight is None:
                dense_weight = role_params.get('dense_weight', 0.6)
            if sparse_weight is None:
                sparse_weight = role_params.get('sparse_weight', 0.4)
            if method == "hybrid":
                method = role_params.get('method', 'hybrid')
        else:
            # Default fallback values
            if temperature is None:
                temperature = 0.1
            if top_k is None:
                top_k = 5
            if dense_weight is None:
                dense_weight = 0.6
            if sparse_weight is None:
                sparse_weight = 0.4
        
        print(f"🚀 GENERATING RESPONSE with ROLE: {user.get('role', 'user') if user else 'unknown'}")
        
        # Use provided parameters or fallback to role config, then to defaults
        final_temperature = temperature if temperature is not None else role_config.get("temperature", 0.2)
        final_top_k = top_k if top_k is not None else role_config.get("top_k", 10)
        final_dense_weight = dense_weight if dense_weight is not None else role_config.get("dense_weight", 0.7)
        final_sparse_weight = sparse_weight if sparse_weight is not None else role_config.get("sparse_weight", 0.3)
        
        # Ensure minimum 10 documents
        if final_top_k < 10:
            final_top_k = 10
        
        print(f"📊 Final params: temperature={final_temperature}, top_k={final_top_k}, dense_weight={final_dense_weight}, sparse_weight={final_sparse_weight}, method={method}")
        
        try:
            # Generate conversation_id if not provided
            if not conversation_id:
                import uuid
                conversation_id = str(uuid.uuid4())
            
            # Create TRUE LangChain Memory (BaseChatMemory)
            if conversation_id and user_id and self.conversations_collection is not None:
                memory = MongoConversationMemory(
                    self.conversations_collection,
                    conversation_id,
                    user_id
                )
            else:
                # No-op memory for sessions without persistence
                memory = type('obj', (object,), {
                    'memory_variables': ["chat_history"],
                    'load_memory_variables': lambda self, inputs: {"chat_history": []},
                    'save_context': lambda self, inputs, outputs: None,
                    'clear': lambda self: None
                })()
            
            # Retrieve documents using Milvus with role-based parameters
            try:
                if self.milvus_client:
                    print(f"🔍 Retrieving {top_k} documents using {method} search")
                    contexts = self.milvus_client.search(
                        query=query,
                        top_k=top_k,
                        method=method
                    )
                    print(f"📚 Retrieved {len(contexts)} documents")
                else:
                    contexts = []
            except Exception as e:
                print(f"❌ Document retrieval error: {str(e)}")
                contexts = []
            
            # Format contexts for LLM
            if contexts:
                context_text = "\n\n".join([
                    f"[Source {i+1}] {ctx.get('document_name', 'Unknown')} (Page {ctx.get('page_idx', 'N/A')}):\n{ctx.get('text', '')}"
                    for i, ctx in enumerate(contexts[:5])
                ])
            else:
                context_text = "No relevant documents found."
            
            # Create enhanced prompt
            prompt = f"""You are VICTOR, a helpful, intelligent AI assistant specializing in government documents and policy materials. Answer the user's question using the information found in the provided context documents.

CONTEXT:
{context_text}

INSTRUCTIONS:
- Use the context as your primary reference while applying deep analytical reasoning
- You may reason and make logical connections between information from different documents
- When reasoning between documents, clearly indicate this is your logical inference based on the provided information
- If after reasoning through the documents you still cannot answer, reply: "I cannot answer this question based on the provided documents."
- Always cite document name and page number for each factual statement
- Explain naturally, clearly, and in a conversational tone
- Structure your response clearly with proper formatting
- Connect information logically and provide meaningful insights
- Use step-by-step reasoning internally, but deliver a cohesive final answer

USER QUESTION:
{query}

ANSWER:"""
            
            # Generate answer using LLM with role-based temperature
            answer = self.llm.generate(prompt, temperature=temperature)
            
            # Save to memory
            try:
                memory.save_context(
                    inputs={"input": query},
                    outputs={"output": answer}
                )
            except Exception as e:
                print(f"❌ Memory save error: {str(e)}")
            
            # Format sources
            sources = []
            for ctx in contexts:
                sources.append({
                    "text": ctx.get('text', '')[:200] + "..." if len(ctx.get('text', '')) > 200 else ctx.get('text', ''),
                    "source": ctx.get("document_name", ""),
                    "page": ctx.get("page_idx", 0),
                    "score": ctx.get("score", 0.0),
                    "document_id": ctx.get("document_id"),
                    "chunk_id": ctx.get("chunk_id"),
                    "global_chunk_id": ctx.get("global_chunk_id"),
                    "chunk_index": ctx.get("chunk_index"),
                    "section_hierarchy": ctx.get("section_hierarchy"),
                    "heading_context": ctx.get("heading_context"),
                    "char_count": ctx.get("char_count"),
                    "word_count": ctx.get("word_count")
                })
            
            print("✅ TRUE LangChain RAG completed")
            
            # Extract and update conversation context
            if conversation_id and user_id:
                self._update_conversation_context(conversation_id, user_id, query, answer, contexts)
            
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
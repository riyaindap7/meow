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
                    template="""You are VICTOR, an intelligent AI assistant for Indian education policy documents.

YOUR APPROACH:
- Think step-by-step and reason logically
- Use a natural, conversational tone (like ChatGPT)
- Stay completely truthful to provided documents
- Connect information intelligently across sources

YOUR RULES:
- Answer ONLY using information from provided documents and conversation history
- If documents don't contain the answer: "I cannot answer this based on the provided documents"
- Always cite sources: [Document: <name>, Page: <number>]
- Use conversation history to understand context and pronouns ("it", "this", "that")
- Think through the problem step-by-step
- Never invent or assume information not in documents

YOUR RESPONSE STYLE:
- Clear, concise, comprehensive
- Natural conversational flow
- Logical reasoning when needed
- Helpful and approachable

=== CONVERSATION HISTORY ===
{chat_history}

=== RETRIEVED DOCUMENTS ===
{context}

=== THINKING PROCESS ===
1. Understand the current question and conversation context
2. Review all relevant documents
3. Think step-by-step about the answer
4. Connect information logically
5. Respond conversationally with citations

=== CURRENT QUESTION ===
{input}

=== YOUR ANSWER ===
(Think it through, then respond naturally with source citations):"""
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
        print("🔧 Initializing FullLangChainRAG with TRUE Memory...")
        
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
        
        # ✅ Initialize conversation_manager (not used for chains, just for reference)
        self.conversation_manager = None
        try:
            from services.conversation_service import get_conversation_service
            self.conversation_manager = get_conversation_service()
            print("✅ Conversation service initialized")
        except Exception as e:
            print(f"⚠️ Conversation service not available: {str(e)}")
        
        # ✅ ADD: Initialize conversation manager
        from services.conversation_service import get_conversation_service
        self.conversation_manager = get_conversation_service()
        print("✅ Conversation manager initialized")
        
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
    
    def create_new_conversation(self, title: str, user_id: str, metadata: Dict = None) -> str:
        """Create a new conversation in MongoDB"""
        try:
            print(f"🆕 Creating new conversation: '{title}' for user {user_id}")
            
            if self.conversations_collection is None:
                print("⚠️ MongoDB not available, using fallback")
                import uuid
                return str(uuid.uuid4())
            
            # Use ConversationService for creating conversations
            from services.conversation_service import get_conversation_service
            conv_service = get_conversation_service()
            result = conv_service.create_conversation(
                user_id=user_id,
                title=title,
                metadata=metadata or {}
            )
            
            conversation_id = result.get("conversation_id")
            print(f"✅ Conversation created: {conversation_id}")
            return conversation_id
            
        except Exception as e:
            print(f"❌ Error creating conversation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for a user"""
        try:
            print(f"📋 Getting conversations for user: {user_id}")
            
            if self.conversations_collection is None:
                print("⚠️ MongoDB not available")
                return []
            
            # Use ConversationService for getting conversations
            from services.conversation_service import get_conversation_service
            conv_service = get_conversation_service()
            conversations = conv_service.get_user_conversations(user_id)
            
            print(f"✅ Found {len(conversations)} conversations")
            return conversations
            
        except Exception as e:
            print(f"❌ Error getting conversations: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
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
    
    def _enhance_query_with_context(self, query: str, context: dict) -> str:
        """Enhance vague queries with conversation context"""
        if not context:
            return query
        
        topics = context.get("topics", [])[:3]
        entities = context.get("main_entities", [])[:3]
        
        # Check if query uses vague terms or pronouns
        vague_patterns = ["it", "this", "that", "these", "those", "tell me", "what about", "explain", "elaborate"]
        query_lower = query.lower()
        is_vague = any(pattern in query_lower for pattern in vague_patterns)
        
        if is_vague and (topics or entities):
            context_terms = topics[:2] if topics else entities[:2]
            enhanced = f"{query} (in context of {', '.join(context_terms)})"
            print(f"🔍 Enhanced query: '{query}' → '{enhanced}'")
            return enhanced
        
        return query
    
    def _retrieve_documents(self, query: str, top_k: int, dense_weight: float, 
                           sparse_weight: float, method: str, 
                           conversation_context: dict = None,
                           filter_expr: str = None,
                           document_keyword: str = None):
        """Retrieve documents - filtered mode or normal mode"""
        
        print(f"\n🔍 DOCUMENT RETRIEVAL")
        print(f"   Query: {query}")
        print(f"   Method: {method}")
        print(f"   Top-K: {top_k}")
        print(f"   Filter: {filter_expr or 'None'}")
        
        try:
            # ✅ Always fetch 50 documents for reranking (regardless of requested top_k)
            retrieval_limit = 50
            
            results = self.milvus_client.search(
                query=query,
                top_k=retrieval_limit,
                method=method,
                filter_expr=filter_expr  # ✅ Metadata filter includes document_id
            )
            
            print(f"   📊 Retrieved: {len(results)} results")
            
            # ✅ REMOVED: No keyword filtering - metadata handles document_id
            
            # Deduplicate
            unique_results = self._deduplicate_results(results)
            print(f"   📊 After deduplication: {len(unique_results)}")
            
            if len(unique_results) == 0:
                print(f"   ❌ No documents found")
                return []
            
            # Cross-encoder reranking (replaces context-based reranking)
            print(f"   🔄 Applying cross-encoder reranking...")
            from services.reranker_service import get_reranker
            reranker = get_reranker()
            reranked_docs = reranker.rerank(
                query=query,
                documents=unique_results,
                top_k=15,  # Changed from 3 to 15
                min_k=3
            )
            
            # Take top-k (already done by reranker, but ensure consistency)
            final_results = reranked_docs[:top_k]
            
            print(f"   ✅ Final: {len(final_results)} documents")
            if final_results:
                avg_chars = sum(r.get('char_count', 0) for r in final_results) / len(final_results)
                doc_names = set(r.get('document_name', 'unknown') for r in final_results)
                print(f"   📊 Avg length: {avg_chars:.0f} chars")
                print(f"   📄 Documents: {', '.join(list(doc_names)[:3])}")
            
            return final_results
            
        except Exception as e:
            print(f"   ❌ Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _deduplicate_results(self, results: list) -> list:
        """Remove duplicate or very similar chunks"""
        if not results:
            return []
        
        unique_results = []
        seen_texts = set()
        
        for result in results:
            text = result.get('text', '').strip()
            
            if len(text) < 50:
                continue
            
            normalized = text.lower()[:200]
            
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_with_context(self, results: list, context: dict) -> list:
        """Rerank results based on conversation context"""
        topics = set(t.lower() for t in context.get("topics", []))
        entities = set(e.lower() for e in context.get("main_entities", []))
        
        for result in results:
            text = result.get("text", "").lower()
            original_score = result.get("score", 0)
            boost = 0.0
            
            # Boost for topic matches
            for topic in topics:
                if topic in text:
                    boost += 0.15
            
            # Boost for entity matches
            for entity in entities:
                if entity in text:
                    boost += 0.10
            
            result["original_score"] = original_score
            result["context_boost"] = boost
            result["score"] = original_score + boost
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        boosted_count = sum(1 for r in results if r.get("context_boost", 0) > 0)
        if boosted_count > 0:
            print(f"🎯 Context reranking: {boosted_count}/{len(results)} documents boosted")
        
        return results
    
    def _generate_answer(self, query: str, documents: list, temperature: float,
                        conversation_history: list = None, conversation_context: dict = None):
        """Generate answer using LLM with documents and conversation history"""
        if conversation_history is None:
            conversation_history = []
        """Generate answer with context-aware system prompt"""
        
        # Build system prompt with conversation context and temporal awareness
        system_prompt = """You are VICTOR, an intelligent AI assistant specializing in Indian education policies and government documents.

Your core capabilities:
- Answer questions using ONLY information from the provided documents
- Apply TEMPORAL ANALYSIS - consider dates, timelines, and chronological context in documents
- Use LOGICAL REASONING - think step-by-step, analyze cause-effect relationships
- Maintain a natural, conversational, and helpful tone (like ChatGPT)
- Be truthful - if documents don't contain the answer, clearly state: "I cannot answer this based on the provided documents"

Your analytical framework:
1. TEMPORAL CONTEXT
   - Identify and note publication dates, effective dates, amendment dates in documents
   - Recognize time-based patterns (before/after policy changes, evolution over time)
   - Compare information across different time periods
   - Highlight what changed when, and what remained constant
   - Use phrases like "As of [date]...", "Prior to [year]...", "Following [event]..."

2. LOGICAL ANALYSIS
   - Think through problems step-by-step
   - Identify cause-and-effect relationships
   - Connect related concepts across documents logically
   - Analyze implications and consequences
   - Recognize contradictions or complementary information
   - Build coherent arguments from evidence

3. INFORMATION SYNTHESIS
   - Cross-reference information from multiple documents
   - Identify patterns and relationships
   - Distinguish between facts, policies, and recommendations
   - Provide context for technical terms and acronyms
   - Connect specific details to broader policy goals

Your response requirements:
- Use conversation history to understand context, pronouns ("it", "this", "that"), and follow-up questions
- Provide specific citations: [Document: <name>, Page: <number>, Date: <if available>]
- Reorganize information logically for clarity
- Never invent or assume information not present in documents
- When temporal information exists, ALWAYS include it in your analysis

Your response style:
- Clear, concise, and comprehensive
- Natural conversational flow with temporal precision
- Logical reasoning made explicit when needed
- Decision support when relevant (pros/cons, implications, considerations)
- Analytical depth - examine information from multiple perspectives
- Cite sources with temporal context: [Document: <name>, Page: <number>, Published: <date>]"""

        if conversation_context:
            topics = conversation_context.get("topics", [])
            entities = conversation_context.get("main_entities", [])
            user_goal = conversation_context.get("user_goal", "")
            summary = conversation_context.get("summary", "")
            
            context_parts = []
            if summary:
                context_parts.append(f"📝 Previous discussion: {summary}")
            if user_goal:
                context_parts.append(f"🎯 User's goal: {user_goal}")
            if topics:
                context_parts.append(f"📚 Current topics: {', '.join(topics[:5])}")
            if entities:
                context_parts.append(f"🔑 Key entities: {', '.join(entities[:5])}")
            
            if context_parts:
                system_prompt += "\n\n=== CONVERSATION CONTEXT ===\n"
                system_prompt += "\n".join(context_parts)
                system_prompt += "\n" + "="*50
                print(f"💬 Added conversation context to prompt")
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n\n=== RECENT CONVERSATION ===\n"
            for msg in conversation_history[-6:]:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
                if len(content) > 150:
                    content = content[:150] + "..."
                history_text += f"{role}: {content}\n"
            history_text += "="*50
        
        # Format retrieved documents with emphasis on temporal data
        doc_context = "\n\n".join([
            f"--- Source {i+1} (Relevance: {doc.get('score', 0):.3f}) ---\n"
            f"Document: {doc.get('document_name', 'Unknown')}\n"
            f"Page: {doc.get('page_idx', 'N/A')}\n"
            f"Published Date: {doc.get('published_date', 'Date not specified')}\n"
            f"Section: {doc.get('section_hierarchy', 'N/A')}\n"
            f"Content: {doc.get('text', '')[:1200]}"
            for i, doc in enumerate(documents[:7])
        ])
        
        # Build final prompt with clear analytical instructions
        full_prompt = f"""{system_prompt}
{history_text}

=== RETRIEVED DOCUMENTS ===
{doc_context}

=== YOUR ANALYTICAL APPROACH ===

STEP 1: TEMPORAL ANALYSIS
- Scan all documents for dates, time periods, and temporal markers
- Note: publication dates, effective dates, amendment dates
- Identify: what changed over time? what's current vs. historical?
- Consider: chronological order of policies, evolution of concepts

STEP 2: LOGICAL ANALYSIS
- Break down the question into sub-components
- Identify what type of answer is needed (definition, comparison, timeline, impact, etc.)
- Map relevant information from documents to question components
- Identify cause-effect relationships, dependencies, implications
- Cross-reference information across sources for completeness

STEP 3: SYNTHESIS & VERIFICATION
- Combine information logically and chronologically
- Check for contradictions or gaps in evidence
- Ensure temporal accuracy (don't mix historical and current policies)
- Verify all claims are grounded in documents
- Prepare citations with temporal context where available

STEP 4: RESPONSE CONSTRUCTION
- Start with direct answer to the question
- Provide temporal context (when did this happen/change?)
- Explain the logic/reasoning behind policies or changes
- Include relevant dates and timeline information
- Add decision support if applicable (implications, considerations)
- Cite all sources with: [Document: <name>, Page: <number>, Date: <if known>]

=== CRITICAL RULES ===
✓ Use temporal context from documents (dates, periods, before/after)
✓ Think step-by-step and show logical reasoning
✓ Analyze from multiple perspectives before answering
✓ Use conversation history to understand pronouns and context
✓ Cite every factual claim with source
✗ Never invent dates, information, or details
✗ Don't mix information from different time periods without noting it
✗ Don't assume causation without evidence

=== CURRENT QUESTION ===
{query}

=== YOUR ANSWER ===
(Apply temporal analysis → logical reasoning → synthesis, then respond clearly with citations and temporal context):"""

        print(f"📤 Calling LLM with temporal & logical analysis prompt")
        print(f"   Prompt length: {len(full_prompt)} chars")
        print(f"   Documents with dates: {sum(1 for d in documents if d.get('published_date'))}")
        
        # Generate answer
        answer = self.llm.generate(full_prompt, temperature=temperature)
        
        print(f"📥 LLM response: {len(answer)} chars")
        print(f"📥 LLM response preview: {answer[:200]}...")
        
        return answer

    def ask(self, query: str, user_id: str = None, conversation_id: str = None,
            temperature: float = 0.1, top_k: int = 5, user: dict = None,
            dense_weight: float = 0.6, sparse_weight: float = 0.4, 
            method: str = "hybrid",
            conversation_context: dict = None,
            filter_expr: str = None,
            document_keyword: str = None):
        """Execute full RAG pipeline with conversation context awareness and filters"""
        
        print(f"\n🔵 LANGCHAIN RAG EXECUTION")
        print(f"   Query: {query}")
        print(f"   Filter: {filter_expr or 'None'}")
        print(f"   Keyword: {document_keyword or 'None'}")
        
        try:
            # STEP 1: Enhance query with context
            enhanced_query = self._enhance_query_with_context(query, conversation_context) if conversation_context else query
            
            # STEP 2: Retrieve documents
            print(f"\n🔍 RETRIEVING DOCUMENTS (method: {method}, top_k: {top_k})")
            documents = self._retrieve_documents(
                enhanced_query, 
                top_k, 
                dense_weight, 
                sparse_weight, 
                method,
                conversation_context,
                filter_expr,
                document_keyword
            )
            print(f"   Retrieved {len(documents)} documents")
            
            # STEP 3: Get conversation history from MongoDB
            conversation_history = []
            if conversation_id:
                from services.mongodb_service import mongodb_service
                conversation_history = mongodb_service.get_last_messages(conversation_id, limit=6)
                print(f"   Loaded {len(conversation_history)} history messages")
            else:
                # Create new conversation if needed
                if user_id:
                    from services.mongodb_service import mongodb_service
                    conversation_id = mongodb_service.create_conversation(user_id, title="New Chat")
                    print(f"   Created new conversation: {conversation_id}")
                else:
                    conversation_id = "temp_" + str(int(time.time()))
                    print(f"   Using temporary conversation: {conversation_id}")
            
            # STEP 4: Generate answer
            print(f"\n🤖 GENERATING ANSWER")
            answer = self._generate_answer(
                query=query,
                documents=documents,
                temperature=temperature,
                conversation_history=conversation_history,
                conversation_context=conversation_context
            )
            
            # STEP 5: Update context if needed
            if conversation_context:
                self._update_conversation_context(
                    conversation_id, 
                    query, 
                    answer, 
                    documents, 
                    conversation_context
                )
            
            # ✅ ALWAYS RETURN A VALID DICTIONARY
            result = {
                "answer": answer if answer else "I apologize, but I couldn't generate an answer.",
                "sources": documents if documents else [],
                "conversation_id": conversation_id,
                "model_used": self.model_name,
                "method": method
            }
            
            print(f"✅ RAG execution complete")
            print(f"   Answer length: {len(result['answer'])} chars")
            print(f"   Sources: {len(result['sources'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in ask(): {e}")
            import traceback
            traceback.print_exc()
            
            # ✅ RETURN ERROR RESULT INSTEAD OF None
            return {
                "answer": f"I encountered an error: {str(e)}. Please try again.",
                "sources": [],
                "conversation_id": conversation_id if conversation_id else "error",
                "model_used": self.model_name,
                "method": method
            }

# Global singleton instance
_langchain_rag_instance = None


def get_full_langchain_rag() -> 'FullLangChainRAG':
    """Get or create FullLangChainRAG singleton instance"""
    global _langchain_rag_instance
    
    if _langchain_rag_instance is None:
        print("🔧 Initializing FullLangChainRAG singleton...")
        _langchain_rag_instance = FullLangChainRAG()
        print("✅ FullLangChainRAG singleton ready")
    
    return _langchain_rag_instance


def reset_langchain_rag():
    """Reset the singleton instance (useful for testing)"""
    global _langchain_rag_instance
    _langchain_rag_instance = None
    print("♻️ FullLangChainRAG singleton reset")
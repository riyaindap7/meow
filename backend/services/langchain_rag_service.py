import os
from typing import List, Dict
from dotenv import load_dotenv
import httpx
import json
import time

load_dotenv()

class LangChainRAG:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "alibaba/tongyi-deepresearch-30b-a3b")
        self.site_url = os.getenv("SITE_URL", "http://localhost:3000")
        self.site_name = os.getenv("SITE_NAME", "VICTOR")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.base_url_alt = "https://openrouter.io/api/v1/chat/completions"
        
        # Debug: Check if API key exists
        if not self.api_key:
            print("⚠️ WARNING: OPENROUTER_API_KEY not set!")
        else:
            print(f"✅ OPENROUTER_API_KEY found (length: {len(self.api_key)})")
        
        # Create httpx client with longer timeouts
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0, connect=30.0),  # 30 second timeout
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            verify=True  # Enable SSL verification
        )
        
        print(f"✅ LangChain RAG initialized with model: {self.model_name}")
        print(f"   Base URL: {self.base_url}")
        print(f"   Site: {self.site_name}")
    
    def create_system_prompt(self) -> str:
        """Create system prompt for RAG with context awareness"""
        return """You are an intelligent assistant that answers questions based on provided documents and conversation context.

Instructions:
- Answer ONLY using information from the provided context and previous conversation
- If information is not available, say "I cannot answer this based on provided documents"
- Always cite source documents and page numbers when referencing them
- Be aware of the conversation history and refer to previous points when relevant
- Be accurate, concise, and maintain conversation continuity
- Match the language style of the question"""
    
    def format_context(self, contexts: List[Dict]) -> str:
        """Format search results as context"""
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            formatted.append(
                f"[Source {i}]\n"
                f"Document: {ctx.get('document_id', 'Unknown')}\n"
                f"Page: {ctx.get('page_idx', 'N/A')}\n"
                f"Section: {ctx.get('section_hierarchy', 'N/A')}\n"
                f"Content: {ctx.get('text', '')}\n"
            )
        return "\n---\n".join(formatted)
    
    def format_conversation_history(self, messages: List[Dict], max_messages: int = 10) -> str:
        """Format conversation history for context maintenance"""
        if not messages:
            return ""
        
        # Get last N messages (excluding current query)
        recent_messages = messages[-max_messages:] if len(messages) > 1 else []
        
        if not recent_messages:
            return ""
        
        formatted = ["=== Conversation History ==="]
        for msg in recent_messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def generate_answer(self, query: str, contexts: List[Dict], temperature: float = 0.1, conversation_history: List[Dict] = None, max_retries: int = 3) -> str:
        """Generate answer using OpenRouter via httpx with conversation context and retry logic"""
        try:
            print(f"🔵 LangChain generating answer for: {query}")
            
            # Format context
            context_text = self.format_context(contexts)
            
            # Format conversation history if provided
            history_text = ""
            if conversation_history:
                history_text = self.format_conversation_history(conversation_history)
                if history_text:
                    print(f"📜 Including conversation history with {len(conversation_history)} messages")
            
            # Create the prompt
            system_prompt = self.create_system_prompt()
            
            # Build user message with conversation context
            user_content_parts = []
            
            if history_text:
                user_content_parts.append(history_text)
            
            user_content_parts.append(f"""DOCUMENT CONTEXT:
{context_text}

CURRENT QUESTION: {query}

ANSWER:""")
            
            user_content = "\n\n".join(user_content_parts)
            
            # Prepare payload for OpenRouter
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": temperature,
                "max_tokens": 2000
            }
            
            print(f"📤 Sending to OpenRouter...")
            print(f"   Model: {self.model_name}")
            print(f"   Endpoint: {self.base_url}")
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    print(f"   Attempt {attempt + 1}/{max_retries}...")
                    
                    # Send request with timeout
                    response = self.client.post(self.base_url, json=payload)
                    
                    print(f"🔵 LLM status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "choices" in result and len(result["choices"]) > 0:
                            answer = result["choices"][0]["message"]["content"]
                            print(f"✅ Answer generated successfully")
                            return answer
                        else:
                            print(f"❌ Unexpected response structure: {result}")
                            raise Exception(f"Unexpected response: {result}")
                    
                    elif response.status_code == 405:
                        # Try alternative endpoint
                        print(f"   Trying alternative endpoint...")
                        response = self.client.post(self.base_url_alt, json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            answer = result["choices"][0]["message"]["content"]
                            print(f"✅ Answer generated successfully (from alt endpoint)")
                            return answer
                    
                    else:
                        print(f"❌ OpenRouter Error ({response.status_code}): {response.text[:200]}")
                        
                        # If not the last attempt, retry
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            print(f"   Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise Exception(f"OpenRouter Error ({response.status_code}): {response.text[:200]}")
                
                except httpx.TimeoutException as e:
                    print(f"⏱️ Timeout on attempt {attempt + 1}: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"   Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"Request timed out after {max_retries} attempts")
                
                except Exception as e:
                    print(f"❌ Error on attempt {attempt + 1}: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"   Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
        
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            raise
    
    def __del__(self):
        """Close the httpx client"""
        if hasattr(self, 'client'):
            self.client.close()

# Global instance
_rag_manager = None

def get_langchain_rag() -> LangChainRAG:
    """Get or create LangChain RAG singleton"""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = LangChainRAG()
    return _rag_manager
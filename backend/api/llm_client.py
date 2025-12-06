import httpx
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Load .env from project root (parent of api folder)
env_path = Path(__file__).parent.parent / ".env"
print(f"🔍 Loading .env from: {env_path}")
print(f"✓ .env exists: {env_path.exists()}")

# Load the .env file
load_dotenv(env_path, override=True)

# Debug: print what's in the .env file
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if 'LLM_MODEL' in line:
                print(f"📄 Found in .env: {line.strip()}")

# Debug: check environment variable
llm_model_env = os.getenv("LLM_MODEL")
print(f"🔍 os.getenv('LLM_MODEL'): {llm_model_env}")

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv("LLM_MODEL")
        self.site_url = os.getenv("SITE_URL", "http://localhost:3000")
        self.site_name = os.getenv("SITE_NAME", "VICTOR")
        
        # Set headers for OpenRouter API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
        
        print(f"🚀 LLM Client Initialized with model: {self.model}")
        print(f"📍 OpenRouter Base URL: {self.base_url}")
        
    def create_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Create RAG prompt with retrieved context"""
        
        # Build context string
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {ctx['source']} (Page {ctx['page']})\n"
                f"Content: {ctx['text']}\n"
            )
        
        context_str = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based STRICTLY on the provided context.

Context Documents:
{context_str}

Instructions:
- You will construct the answer and present it with citations to the source documents.
- Answer the user's question using ONLY the information from the context above
- If the answer is not in the context, say "I cannot answer this question based on the provided documents"
- Always cite the source document and page number when answering
- Be accurate
- Do not add information that is not in the context
- Try to understand the language-style of the question and answer in the same language-style.

User Question: {query}

Answer:"""
        
        return prompt
    
    async def generate_answer(
        self, 
        query: str, 
        contexts: List[Dict], 
        temperature: float = 0.1
    ) -> str:
        """Generate answer using OpenRouter"""
        try:
            import json
            
            # Format contexts with all VictorText fields
            context_text = "\n\n".join([
                f"[Document: {ctx.get('document_name', 'Unknown')}] "
                f"[Page: {ctx.get('page_idx', 'N/A')}] "
                f"[Section: {ctx.get('section_hierarchy', 'N/A')}]\n"
                f"{ctx.get('text', '')}"
                for ctx in contexts
            ])
            
            # Create the prompt
            prompt = f"""Based on the following context from multiple documents, answer the user's question accurately and concisely.

CONTEXT:
{context_text}

INSTRUCTIONS:
- Answer using ONLY information from the context above
- If the answer is not in the context, say "I cannot answer this question based on the provided documents"
- Always cite the source document and page number when answering
- Be accurate and do not add information not in the context
- Match the language style of the question

USER QUESTION: {query}

ANSWER:"""

            print(f"📝 Prompt created for LLM")
            
            # Call OpenRouter API using httpx
            print(f"🔵 Calling OpenRouter API...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 2000
                    }
                )
            
            print(f"🔵 OpenRouter response status: {response.status_code}")
            
            if response.status_code != 200:
                error_detail = response.text
                print(f"❌ OpenRouter API Error ({response.status_code}): {error_detail}")
                raise Exception(f"OpenRouter API Error ({response.status_code}): {error_detail}")
            
            result = response.json()
            print(f"🟢 Got result from OpenRouter")
            print(f"📊 Response keys: {result.keys()}")
            
            # Extract answer from response
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    answer = choice["message"]["content"]
                elif "text" in choice:
                    answer = choice["text"]
                else:
                    print(f"❌ Unexpected choice format: {choice.keys()}")
                    raise Exception(f"Unexpected response format: {result}")
                
                print(f"✅ Answer generated successfully")
                return answer
            else:
                print(f"❌ Unexpected OpenRouter response structure: {result}")
                raise Exception(f"Unexpected response structure from OpenRouter: {json.dumps(result)}")
        
        except Exception as e:
            print(f"❌ Error calling OpenRouter: {str(e)}")
            raise

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        """
        Simple synchronous generation for query decomposition
        (Used by self-query retriever)
        """
        try:
            import httpx
            
            # Synchronous HTTP call
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
            
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"OpenRouter API Error ({response.status_code}): {error_detail}")
            
            result = response.json()
            
            # Extract content from response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Unexpected response structure: {result}")
        
        except Exception as e:
            print(f"⚠️ LLM generate() error: {e}")
            raise

# Global instance
llm_client = None

def get_llm_client() -> OpenRouterClient:
    """Get or create OpenRouter client singleton"""
    global llm_client
    if llm_client is None:
        llm_client = OpenRouterClient()
    return llm_client
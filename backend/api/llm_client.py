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
        """Create enhanced RAG prompt with VictorText2 metadata"""
        if not contexts:
            context_str = "No relevant documents found."
        else:
            context_parts = []
            for i, ctx in enumerate(contexts, 1):
                context_desc = f"[Source {i}] Document: {ctx.get('document_name', 'Unknown')}"
                if ctx.get('page_idx'):
                    context_desc += f" (Page {ctx['page_idx']})"
                if ctx.get('ministry'):
                    context_desc += f" | Ministry: {ctx['ministry']}"
                if ctx.get('date'):
                    context_desc += f" | Date: {ctx['date']}"
                context_desc += f"\n{ctx.get('text', '')}"
                context_parts.append(context_desc)
            
            context_str = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are VICTOR, a helpful, intelligent AI assistant specializing in government document analysis. Answer the user's question using the information found in the provided context from multiple documents.

CONTEXT:
{context_str}

INSTRUCTIONS:
- Use the context as your primary reference while applying deep analytical reasoning
- You may reason and make logical connections between information from different documents
- When reasoning across documents, clearly indicate your logical inference process
- Always cite document name, page number, and section when referencing information
- Include relevant metadata (ministry, date, document type) when it provides valuable context
- If the context doesn't contain sufficient information, state: "I cannot fully answer this question based on the provided documents."
- Explain naturally, clearly, and in a professional yet conversational tone
- Use step-by-step reasoning internally, but provide a cohesive, well-structured final answer

USER QUESTION:
{query}

ANSWER:"""
        return prompt

    async def generate_answer(
        self, 
        query: str, 
        contexts: List[Dict], 
        temperature: float = 0.1
    ) -> str:
        """Generate answer using OpenRouter with enhanced VictorText2 context"""
        if not contexts:
            context_text = "No relevant documents found."
        else:
            context_parts = []
            for i, ctx in enumerate(contexts, 1):
                context_desc = f"[Source {i}] Document: {ctx.get('document_name', 'Unknown')}"
                if ctx.get('page_idx'):
                    context_desc += f" (Page {ctx['page_idx']})"
                if ctx.get('ministry'):
                    context_desc += f" | Ministry: {ctx['ministry']}"
                if ctx.get('date'):
                    context_desc += f" | Date: {ctx['date']}"
                context_desc += f"\n{ctx.get('text', '')}"
                context_parts.append(context_desc)
            
            context_text = "\n\n---\n\n".join(context_parts)
            
            # Create the enhanced prompt
            prompt = f"""You are VICTOR, a helpful, intelligent AI assistant specializing in government document analysis. Answer the user's question using the information found in the provided context from multiple documents.

CONTEXT:
{context_text}

INSTRUCTIONS:
- Use the context as your primary reference while applying deep analytical reasoning
- You may reason and connect information across documents when logical
- Always cite document name, page number, and section/heading when referencing information
- Include relevant metadata (ministry, date, document type) when it adds valuable context
- If you cannot answer based on the provided documents, clearly state this
- Explain naturally, clearly, and in a professional yet conversational tone
- Connect information logically and provide meaningful insights
- Use step-by-step reasoning internally, but deliver a cohesive final answer

USER QUESTION:
{query}

ANSWER:"""

            print(f"📝 Enhanced prompt created with {len(contexts)} contexts")
            
            try:
                # Call OpenRouter API
                print(f"🔵 Calling OpenRouter API...")
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.base_url,
                        headers=self.headers,
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                            "max_tokens": 2500  # Increased for richer responses
                        }
                    )
                
                print(f"🔵 OpenRouter response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_detail = response.text
                    print(f"❌ OpenRouter API Error ({response.status_code}): {error_detail}")
                    raise Exception(f"OpenRouter API Error ({response.status_code}): {error_detail}")
                
                result = response.json()
                print(f"🟢 Got enhanced result from OpenRouter")
                
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
                    
                    print(f"✅ Enhanced answer generated successfully")
                    return answer
                else:
                    print(f"❌ Unexpected OpenRouter response structure: {result}")
                    raise Exception(f"Unexpected response structure from OpenRouter: {result}")
            
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
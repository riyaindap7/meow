from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    collection_name: str
    query: str
    top_k: int = 5

@router.post("/")
async def chat(request: ChatRequest):
    """RAG Q&A endpoint"""
    # Basic demo response without LLM
    return {
        "query": request.query,
        "answer": "This is a demo answer from the Victor API.",
        "sources": [
            {"id": "1", "text": "Source document 1", "score": 0.95}
        ]
    }
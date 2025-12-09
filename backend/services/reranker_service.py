"""
Cross-Encoder Reranker Service
Uses a local cross-encoder model to rerank retrieved documents
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class CrossEncoderReranker:
    """
    Local cross-encoder based reranker for document reranking
    Uses sentence-transformers cross-encoder models
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the cross-encoder reranker
        
        Args:
            model_name: Name of the cross-encoder model to use
                       Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
                       Other options:
                       - "cross-encoder/ms-marco-MiniLM-L-12-v2" (better but slower)
                       - "BAAI/bge-reranker-base" (multilingual)
                       - "BAAI/bge-reranker-large" (best quality but slowest)
        """
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", 
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        print(f"🔧 Loading cross-encoder reranker: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            print(f"✅ Cross-encoder reranker loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load cross-encoder: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 15,
        min_k: int = 3,
        score_threshold: float = None
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'text' field
            top_k: Target number of documents to return (default: 15)
            min_k: Minimum number to return if fewer than top_k available (default: 3)
            score_threshold: Optional minimum score threshold
        
        Returns:
            List of reranked documents (top_k best matches, or min_k if fewer available)
        """
        if not documents:
            print("⚠️ No documents to rerank")
            return []
        
        print(f"\n🔄 CROSS-ENCODER RERANKING")
        print(f"   Input documents: {len(documents)}")
        print(f"   Target top-k: {top_k}")
        print(f"   Minimum fallback: {min_k}")
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                text = doc.get('text', '')
                # Truncate very long texts for efficiency
                if len(text) > 1000:
                    text = text[:1000] + "..."
                pairs.append([query, text])
            
            # Get cross-encoder scores
            print(f"   Computing cross-encoder scores...")
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            scored_docs = []
            for doc, score in zip(documents, scores):
                # Store original score if it exists
                if 'score' in doc:
                    doc['original_score'] = doc['score']
                
                # Set new cross-encoder score
                doc['score'] = float(score)
                doc['reranker_score'] = float(score)
                
                # Apply threshold if specified
                if score_threshold is None or score >= score_threshold:
                    scored_docs.append(doc)
            
            # Sort by cross-encoder score (descending)
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Determine how many to return
            num_available = len(scored_docs)
            if num_available >= top_k:
                num_to_return = top_k
            elif num_available >= min_k:
                num_to_return = num_available  # Return all available
            else:
                num_to_return = min(min_k, num_available)  # Return min_k or all if less
            
            # Take the determined number
            reranked_docs = scored_docs[:num_to_return]
            
            print(f"   ✅ Reranked: {len(scored_docs)} documents passed threshold")
            print(f"   📊 Returning {len(reranked_docs)} documents (target: {top_k}, min: {min_k})")
            
            if reranked_docs:
                score_range = f"{reranked_docs[-1]['score']:.3f} to {reranked_docs[0]['score']:.3f}"
                print(f"   📈 Score range: {score_range}")
            
            return reranked_docs
            
        except Exception as e:
            print(f"❌ Reranking error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original documents
            fallback_count = min(min_k, len(documents))
            return documents[:fallback_count]
    
    def rerank_with_details(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 15,
        min_k: int = 3
    ) -> Tuple[List[Dict], Dict]:
        """
        Rerank documents and return detailed statistics
        
        Returns:
            Tuple of (reranked_documents, statistics_dict)
        """
        reranked = self.rerank(query, documents, top_k, min_k)
        
        stats = {
            "input_count": len(documents),
            "output_count": len(reranked),
            "target_k": top_k,
            "min_k": min_k,
            "model_name": self.model_name,
            "top_score": reranked[0]['score'] if reranked else 0.0,
            "bottom_score": reranked[-1]['score'] if reranked else 0.0,
            "avg_score": sum(d['score'] for d in reranked) / len(reranked) if reranked else 0.0
        }
        
        return reranked, stats


# Singleton instance
_reranker_instance = None


def get_reranker() -> CrossEncoderReranker:
    """Get or create singleton reranker instance"""
    global _reranker_instance
    
    if _reranker_instance is None:
        print("🔧 Initializing CrossEncoderReranker singleton...")
        _reranker_instance = CrossEncoderReranker()
        print("✅ CrossEncoderReranker singleton ready")
    
    return _reranker_instance


def reset_reranker():
    """Reset singleton instance (useful for testing)"""
    global _reranker_instance
    _reranker_instance = None
    print("♻️ CrossEncoderReranker singleton reset")

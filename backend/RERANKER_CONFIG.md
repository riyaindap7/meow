# Cross-Encoder Reranker Configuration

## Overview
The system now uses a local cross-encoder model for document reranking to improve retrieval accuracy.

## How It Works
1. **Initial Retrieval**: Fetch 50 documents using hybrid search (RRF-based dense + sparse)
2. **Deduplication**: Remove duplicate chunks
3. **Cross-Encoder Reranking**: Rerank all documents using a cross-encoder model
4. **Final Selection**: Return top 10 (or top_k) best-matching documents

## Configuration

### Environment Variable (Optional)
Add to your `.env` file to customize the reranker model:

```bash
# Cross-Encoder Reranker Model
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Available Models

#### Fast & Lightweight (Default)
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```
- **Speed**: ⚡⚡⚡ Very Fast
- **Quality**: ⭐⭐⭐ Good
- **Size**: ~80MB
- **Best for**: Production with high throughput needs

#### Balanced Performance
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```
- **Speed**: ⚡⚡ Fast
- **Quality**: ⭐⭐⭐⭐ Better
- **Size**: ~130MB
- **Best for**: Balance between speed and accuracy

#### Multilingual Support
```bash
RERANKER_MODEL=BAAI/bge-reranker-base
```
- **Speed**: ⚡⚡ Fast
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Size**: ~280MB
- **Best for**: Multilingual documents
- **Supports**: 100+ languages

#### Highest Quality
```bash
RERANKER_MODEL=BAAI/bge-reranker-large
```
- **Speed**: ⚡ Slower
- **Quality**: ⭐⭐⭐⭐⭐ Best
- **Size**: ~560MB
- **Best for**: Maximum accuracy when latency is not critical

## Performance Characteristics

### Default Configuration
- **Initial Retrieval**: 50 documents
- **Final Output**: 10 documents (configurable via `top_k` parameter)
- **Processing Time**: ~100-300ms per query (depends on model choice)

### Retrieval Flow
```
Query → Hybrid Search (50 docs) → Deduplication → Cross-Encoder Reranking → Top 10 docs
```

## Benefits Over RRF Reranking

1. **Semantic Understanding**: Cross-encoders jointly encode query and document
2. **Better Accuracy**: Captures complex relationships between query and documents
3. **No Manual Tuning**: No need to tune weights or fusion parameters
4. **Relevance Scoring**: More accurate relevance scores

## Code Changes

### Before (RRF-based context reranking)
```python
# Old: Context-based boost reranking
if conversation_context and unique_results:
    unique_results = self._rerank_with_context(unique_results, conversation_context)
```

### After (Cross-encoder reranking)
```python
# New: Cross-encoder reranking
from services.reranker_service import get_reranker
reranker = get_reranker()
unique_results = reranker.rerank(
    query=query,
    documents=unique_results,
    top_k=top_k
)
```

## Testing

### Test the Reranker Service
```python
from services.reranker_service import get_reranker

# Initialize reranker
reranker = get_reranker()

# Test data
query = "What is the policy on remote work?"
documents = [
    {"text": "Remote work policy allows 3 days per week...", "score": 0.75},
    {"text": "Office space allocation guidelines...", "score": 0.73},
    {"text": "Work from home procedures and requirements...", "score": 0.71},
]

# Rerank
reranked = reranker.rerank(query, documents, top_k=2)

# Check results
for i, doc in enumerate(reranked):
    print(f"{i+1}. Score: {doc['score']:.3f} - {doc['text'][:50]}...")
```

### Test via API
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "What are the guidelines for remote work?",
    "top_k": 10,
    "method": "hybrid"
  }'
```

## Monitoring

The reranker logs detailed information:
```
🔄 CROSS-ENCODER RERANKING
   Input documents: 50
   Target top-k: 10
   Computing cross-encoder scores...
   ✅ Reranked: 50 documents passed threshold
   📊 Returning top 10 documents
   📈 Score range: 0.125 to 0.876
```

## Troubleshooting

### Model Download Issues
If the model fails to download, manually download it:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

### Out of Memory
If you encounter memory issues, switch to a smaller model:
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Slow Performance
For faster processing:
1. Use the L-6 model instead of L-12
2. Reduce the number of initial documents (currently 50)
3. Enable GPU if available

## Advanced Configuration

### Custom Score Threshold
Modify `reranker_service.py` to add a score threshold:
```python
reranked = reranker.rerank(
    query=query,
    documents=unique_results,
    top_k=top_k,
    score_threshold=0.3  # Only return docs with score > 0.3
)
```

### Adjust Initial Retrieval Count
In `full_langchain_service.py`, change the retrieval limit:
```python
# Fetch more or fewer documents for reranking
retrieval_limit = 100  # Default is 50
```

## Related Files
- `backend/services/reranker_service.py` - Reranker implementation
- `backend/services/full_langchain_service.py` - RAG pipeline integration
- `backend/api/milvus_client.py` - Initial RRF-based hybrid search

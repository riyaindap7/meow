# Quick Reference: Cross-Encoder Reranker

## What Changed?

### Before
- Fetched `top_k * 2` or `top_k * 3` documents
- Used RRF (Reciprocal Rank Fusion) for initial ranking
- Optional context-based boost reranking
- Returned top_k results

### After
- **Always fetches 50 documents** regardless of top_k
- Uses RRF for initial ranking (same as before)
- **Uses cross-encoder neural reranking** (NEW!)
- **Returns top 10 documents** (or custom top_k)

## Usage

### No Code Changes Required!
The reranker is automatically integrated into your existing RAG pipeline.

### Optional: Customize Model
Add to `.env`:
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Test It
```bash
cd backend
python test_reranker.py
```

## Models Quick Guide

| Model | Speed | Quality | Size | Use Case |
|-------|-------|---------|------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ⚡⚡⚡ | ⭐⭐⭐ | 80MB | **Default** - Production |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ⚡⚡ | ⭐⭐⭐⭐ | 130MB | Balanced |
| `BAAI/bge-reranker-base` | ⚡⚡ | ⭐⭐⭐⭐ | 280MB | Multilingual |
| `BAAI/bge-reranker-large` | ⚡ | ⭐⭐⭐⭐⭐ | 560MB | Best quality |

## Monitoring

Watch for these logs in your console:
```
🔍 DOCUMENT RETRIEVAL
   Query: What are the remote work policies?
   Method: hybrid
   Top-K: 10
   📊 Retrieved: 50 results
   📊 After deduplication: 48
   🔄 Applying cross-encoder reranking...

🔄 CROSS-ENCODER RERANKING
   Input documents: 48
   Target top-k: 10
   Computing cross-encoder scores...
   ✅ Reranked: 48 documents passed threshold
   📊 Returning top 10 documents
   📈 Score range: 0.125 to 0.876

   ✅ Final: 10 documents
```

## Benefits

✅ **Better Accuracy** - Neural semantic matching  
✅ **Consistent Quality** - Always evaluates 50 docs  
✅ **Easy to Use** - No code changes needed  
✅ **Flexible** - Swap models via env variable  
✅ **Local** - No external API calls  

## Files Created/Modified

1. ✅ `backend/services/reranker_service.py` - Reranker implementation
2. ✅ `backend/services/full_langchain_service.py` - Integration
3. ✅ `backend/test_reranker.py` - Test script
4. ✅ `backend/RERANKER_CONFIG.md` - Full documentation
5. ✅ `IMPLEMENTATION_SUMMARY.md` - Change summary

## Troubleshooting

**Q: Model not downloading?**  
A: Check internet connection. Model downloads automatically on first use.

**Q: Out of memory?**  
A: Use smaller model: `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`

**Q: Too slow?**  
A: Already using fastest model by default. Consider GPU if available.

**Q: Need multilingual?**  
A: Use `RERANKER_MODEL=BAAI/bge-reranker-base`

## That's It!

Your RAG system now uses state-of-the-art cross-encoder reranking! 🎉

# Cross-Encoder Reranker Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│                    "remote work policy"                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                          │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐ │
│  │  Dense Embedding     │  │    Sparse Embedding             │ │
│  │  (BAAI/bge-m3)       │  │    (BAAI/bge-m3)                │ │
│  │  [768 dimensions]    │  │    {token_id: weight}           │ │
│  └──────────────────────┘  └─────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID SEARCH (RRF)                           │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐ │
│  │  Dense Search        │  │    Sparse Search                │ │
│  │  (Vector Similarity) │  │    (Lexical Matching)           │ │
│  │  Top 50 results      │  │    Top 50 results               │ │
│  └──────────────────────┘  └─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │   RRF Fusion    │                          │
│                    │   (k=60)        │                          │
│                    └─────────────────┘                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVE 50 DOCUMENTS                         │
│                                                                  │
│  Doc 1: "Office dress code..." (score: 0.82)                    │
│  Doc 2: "Remote work policy..." (score: 0.79)                   │
│  Doc 3: "Government employees..." (score: 0.76)                 │
│  ...                                                             │
│  Doc 50: "Parking permits..." (score: 0.35)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DEDUPLICATION                               │
│  Remove duplicate/similar chunks based on text similarity        │
│  50 docs → ~45-48 unique docs                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           ⭐ CROSS-ENCODER RERANKING (NEW!) ⭐                   │
│                                                                  │
│  Model: cross-encoder/ms-marco-MiniLM-L-6-v2                    │
│                                                                  │
│  For each document:                                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Input: [Query, Document] pair                         │    │
│  │  ↓                                                      │    │
│  │  Cross-Encoder Neural Network                          │    │
│  │  (Jointly encodes query + document)                    │    │
│  │  ↓                                                      │    │
│  │  Output: Relevance Score (0.0 to 1.0)                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Process all ~45-48 documents in batch                          │
│  Compute cross-encoder scores for each                          │
│  Sort by new relevance scores                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOP 10 DOCUMENTS                              │
│                                                                  │
│  Doc 2: "Remote work policy..." (reranker: 0.876, orig: 0.79)   │
│  Doc 3: "Government employees..." (reranker: 0.754, orig: 0.76) │
│  Doc 5: "Work from home..." (reranker: 0.689, orig: 0.71)       │
│  Doc 7: "Telecommuting regs..." (reranker: 0.623, orig: 0.65)   │
│  ...                                                             │
│  Doc 4: "Annual leave policy..." (reranker: 0.234, orig: 0.74)  │
│                                                                  │
│  (Note: Original ranking changed based on semantic relevance!)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ANSWER GENERATION                         │
│                                                                  │
│  Context: Top 10 reranked documents                              │
│  Model: OpenRouter LLM                                           │
│  Output: Natural language answer with citations                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. RRF Fusion (Existing)
- **Purpose**: Combine dense + sparse search results
- **Method**: Reciprocal Rank Fusion
- **Output**: 50 candidate documents

### 2. Cross-Encoder Reranker (NEW!)
- **Purpose**: Deep semantic reranking
- **Input**: 50 documents + query
- **Method**: Neural cross-encoder model
- **Output**: Top 10 most relevant documents

## Score Comparison

### Before Reranking (RRF Scores)
```
Doc 1: 0.82 ← High RRF score (but maybe not most relevant)
Doc 2: 0.79
Doc 3: 0.76
Doc 4: 0.74
Doc 5: 0.71
```

### After Reranking (Cross-Encoder Scores)
```
Doc 2: 0.876 ← Most semantically relevant to query!
Doc 3: 0.754
Doc 5: 0.689
Doc 7: 0.623
Doc 1: 0.234 ← Demoted (less relevant despite high RRF score)
```

## Why Cross-Encoder is Better

### Traditional Vector Search
```
Query Embedding → Compare → Document Embedding
    [768D]          cosine      [768D]
                   similarity
```
**Problem**: Query and document encoded separately, limited interaction

### Cross-Encoder Reranking
```
[Query + Document] → Cross-Encoder → Relevance Score
    Combined Input      Neural Net      (0.0 to 1.0)
```
**Advantage**: Joint encoding captures complex semantic relationships

## Performance Characteristics

```
Pipeline Stage              | Time (ms) | Documents
----------------------------+-----------+-----------
Embedding Generation        |    10-20  |     -
Hybrid Search (RRF)         |    30-50  |    50
Deduplication               |     5-10  |    45-48
Cross-Encoder Reranking     |   100-300 |    45-48
Final Selection             |       <1  |    10
----------------------------+-----------+-----------
Total Retrieval Time        |   145-380 |    10
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Initial Retrieval** | top_k × 2-3 | Always 50 |
| **Reranking Method** | Context boost | Neural cross-encoder |
| **Semantic Understanding** | Limited | Deep |
| **Final Documents** | top_k | top_k (default 10) |
| **Accuracy** | Good | Excellent |
| **Latency** | 50-100ms | 145-380ms |

## Code Location

```
backend/
├── services/
│   ├── reranker_service.py          ← NEW! Cross-encoder implementation
│   └── full_langchain_service.py    ← MODIFIED! Integration point
├── test_reranker.py                 ← NEW! Test script
├── RERANKER_CONFIG.md               ← NEW! Configuration guide
└── requirements.txt                 ← No changes (already has sentence-transformers)
```

## Configuration

Simply add to `.env` (optional):
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Success Criteria

✅ Fetches 50 documents regardless of top_k  
✅ Uses local cross-encoder model (no API calls)  
✅ Returns top 10 best-matched documents  
✅ Replaces simple context-based reranking  
✅ Maintains backward compatibility  
✅ Provides detailed logging  
✅ Zero configuration required (works out of the box)  

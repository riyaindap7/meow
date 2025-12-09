"""
Test script for Cross-Encoder Reranker
Run this to verify the reranker is working correctly
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.reranker_service import get_reranker


def test_reranker():
    """Test the cross-encoder reranker"""
    print("=" * 60)
    print("Testing Cross-Encoder Reranker")
    print("=" * 60)
    
    # Initialize reranker
    print("\n1. Initializing reranker...")
    reranker = get_reranker()
    print(f"   Model: {reranker.model_name}")
    
    # Test query
    query = "What are the remote work policies for government employees?"
    
    # Sample documents (simulating retrieval results)
    documents = [
        {
            "text": "The office dress code requires formal attire on weekdays. Business casual is acceptable on Fridays.",
            "score": 0.82,
            "document_name": "HR_Guidelines.pdf",
            "page": 5
        },
        {
            "text": "Remote work policy: Employees may work from home up to 3 days per week with manager approval. Must maintain availability during core hours 10 AM - 3 PM.",
            "score": 0.79,
            "document_name": "Remote_Work_Policy.pdf",
            "page": 2
        },
        {
            "text": "Government employees are eligible for remote work arrangements subject to departmental approval. Positions must be suitable for remote execution.",
            "score": 0.76,
            "document_name": "Government_Employment.pdf",
            "page": 12
        },
        {
            "text": "Annual leave policy: Employees accrue 15 days of paid leave per year. Unused leave may be carried forward.",
            "score": 0.74,
            "document_name": "Leave_Policy.pdf",
            "page": 3
        },
        {
            "text": "Work from home guidelines: Ensure secure VPN connection, maintain confidentiality, and attend all virtual meetings.",
            "score": 0.71,
            "document_name": "WFH_Guidelines.pdf",
            "page": 1
        },
        {
            "text": "Parking permits are available for employees who commute to the office. Apply through the facilities department.",
            "score": 0.68,
            "document_name": "Facilities.pdf",
            "page": 8
        },
        {
            "text": "Government regulations on telecommuting: Agencies may establish remote work programs for eligible positions following federal guidelines.",
            "score": 0.65,
            "document_name": "Federal_Regulations.pdf",
            "page": 45
        },
        {
            "text": "Employee benefits include health insurance, retirement plans, and professional development opportunities.",
            "score": 0.62,
            "document_name": "Benefits_Guide.pdf",
            "page": 1
        },
    ]
    
    print(f"\n2. Test Query: '{query}'")
    print(f"\n3. Initial Documents: {len(documents)}")
    print("\nOriginal Ranking (by initial retrieval score):")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. Score: {doc['score']:.3f} - {doc['text'][:60]}...")
    
    # Rerank documents
    print(f"\n4. Reranking with cross-encoder...")
    reranked_docs = reranker.rerank(
        query=query,
        documents=documents.copy(),
        top_k=5
    )
    
    print(f"\n5. Reranked Results (Top 5):")
    print("\nNew Ranking (by cross-encoder score):")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"   {i}. Score: {doc['score']:.3f} (was {doc.get('original_score', 'N/A'):.3f}) - {doc['text'][:60]}...")
    
    # Analysis
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    
    if len(reranked_docs) > 0:
        print(f"✅ Reranking successful!")
        print(f"   Top score: {reranked_docs[0]['score']:.3f}")
        print(f"   Bottom score: {reranked_docs[-1]['score']:.3f}")
        print(f"   Score range: {reranked_docs[0]['score'] - reranked_docs[-1]['score']:.3f}")
        
        # Check if remote work docs are ranked higher
        remote_keywords = ['remote', 'work from home', 'telecommuting', 'wfh']
        top_doc = reranked_docs[0]['text'].lower()
        is_relevant = any(keyword in top_doc for keyword in remote_keywords)
        
        if is_relevant:
            print(f"   ✅ Top result is highly relevant to query!")
        else:
            print(f"   ⚠️ Top result may not be the most relevant")
    else:
        print("❌ Reranking failed - no documents returned")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_reranker()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

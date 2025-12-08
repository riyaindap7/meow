import time
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Connect
connections.connect("default", host="localhost", port="19530")
collection = Collection("pdf_vectors")
collection.load()

# Load model
model = SentenceTransformer('BAAI/bge-m3')

def search_pdfs(query_text, top_k=5, output_file=None):
    """Search for similar chunks"""
    # Generate query embedding
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    )
    
    # Search parameters
    search_params = {
        "metric_type": "IP",
        "params": {"ef": 64}
    }
    
    # Measure latency
    start_time = time.time()
    
    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "source", "page"]
    )
    
    latency = (time.time() - start_time) * 1000
    
    # Prepare output text
    output_text = []
    output_text.append(f"\n{'='*80}")
    output_text.append(f"Query: {query_text}")
    output_text.append(f"Latency: {latency:.2f}ms")
    output_text.append('='*80)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Query: {query_text}")
    print(f"Latency: {latency:.2f}ms")
    print('='*80)
    
    for i, hits in enumerate(results):
        for rank, hit in enumerate(hits, 1):
            result_text = f"\n[Result {rank}] Score: {hit.score:.4f}"
            source_text = f"Source: {hit.entity.get('source')} (Page {hit.entity.get('page')})"
            content_text = f"Text: {hit.entity.get('text')[:300]}..."
            
            print(result_text)
            print(source_text)
            print(content_text)
            
            output_text.append(result_text)
            output_text.append(source_text)
            output_text.append(content_text)
    
    # Write to file if specified
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n".join(output_text) + "\n")
    
    return results

if __name__ == "__main__":
    main_query="What is restrictions on carbonated drinks in schools?"
    print(f"Running main query: {main_query}")
    search_pdfs(main_query, top_k=3)
    """
    # Load test queries from file
    test_queries_file = "../test_queries.txt"
    
    # Create results file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../test_results_{timestamp}.txt"
    
    print(f"Loading test queries from {test_queries_file}...")
    
    # Write header to results file
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("FSSAI RAG SYSTEM - TEST RESULTS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    test_queries = []
    with open(test_queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments (#), and section headers (##)
            if line and not line.startswith("#") and not line.startswith("##"):
                # Remove leading numbers and dots (e.g., "1. ", "23. ")
                if line[0].isdigit():
                    # Find the first space after the number and dot
                    query = line.split(". ", 1)[1] if ". " in line else line
                else:
                    query = line
                test_queries.append(query)
    
    print(f"✅ Loaded {len(test_queries)} test queries")
    print(f"✅ Results will be saved to: {results_file}\n")
    
    # Run all test queries
    for idx, query in enumerate(test_queries, 1):
        header = f"\n{'#'*80}\nQUERY {idx}/{len(test_queries)}\n{'#'*80}\n"
        print(header)
        
        # Write query header to file
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(header)
        
        search_pdfs(query, top_k=3, output_file=results_file)
        print("\n")
        
        # Optional: Add a small delay to avoid overwhelming the system
        # time.sleep(0.5)
    
    # Write summary to results file
    with open(results_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"TEST COMPLETED: {len(test_queries)} queries executed\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"✅ All {len(test_queries)} queries completed!")
    print(f"✅ Results saved to: {results_file}")
    print(f"{'='*80}")
"""
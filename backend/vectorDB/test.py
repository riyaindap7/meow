import numpy as np
from scipy import sparse

# Check text sparse
text_sparse = np.load("sparse_embeddings/text_sparse_embeddings.npz", allow_pickle=True)
print("Text sparse keys:", list(text_sparse.keys()))
print("Text sparse shapes:", {k: text_sparse[k].shape for k in text_sparse.keys()})

# Check table sparse  
table_sparse = np.load("sparse_embeddings/table_sparse_embeddings.npz", allow_pickle=True)
print("Table sparse keys:", list(table_sparse.keys()))
print("Table sparse shapes:", {k: table_sparse[k].shape for k in table_sparse.keys()})
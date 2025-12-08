"""
BGE-M3 Sparse Embedding Generator - ONLY sparse embeddings
Uses BAAI/bge-m3 to generate sparse embeddings and saves as NPZ (compressed numpy)
GPU-optimized version
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel
import logging
from tqdm import tqdm
from scipy import sparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SparseEmbeddingGenerator:
    """Generate ONLY sparse embeddings using BGE-M3"""
    
    def _init_(self, model_name: str = 'BAAI/bge-m3', device: str = None):
        """
        Initialize BGE-M3 model
        
        Args:
            model_name: Model identifier
            use_fp16: Use half precision for faster inference
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Detect and set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Log GPU info
        if self.device == 'cuda':
            logger.info(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            logger.info(f"   Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("⚠  No GPU available, using CPU (will be slower)")
        
        logger.info(f"Loading BGE-M3 model: {model_name}")
        
        # Initialize model with explicit device
        self.model = BGEM3FlagModel(
            model_name, 
            device=self.device
        )
        
        logger.info(f"✅ Model loaded successfully on {self.device.upper()}")
        
        # Verify model is on GPU
        if self.device == 'cuda':
            self._verify_gpu_usage()
    
    def _verify_gpu_usage(self):
        """Verify that model is actually using GPU"""
        try:
            # Check if model parameters are on GPU
            for name, param in self.model.model.named_parameters():
                if param.is_cuda:
                    logger.info(f"✅ Model confirmed on GPU: {param.device}")
                    break
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"📊 GPU Memory Usage:")
            logger.info(f"   Allocated: {memory_allocated:.2f} GB")
            logger.info(f"   Reserved: {memory_reserved:.2f} GB")
            
        except Exception as e:
            logger.warning(f"Could not verify GPU usage: {e}")
    
    def extract_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text content from chunk"""
        text_parts = []
        
        # Primary text field
        if text := chunk.get('text'):
            text_parts.append(text)
        
        # Section hierarchy for context
        if hierarchy := chunk.get('section_hierarchy'):
            text_parts.append(hierarchy)
        
        # Heading context
        if heading := chunk.get('heading_context'):
            text_parts.append(heading)
        
        # For tables, use table_text or table_markdown
        if table_text := chunk.get('table_text'):
            text_parts.append(table_text)
        elif table_md := chunk.get('table_markdown'):
            text_parts.append(table_md)
        
        return ' '.join(filter(None, text_parts))
    
    def generate_sparse(self, chunks: List[Dict[str, Any]], 
                       batch_size: int = 32) -> tuple:
        """
        Generate ONLY sparse embeddings for chunks
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for encoding
            
        Returns:
            Tuple of (sparse_embeddings, valid_indices)
        """
        logger.info(f"Generating sparse embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [self.extract_text(chunk) for chunk in chunks]
        
        # Filter empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            logger.error("No valid texts found!")
            return [], []
        
        logger.info(f"Processing {len(valid_texts)} valid texts...")
        
        # Adjust batch size for GPU
        if self.device == 'cuda':
            batch_size = min(batch_size * 2, 64)
            logger.info(f"🚀 Using GPU-optimized batch size: {batch_size}")
        
        # Generate sparse embeddings in batches
        all_sparse = []
        
        progress_bar = tqdm(range(0, len(valid_texts), batch_size), desc="Encoding batches")
        
        for i in progress_bar:
            batch_texts = valid_texts[i:i + batch_size]
            
            # Monitor GPU memory if using CUDA
            if self.device == 'cuda' and i % 10 == 0:
                mem_used = torch.cuda.memory_allocated(0) / 1e9
                progress_bar.set_postfix({'GPU_mem': f'{mem_used:.2f}GB'})
            
            # BGE-M3: Get ONLY sparse embeddings
            with torch.no_grad():
                output = self.model.encode(
                    batch_texts,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                    batch_size=batch_size
                )
            
            # Collect sparse embeddings (lexical_weights)
            batch_sparse = output['lexical_weights']
            all_sparse.extend(batch_sparse)
            
            # Clear GPU cache periodically
            if self.device == 'cuda' and i % 50 == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"✅ Generated {len(all_sparse)} sparse embeddings")
        
        # Final GPU memory report
        if self.device == 'cuda':
            mem_final = torch.cuda.memory_allocated(0) / 1e9
            mem_peak = torch.cuda.max_memory_allocated(0) / 1e9
            logger.info(f"📊 Final GPU Memory:")
            logger.info(f"   Current: {mem_final:.2f} GB")
            logger.info(f"   Peak: {mem_peak:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        return all_sparse, valid_indices
    
    def save_sparse_npz(self, sparse_embeddings: List[Dict],
                        chunks: List[Dict[str, Any]],
                        output_path: Path):
        """
        Save sparse embeddings in NPZ format (CSR - Compressed Sparse Row)
        Much more efficient than JSON for sparse data
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {len(sparse_embeddings)} embeddings to CSR format...")
        
        # Build CSR matrix components
        all_indices = []
        all_values = []
        indptr = [0]
        
        # Collect metadata
        metadata = []
        
        for chunk, sparse_dict in tqdm(zip(chunks, sparse_embeddings), 
                                      total=len(chunks), 
                                      desc="Converting to CSR"):
            # Extract indices and values from sparse dict
            if sparse_dict:
                indices = list(sparse_dict.keys())
                values = list(sparse_dict.values())
                
                # Convert to numpy arrays with correct types
                indices = np.array([int(idx) for idx in indices], dtype=np.int32)
                values = np.array([float(val) for val in values], dtype=np.float32)
                
                all_indices.extend(indices)
                all_values.extend(values)
            
            # Update pointer
            indptr.append(len(all_indices))
            
            # Store metadata
            metadata.append({
                'global_chunk_id': chunk.get('global_chunk_id'),
                'chunk_id': chunk.get('chunk_id'),
                'document_id': chunk.get('document_id'),
                'source_file': chunk.get('source_file'),
                'page_idx': int(chunk.get('page_idx', 0)),
                'chunk_index': int(chunk.get('chunk_index', 0)),
                'num_nonzero': len(sparse_dict) if sparse_dict else 0
            })
        
        # Convert to numpy arrays
        all_indices = np.array(all_indices, dtype=np.int32)
        all_values = np.array(all_values, dtype=np.float32)
        indptr = np.array(indptr, dtype=np.int32)
        
        # Determine vocabulary size
        vocab_size = max(all_indices) + 1 if all_indices.size > 0 else 0
        
        logger.info(f"💾 Saving to {output_path}...")
        
        # Save as compressed NPZ with CSR format
        np.savez_compressed(
            output_path,
            indices=all_indices,      # Token IDs
            values=all_values,        # Weights
            indptr=indptr,            # Row pointers
            shape=np.array([len(chunks), vocab_size], dtype=np.int32),
            metadata=np.array(metadata, dtype=object)
        )
        
        logger.info(f"✅ Saved sparse embeddings: {output_path}")
        logger.info(f"   Total chunks: {len(chunks)}")
        logger.info(f"   Total non-zero elements: {len(all_indices)}")
        logger.info(f"   Vocabulary size: {vocab_size}")
        
        # Calculate statistics
        avg_nonzero = len(all_indices) / len(chunks) if chunks else 0
        logger.info(f"   Average non-zero per embedding: {avg_nonzero:.2f}")
        
        # File size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"   File size: {file_size_mb:.2f} MB")
        
        # Compare with theoretical JSON size
        json_size_estimate = (len(all_indices) * 20) / (1024 * 1024)
        logger.info(f"   Estimated JSON size: {json_size_estimate:.2f} MB")
        logger.info(f"   Compression ratio: {json_size_estimate / file_size_mb:.2f}x smaller")


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from JSON file"""
    logger.info(f"📂 Loading chunks from {file_path}")
    
    if not file_path.exists():
        logger.warning(f"❌ File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks
        if chunk and (chunk.get('text') or chunk.get('table_text'))
    ]
    
    logger.info(f"✅ Loaded {len(valid_chunks)} valid chunks (from {len(chunks)} total)")
    return valid_chunks


def main():
    """Main execution"""
    logger.info("\n" + "="*60)
    logger.info("BGE-M3 SPARSE EMBEDDING GENERATOR (NPZ FORMAT)")
    logger.info("GPU-optimized | Compressed Sparse Row (CSR)")
    logger.info("="*60 + "\n")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA is available!")
        logger.info(f"   Device count: {torch.cuda.device_count()}")
        logger.info(f"   Current device: {torch.cuda.current_device()}")
    else:
        logger.warning("⚠  CUDA not available")
    
    # Paths
    base_path = Path(__file__).parent
    text_chunks_path = base_path / "chunked_outputs_v2" / "all_text_chunks.json"
    table_chunks_path = base_path / "chunked_outputs_v2" / "all_table_chunks.json"
    output_dir = base_path / "sparse_embeddings"
    
    # Load chunks
    text_chunks = load_chunks(text_chunks_path)
    table_chunks = load_chunks(table_chunks_path)
    
    if not text_chunks and not table_chunks:
        logger.error("❌ No chunks found to process!")
        return
    
    # Initialize generator with GPU
    generator = SparseEmbeddingGenerator(
        model_name='BAAI/bge-m3',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process text chunks
    if text_chunks:
        logger.info(f"\n📝 Processing {len(text_chunks)} TEXT chunks...")
        sparse, valid_idx = generator.generate_sparse(text_chunks, batch_size=32)
        
        # Keep only valid chunks
        valid_text_chunks = [text_chunks[i] for i in valid_idx]
        
        # Save as NPZ
        output_path = output_dir / "text_sparse_embeddings.npz"
        generator.save_sparse_npz(sparse, valid_text_chunks, output_path)
    
    # Process table chunks
    if table_chunks:
        logger.info(f"\n📊 Processing {len(table_chunks)} TABLE chunks...")
        sparse, valid_idx = generator.generate_sparse(table_chunks, batch_size=16)
        
        # Keep only valid chunks
        valid_table_chunks = [table_chunks[i] for i in valid_idx]
        
        # Save as NPZ
        output_path = output_dir / "table_sparse_embeddings.npz"
        generator.save_sparse_npz(sparse, valid_table_chunks, output_path)
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("\n🧹 GPU cache cleared")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ SPARSE EMBEDDING GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Text chunks processed: {len(text_chunks) if text_chunks else 0}")
    logger.info(f"Table chunks processed: {len(table_chunks) if table_chunks else 0}")
    logger.info("\nGenerated files:")
    logger.info("  - text_sparse_embeddings.npz")
    logger.info("  - table_sparse_embeddings.npz")
    logger.info("\nFormat: Compressed Sparse Row (CSR) in NPZ")
    logger.info("  - 10x smaller than JSON")
    logger.info("  - Fast loading with numpy/scipy")
    logger.info("  - Compatible with Milvus sparse vectors")


if __name__ == "__main__":
    main()
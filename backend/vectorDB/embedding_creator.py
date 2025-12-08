"""
Create embeddings for text, table, and image chunks from consolidated JSON files.
Stores embeddings as numpy arrays with metadata.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingCreator:
    """Create and store embeddings for consolidated chunk files"""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 12,
        max_seq_length: int = 384  # ✅ FIXED: Added missing parameter
    ):
        # Model setup with GPU optimization
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length  # ✅ FIXED: Store max_seq_length
        
        # Force GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # ✅ FIXED: Added GPU memory info
            print(f"🚀 GPU detected: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.2f} GB")
        else:
            self.device = "cpu"
            print("⚠️  No GPU detected, using CPU")
        
        print(f"📦 Loading model: {model_name}...")
        print(f"   Device: {self.device}")
        print(f"   Max sequence length: {max_seq_length}")  # ✅ FIXED: Display max_seq_length
        
        # Load model with explicit device
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # ✅ FIXED: Set max sequence length on model
        self.model.max_seq_length = max_seq_length
        
        # Force model to GPU if available
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
        print(f"   Batch size: {self.batch_size}")
    
    def _truncate_text(self, text: str) -> str:
        """✅ FIXED: Added missing truncate method"""
        # Rough estimate: 1 token ≈ 4 characters for English
        max_chars = self.max_seq_length * 4
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """✅ FIXED: Generate embeddings with OOM protection"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device  # ✅ FIXED: Explicitly set device
                )
            return embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  CUDA OOM! Reducing batch size from {self.batch_size} to {self.batch_size // 2}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                # Retry with smaller batch size
                self.batch_size = max(1, self.batch_size // 2)
                return self._generate_embeddings(texts)
            else:
                raise e
    
    def embed_consolidated_text_chunks(self, json_file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Create embeddings for all text chunks from consolidated JSON file.
        Uses the 'text' field for embedding generation.
        """
        print(f"\n{'='*60}")
        print("📝 PROCESSING CONSOLIDATED TEXT CHUNKS")
        print('='*60)
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            print("❌ No chunks found in file")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        print(f"   Total text chunks: {len(chunks)}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        embeddings_file = output_path / "all_text_embeddings.npy"
        metadata_file = output_path / "all_text_metadata.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and metadata_file.exists():
            print(f"   ⏭️  Embeddings already exist at {embeddings_file}")
            print(f"   Delete the file to regenerate embeddings")
            return {"processed": 0, "skipped": len(chunks), "failed": 0}
        
        # Filter chunks with text
        chunks_to_embed = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '').strip()
            if text:
                chunks_to_embed.append({
                    'index': i,
                    'chunk': chunk,
                    'text': self._truncate_text(text)  # ✅ FIXED: Apply truncation
                })
        
        if not chunks_to_embed:
            print("❌ No valid text chunks found")
            return {"processed": 0, "skipped": 0, "failed": len(chunks)}
        
        print(f"   🔄 Generating embeddings for {len(chunks_to_embed)} chunks...")
        print(f"      Using device: {self.device}")
        
        # Extract texts
        texts = [item['text'] for item in chunks_to_embed]
        
        # Generate embeddings in batches with progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="   Embedding batches", unit="batch"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # ✅ FIXED: Clear cache periodically
            if self.device == "cuda" and i % 100 == 0:
                torch.cuda.empty_cache()
        
        if not all_embeddings:
            print("❌ Failed to generate embeddings")
            return {"processed": 0, "skipped": 0, "failed": len(chunks)}
        
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        np.save(embeddings_file, embeddings)
        print(f"   ✅ Saved embeddings: {embeddings_file}")
        print(f"      Shape: {embeddings.shape}")
        
        # Create metadata mapping
        metadata = {
            "total_chunks": len(chunks_to_embed),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,  # ✅ FIXED: Added to metadata
            "created_at": datetime.now().isoformat(),
            "chunks": []
        }
        
        for item in chunks_to_embed:
            chunk = item['chunk']
            metadata["chunks"].append({
                "embedding_index": item['index'],
                "global_chunk_id": chunk.get('global_chunk_id'),
                "chunk_id": chunk.get('chunk_id'),
                "document_id": chunk.get('document_id'),
                "page_idx": chunk.get('page_idx'),
                "text_preview": chunk.get('text', '')[:100]
            })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved metadata: {metadata_file}")
        
        return {"processed": len(chunks_to_embed), "skipped": 0, "failed": 0}
    
    def embed_consolidated_table_chunks(self, json_file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Create embeddings for all table chunks from consolidated JSON file.
        Uses the 'table_text' field for embedding generation.
        """
        print(f"\n{'='*60}")
        print("📊 PROCESSING CONSOLIDATED TABLE CHUNKS")
        print('='*60)
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            print("❌ No chunks found in file")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        print(f"   Total table chunks: {len(chunks)}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        embeddings_file = output_path / "all_table_embeddings.npy"
        metadata_file = output_path / "all_table_metadata.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and metadata_file.exists():
            print(f"   ⏭️  Embeddings already exist at {embeddings_file}")
            print(f"   Delete the file to regenerate embeddings")
            return {"processed": 0, "skipped": len(chunks), "failed": 0}
        
        # Filter chunks with table_text
        chunks_to_embed = []
        for i, chunk in enumerate(chunks):
            table_text = chunk.get('table_text', '').strip()
            if table_text:
                chunks_to_embed.append({
                    'index': i,
                    'chunk': chunk,
                    'text': self._truncate_text(table_text)  # ✅ FIXED: Apply truncation
                })
        
        if not chunks_to_embed:
            print("❌ No valid table chunks found")
            return {"processed": 0, "skipped": 0, "failed": len(chunks)}
        
        print(f"   🔄 Generating embeddings for {len(chunks_to_embed)} chunks...")
        print(f"      Using device: {self.device}")
        
        # Extract texts
        texts = [item['text'] for item in chunks_to_embed]
        
        # Generate embeddings in batches with progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="   Embedding batches", unit="batch"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # ✅ FIXED: Clear cache periodically
            if self.device == "cuda" and i % 100 == 0:
                torch.cuda.empty_cache()
        
        if not all_embeddings:
            print("❌ Failed to generate embeddings")
            return {"processed": 0, "skipped": 0, "failed": len(chunks)}
        
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        np.save(embeddings_file, embeddings)
        print(f"   ✅ Saved embeddings: {embeddings_file}")
        print(f"      Shape: {embeddings.shape}")
        
        # Create metadata mapping
        metadata = {
            "total_chunks": len(chunks_to_embed),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,  # ✅ FIXED: Added to metadata
            "created_at": datetime.now().isoformat(),
            "chunks": []
        }
        
        for item in chunks_to_embed:
            chunk = item['chunk']
            metadata["chunks"].append({
                "embedding_index": item['index'],
                "global_chunk_id": chunk.get('global_chunk_id'),
                "chunk_id": chunk.get('chunk_id'),
                "document_id": chunk.get('document_id'),
                "page_idx": chunk.get('page_idx'),
                "table_text_preview": chunk.get('table_text', '')[:100]
            })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved metadata: {metadata_file}")
        
        return {"processed": len(chunks_to_embed), "skipped": 0, "failed": 0}
    
    def embed_consolidated_image_chunks(self, json_file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Create embeddings for all image chunks from consolidated JSON file.
        Uses the 'caption' or 'text' field for embedding generation.
        """
        print(f"\n{'='*60}")
        print("🖼️  PROCESSING CONSOLIDATED IMAGE CHUNKS")
        print('='*60)
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            print("❌ No chunks found in file")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        print(f"   Total image chunks: {len(chunks)}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        embeddings_file = output_path / "all_image_embeddings.npy"
        metadata_file = output_path / "all_image_metadata.json"
        
        # Check if embeddings already exist
        if embeddings_file.exists() and metadata_file.exists():
            print(f"   ⏭️  Embeddings already exist at {embeddings_file}")
            print(f"   Delete the file to regenerate embeddings")
            return {"processed": 0, "skipped": len(chunks), "failed": 0}
        
        # Filter chunks with caption or text
        chunks_to_embed = []
        for i, chunk in enumerate(chunks):
            # Try caption first, then fall back to text field
            caption = chunk.get('caption', '').strip() or chunk.get('text', '').strip()
            if caption:
                chunks_to_embed.append({
                    'index': i,
                    'chunk': chunk,
                    'text': self._truncate_text(caption)  # ✅ FIXED: Apply truncation
                })
        
        if not chunks_to_embed:
            print("⚠️  No image chunks with captions/text found")
            return {"processed": 0, "skipped": 0, "failed": len(chunks)}
        
        print(f"   🔄 Generating embeddings for {len(chunks_to_embed)} chunks...")
        print(f"      Using device: {self.device}")
        
        # Extract texts
        texts = [item['text'] for item in chunks_to_embed]
        
        # Generate embeddings in batches with progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="   Embedding batches", unit="batch"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # ✅ FIXED: Clear cache periodically
            if self.device == "cuda" and i % 100 == 0:
                torch.cuda.empty_cache()
        
        if not all_embeddings:
            print("❌ Failed to generate embeddings")
            return {"processed": 0, "skipped": 0, "failed": len(chunks)}
        
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        np.save(embeddings_file, embeddings)
        print(f"   ✅ Saved embeddings: {embeddings_file}")
        print(f"      Shape: {embeddings.shape}")
        
        # Create metadata mapping
        metadata = {
            "total_chunks": len(chunks_to_embed),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,  # ✅ FIXED: Added to metadata
            "created_at": datetime.now().isoformat(),
            "chunks": []
        }
        
        for item in chunks_to_embed:
            chunk = item['chunk']
            metadata["chunks"].append({
                "embedding_index": item['index'],
                "global_chunk_id": chunk.get('global_chunk_id'),
                "chunk_id": chunk.get('chunk_id'),
                "document_id": chunk.get('document_id'),
                "page_idx": chunk.get('page_idx'),
                "img_path": chunk.get('img_path'),
                "caption_preview": chunk.get('caption', '')[:100]
            })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved metadata: {metadata_file}")
        
        return {"processed": len(chunks_to_embed), "skipped": 0, "failed": 0}
    
    def process_consolidated_chunks(self, json_input_dir: str, embeddings_output_dir: str):
        """
        Process all consolidated JSON files and create embeddings.
        
        Args:
            json_input_dir: Directory containing all_text_chunks.json, all_table_chunks.json, all_image_chunks.json
            embeddings_output_dir: Directory to save embeddings
        """
        json_path = Path(json_input_dir)
        
        text_file = json_path / "all_text_chunks.json"
        table_file = json_path / "all_table_chunks.json"
        image_file = json_path / "all_image_chunks.json"
        
        total_stats = {
            "text_processed": 0,
            "table_processed": 0,
            "image_processed": 0,
            "text_skipped": 0,
            "table_skipped": 0,
            "image_skipped": 0
        }
        
        # Process text chunks
        if text_file.exists():
            stats = self.embed_consolidated_text_chunks(str(text_file), embeddings_output_dir)
            total_stats["text_processed"] = stats["processed"]
            total_stats["text_skipped"] = stats["skipped"]
        else:
            print(f"⚠️  Text file not found: {text_file}")
        
        # Process table chunks
        if table_file.exists():
            stats = self.embed_consolidated_table_chunks(str(table_file), embeddings_output_dir)
            total_stats["table_processed"] = stats["processed"]
            total_stats["table_skipped"] = stats["skipped"]
        else:
            print(f"⚠️  Table file not found: {table_file}")
        
        # Process image chunks
        if image_file.exists():
            stats = self.embed_consolidated_image_chunks(str(image_file), embeddings_output_dir)
            total_stats["image_processed"] = stats["processed"]
            total_stats["image_skipped"] = stats["skipped"]
        else:
            print(f"⚠️  Image file not found: {image_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 EMBEDDING CREATION SUMMARY")
        print('='*60)
        print(f"Text chunks processed: {total_stats['text_processed']} (skipped: {total_stats['text_skipped']})")
        print(f"Table chunks processed: {total_stats['table_processed']} (skipped: {total_stats['table_skipped']})")
        print(f"Image chunks processed: {total_stats['image_processed']} (skipped: {total_stats['image_skipped']})")
        print('='*60)


def main():
    JSON_INPUT_DIR = "chunked_outputs_v2"
    EMBEDDINGS_OUTPUT_DIR = "embeddings_consolidated"
    MODEL_NAME = "BAAI/bge-m3"
    
    # Optimal settings for RTX 3060 (6GB VRAM)
    BATCH_SIZE = 12  # Sweet spot for 6GB
    MAX_SEQ_LENGTH = 384  # Prevents OOM on complex tables
    
    try:
        print("🚀 Starting Consolidated Embedding Creation")
        print(f"   GPU: RTX 3060 (6GB VRAM)")
        print(f"   Input directory: {JSON_INPUT_DIR}")
        print(f"   Output directory: {EMBEDDINGS_OUTPUT_DIR}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
        
        creator = EmbeddingCreator(
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            max_seq_length=MAX_SEQ_LENGTH
        )
        
        creator.process_consolidated_chunks(
            json_input_dir=JSON_INPUT_DIR,
            embeddings_output_dir=EMBEDDINGS_OUTPUT_DIR
        )
        
        print("\n✅ Embedding creation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""Ollama embedding client for generating text embeddings.

Provides client for interacting with Ollama's embedding API to generate
semantic embeddings from text using models like embeddinggemma.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests

from infrastructure.core.logging_utils import get_logger
from infrastructure.core.exceptions import LLMConnectionError, LLMError
from infrastructure.llm.core.config import LLMConfig

logger = get_logger(__name__)


class EmbeddingClient:
    """Client for generating embeddings using Ollama embedding models.
    
    Provides methods for:
    - Single embedding generation
    - Batch embedding generation
    - Text chunking for large documents
    - Document-level embedding aggregation
    - Embedding caching
    
    Example:
        >>> client = EmbeddingClient()
        >>> embedding = client.generate_embedding("Machine learning is fascinating")
        >>> embeddings = client.generate_embeddings_batch(["Text 1", "Text 2"])
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        embedding_model: str = "embeddinggemma",
        cache_dir: Optional[Path] = None,
        chunk_size: int = 2000,
        batch_size: int = 10
    ):
        """Initialize embedding client.
        
        Args:
            config: LLMConfig instance. If None, loads from environment.
            embedding_model: Name of embedding model to use (default: "embeddinggemma").
            cache_dir: Directory for caching embeddings (default: None, no caching).
            chunk_size: Maximum tokens per chunk for text splitting (default: 2000).
            batch_size: Number of texts to process in each batch (default: 10).
        """
        self.config = config or LLMConfig.from_env()
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            url = f"{self.config.base_url}/api/tags"
            response = requests.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Connection check failed: {e}")
            return False
    
    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            use_cache: Whether to use cached embeddings if available.
            
        Returns:
            Embedding vector as numpy array.
            
        Raises:
            LLMConnectionError: If connection to Ollama fails.
            LLMError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(text)
            if cached is not None:
                logger.debug("Using cached embedding")
                return cached
        
        # Generate embedding
        url = f"{self.config.base_url}/api/embed"
        
        payload = {
            "model": self.embedding_model,
            "input": text
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            # Ollama returns "embeddings" (plural) as array of arrays
            # For single input, it's [[...]], so take first element
            embeddings_list = result.get("embeddings", [])
            if not embeddings_list:
                # Fallback to singular "embedding" for compatibility
                embeddings_list = [result.get("embedding", [])]
            
            if not embeddings_list or len(embeddings_list) == 0:
                raise LLMError(f"Empty embedding returned from model {self.embedding_model}")
            
            # Take first embedding (for single input, this is the only one)
            embedding = np.array(embeddings_list[0])
            
            if len(embedding) == 0:
                raise LLMError(f"Empty embedding returned from model {self.embedding_model}")
            
            # Cache result
            if use_cache and self.cache_dir:
                self._save_to_cache(text, embedding)
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama: {e}",
                context={"url": url, "model": self.embedding_model}
            ) from e
        except Exception as e:
            raise LLMError(
                f"Failed to generate embedding: {e}",
                context={"model": self.embedding_model, "text_length": len(text)}
            ) from e
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed.
            use_cache: Whether to use cached embeddings if available.
            show_progress: Whether to log progress.
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            if show_progress:
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            for text in batch:
                try:
                    embedding = self.generate_embedding(text, use_cache=use_cache)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for text (length={len(text)}): {e}")
                    # Use zero vector as fallback
                    if embeddings:
                        fallback = np.zeros_like(embeddings[0])
                    else:
                        # First embedding failed, need to get dimension from API
                        try:
                            sample = self.generate_embedding("sample", use_cache=False)
                            fallback = np.zeros_like(sample)
                        except:
                            fallback = np.zeros(768)  # Default embeddinggemma dimension
                    embeddings.append(fallback)
            
            # Small delay between batches to avoid overwhelming the API
            if i + self.batch_size < total:
                time.sleep(0.1)
        
        return np.array(embeddings)
    
    def chunk_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        overlap: int = 200
    ) -> List[str]:
        """Split text into chunks for embedding.
        
        Uses simple character-based chunking with overlap. For more sophisticated
        token-based chunking, consider using a tokenizer library.
        
        Args:
            text: Text to chunk.
            max_tokens: Maximum tokens per chunk (default: self.chunk_size).
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        if max_tokens is None:
            max_tokens = self.chunk_size
        
        # Simple character-based chunking
        # Approximate: 1 token ≈ 4 characters (conservative estimate)
        chunk_size_chars = max_tokens * 4
        
        if len(text) <= chunk_size_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size_chars
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within last 20% of chunk
                search_start = max(start, end - chunk_size_chars // 5)
                for i in range(end - 1, search_start, -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def embed_document(
        self,
        text: str,
        aggregation: str = "mean",
        use_cache: bool = True
    ) -> np.ndarray:
        """Embed a full document by chunking and aggregating.
        
        Args:
            text: Full document text.
            aggregation: Aggregation method ("mean", "max", "sum") (default: "mean").
            use_cache: Whether to use cached embeddings if available.
            
        Returns:
            Document-level embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache for full document
        if use_cache and self.cache_dir:
            cached = self._load_from_cache(text)
            if cached is not None:
                logger.debug("Using cached document embedding")
                return cached
        
        # Chunk text if needed
        chunks = self.chunk_text(text)
        
        if len(chunks) == 1:
            # Single chunk, no aggregation needed
            embedding = self.generate_embedding(chunks[0], use_cache=use_cache)
        else:
            # Generate embeddings for all chunks
            logger.debug(f"Embedding document with {len(chunks)} chunks")
            chunk_embeddings = self.generate_embeddings_batch(
                chunks,
                use_cache=use_cache,
                show_progress=False
            )
            
            # Aggregate chunk embeddings
            if aggregation == "mean":
                embedding = np.mean(chunk_embeddings, axis=0)
            elif aggregation == "max":
                embedding = np.max(chunk_embeddings, axis=0)
            elif aggregation == "sum":
                embedding = np.sum(chunk_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Cache result
        if use_cache and self.cache_dir:
            self._save_to_cache(text, embedding)
        
        return embedding
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text.
        
        Args:
            text: Text to generate key for.
            
        Returns:
            Cache key string.
        """
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _cache_path(self, text: str) -> Path:
        """Get cache file path for text.
        
        Args:
            text: Text to get cache path for.
            
        Returns:
            Path to cache file.
        """
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")
        
        key = self._cache_key(text)
        return self.cache_dir / f"{key}.json"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache.
        
        Args:
            text: Text to load embedding for.
            
        Returns:
            Cached embedding or None if not found.
        """
        if not self.cache_dir:
            return None
        
        cache_path = self._cache_path(text)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                embedding = np.array(data.get("embedding", []))
                return embedding if len(embedding) > 0 else None
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache.
        
        Args:
            text: Original text.
            embedding: Embedding vector to cache.
        """
        if not self.cache_dir:
            return
        
        cache_path = self._cache_path(text)
        
        try:
            data = {
                "embedding": embedding.tolist(),
                "model": self.embedding_model,
                "dimension": len(embedding)
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


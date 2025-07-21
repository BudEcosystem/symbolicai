"""
FAISS integration for efficient similarity search in large-scale prototype sets.
Provides automatic index selection, thread-safe operations, and numpy fallback.
"""

import numpy as np
import threading
from typing import Optional, Tuple, Union, List
import logging
import os

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.info("FAISS not available. Using numpy fallback for similarity search.")


class FAISSManager:
    """
    Manages FAISS indices for efficient similarity search.
    Falls back to numpy when FAISS is not available.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)"""
        with cls._lock:
            cls._instance = None
    
    def __init__(
        self,
        auto_index_threshold: int = 1000,
        normalize_embeddings: bool = True,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS manager.
        
        Args:
            auto_index_threshold: Number of vectors above which to use approximate search
            normalize_embeddings: Whether to L2-normalize embeddings
            use_gpu: Whether to use GPU acceleration (if available)
        """
        if self._initialized:
            # Allow updating configuration for existing instance
            self.auto_index_threshold = auto_index_threshold
            self.normalize_embeddings = normalize_embeddings
            self.use_gpu = use_gpu and FAISS_AVAILABLE and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
            return
            
        self.auto_index_threshold = auto_index_threshold
        self.normalize_embeddings = normalize_embeddings
        self.use_gpu = use_gpu and FAISS_AVAILABLE and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        self.use_numpy_fallback = not FAISS_AVAILABLE
        self._indices = {}  # Cache for indices
        self._index_lock = threading.RLock()
        self._initialized = True
    
    def create_index(
        self,
        dimension: int,
        index_type: str = 'Flat',
        **kwargs
    ) -> Optional['faiss.Index']:
        """
        Create a FAISS index of specified type.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index ('Flat', 'IVF', 'HNSW')
            **kwargs: Additional parameters for specific index types
            
        Returns:
            FAISS index or None if using numpy fallback
        """
        if self.use_numpy_fallback:
            return None
            
        with self._index_lock:
            if index_type == 'Flat':
                # Exact search
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            elif index_type == 'IVF':
                # Inverted file index for approximate search
                nlist = kwargs.get('nlist', 100)
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
            elif index_type == 'HNSW':
                # Hierarchical Navigable Small World graph
                M = kwargs.get('M', 32)
                index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
                
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # Move to GPU if requested
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            
            return index
    
    def create_auto_index(self, dimension: int, n_vectors: int) -> Optional['faiss.Index']:
        """
        Automatically select index type based on dataset size.
        
        Args:
            dimension: Embedding dimension
            n_vectors: Expected number of vectors
            
        Returns:
            Appropriate FAISS index
        """
        if self.use_numpy_fallback:
            return None
            
        if n_vectors < 100:
            # Very small dataset - use exact search
            return self.create_index(dimension, 'Flat')
        elif n_vectors < self.auto_index_threshold:
            # Medium dataset - use IVF
            nlist = min(int(np.sqrt(n_vectors)), n_vectors // 4)
            nlist = max(nlist, 4)  # Ensure at least 4 clusters
            return self.create_index(dimension, 'IVF', nlist=nlist)
        else:
            # Large dataset - use HNSW
            return self.create_index(dimension, 'HNSW')
    
    def get_index_type(self, index: 'faiss.Index') -> str:
        """Get the type of a FAISS index"""
        if index is None:
            return 'numpy'
        
        index_class = type(index).__name__
        if 'Flat' in index_class:
            return 'Flat'
        elif 'IVF' in index_class:
            return 'IVF'
        elif 'HNSW' in index_class:
            return 'HNSW'
        else:
            return 'Unknown'
    
    def add_embeddings(
        self,
        index: Optional['faiss.Index'],
        embeddings: np.ndarray,
        thread_safe: bool = True
    ) -> None:
        """
        Add embeddings to index.
        
        Args:
            index: FAISS index (None for numpy fallback)
            embeddings: Embeddings to add (n_vectors x dimension)
            thread_safe: Whether to use thread locking
        """
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # Normalize if requested
        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        
        if index is None:
            # Numpy fallback - embeddings stored separately
            return
            
        if thread_safe:
            with self._index_lock:
                self._add_to_index(index, embeddings)
        else:
            self._add_to_index(index, embeddings)
    
    def _add_to_index(self, index: 'faiss.Index', embeddings: np.ndarray):
        """Internal method to add embeddings to index"""
        # Train index if needed (for IVF)
        if hasattr(index, 'is_trained') and not index.is_trained:
            index.train(embeddings)
        
        # Add embeddings
        index.add(embeddings)
    
    def search(
        self,
        index: Optional['faiss.Index'],
        queries: np.ndarray,
        k: int = 5,
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            index: FAISS index (None for numpy fallback)
            queries: Query embeddings (n_queries x dimension)
            k: Number of nearest neighbors
            embeddings: Original embeddings (required for numpy fallback)
            
        Returns:
            distances: Similarity scores (n_queries x k)
            indices: Indices of nearest neighbors (n_queries x k)
        """
        # Ensure float32 and proper shape
        queries = queries.astype(np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        # Normalize queries if needed
        if self.normalize_embeddings:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / (norms + 1e-10)
        
        if index is None:
            # Numpy fallback
            if embeddings is None:
                raise ValueError("Embeddings required for numpy fallback")
            return self.search_numpy(embeddings, queries, k)
        
        # FAISS search
        with self._index_lock:
            distances, indices = index.search(queries, k)
        
        return distances, indices
    
    def search_numpy(
        self,
        embeddings: np.ndarray,
        queries: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numpy-based similarity search (fallback).
        
        Args:
            embeddings: All embeddings (n_embeddings x dimension)
            queries: Query embeddings (n_queries x dimension)
            k: Number of nearest neighbors
            
        Returns:
            distances: Similarity scores (n_queries x k)
            indices: Indices of nearest neighbors (n_queries x k)
        """
        # Normalize if needed
        if self.normalize_embeddings:
            emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (emb_norms + 1e-10)
            
            query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / (query_norms + 1e-10)
        
        # Compute similarities
        similarities = np.dot(queries, embeddings.T)
        
        # Get top k
        k = min(k, embeddings.shape[0])
        indices = np.zeros((queries.shape[0], k), dtype=np.int64)
        distances = np.zeros((queries.shape[0], k), dtype=np.float32)
        
        for i in range(queries.shape[0]):
            # Get top k indices
            top_k_idx = np.argpartition(similarities[i], -k)[-k:]
            # Sort by similarity
            top_k_idx = top_k_idx[np.argsort(similarities[i][top_k_idx])[::-1]]
            
            indices[i] = top_k_idx
            distances[i] = similarities[i][top_k_idx]
        
        return distances, indices
    
    def save_index(self, index: 'faiss.Index', filepath: str) -> None:
        """Save FAISS index to disk"""
        if index is None:
            return
            
        with self._index_lock:
            faiss.write_index(index, filepath)
    
    def load_index(self, filepath: str) -> Optional['faiss.Index']:
        """Load FAISS index from disk"""
        if self.use_numpy_fallback:
            return None
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")
            
        with self._index_lock:
            index = faiss.read_index(filepath)
            
            # Move to GPU if needed
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            
            return index
    
    def get_statistics(self) -> dict:
        """Get manager statistics"""
        stats = {
            'faiss_available': FAISS_AVAILABLE,
            'using_gpu': self.use_gpu,
            'using_numpy_fallback': self.use_numpy_fallback,
            'auto_index_threshold': self.auto_index_threshold,
            'normalize_embeddings': self.normalize_embeddings,
            'num_cached_indices': len(self._indices)
        }
        
        if FAISS_AVAILABLE:
            stats['faiss_version'] = faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'
            
        return stats
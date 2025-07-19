from typing import List, Dict, Optional
import numpy as np
import threading
from .semantic_cache import SemanticCache

class Model2VecManager:
    """Manages model2vec instance and provides embedding functions"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.cache = SemanticCache()
        self._initialized = True
        
    async def initialize(self, model_name: str = 'minishlab/potion-base-8M'):
        """Initialize the model asynchronously"""
        if self.model is None:
            try:
                from model2vec import StaticModel
                self.model = StaticModel.from_pretrained(model_name)
            except ImportError:
                raise ImportError("model2vec not installed. Install with: pip install model2vec")
    
    def initialize_sync(self, model_name: str = 'minishlab/potion-base-8M'):
        """Initialize the model synchronously"""
        if self.model is None:
            try:
                from model2vec import StaticModel
                self.model = StaticModel.from_pretrained(model_name)
            except ImportError:
                raise ImportError("model2vec not installed. Install with: pip install model2vec")
        
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for texts with caching"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Check cache
        uncached_texts, cached_embeddings = self.cache.get_batch(texts)
        
        # Get embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts)
            
            # Cache new embeddings
            embedding_dict = {text: emb for text, emb in zip(uncached_texts, new_embeddings)}
            self.cache.put_batch(embedding_dict)
            
            # Combine with cached
            cached_embeddings.update(embedding_dict)
        
        # Return in original order
        return [cached_embeddings[text] for text in texts]
    
    def embed_sync(self, texts: List[str]) -> List[np.ndarray]:
        """Synchronous embedding function"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Check cache
        uncached_texts, cached_embeddings = self.cache.get_batch(texts)
        
        # Get embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts)
            
            # Cache new embeddings
            embedding_dict = {text: emb for text, emb in zip(uncached_texts, new_embeddings)}
            self.cache.put_batch(embedding_dict)
            
            # Combine with cached
            cached_embeddings.update(embedding_dict)
        
        # Return in original order
        return [cached_embeddings[text] for text in texts]
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))

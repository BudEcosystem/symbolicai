from typing import Dict, Optional, List, Tuple
import numpy as np
from collections import OrderedDict
import asyncio
import threading

class SemanticCache:
    """Thread-safe LRU cache for embeddings"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.lock = threading.Lock()
        
    def get(self, text: str) -> Optional[np.ndarray]:
        with self.lock:
            if text in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(text)
                return self.cache[text]
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        with self.lock:
            if text in self.cache:
                self.cache.move_to_end(text)
                return
                
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
                
            self.cache[text] = embedding
    
    def get_batch(self, texts: List[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Returns texts not in cache and cached embeddings"""
        with self.lock:
            uncached = []
            cached = {}
            
            for text in texts:
                if text in self.cache:
                    self.cache.move_to_end(text)
                    cached[text] = self.cache[text]
                else:
                    uncached.append(text)
                    
            return uncached, cached
    
    def put_batch(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Add multiple embeddings to cache"""
        with self.lock:
            for text, embedding in embeddings.items():
                # Inline the put logic to avoid nested lock acquisition
                if text in self.cache:
                    self.cache.move_to_end(text)
                    continue
                    
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
                    
                self.cache[text] = embedding

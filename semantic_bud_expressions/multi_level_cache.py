"""
Multi-level caching system for semantic bud expressions.
Provides hierarchical caching for expression results, embeddings, and patterns.
"""
from typing import Dict, Any, Optional, Tuple, List
import time
import threading
from collections import OrderedDict
import numpy as np


class MultiLevelCache:
    """
    Multi-level caching system with three levels:
    - L1: Expression-level results cache (full match results)
    - L2: Token/phrase embeddings cache 
    - L3: Pre-computed prototype embeddings
    """
    
    def __init__(
        self,
        l1_size: int = 1000,
        l2_size: int = 10000, 
        l3_size: int = 5000,
        ttl_seconds: Optional[float] = None
    ):
        """
        Initialize multi-level cache.
        
        Args:
            l1_size: Maximum size for expression cache
            l2_size: Maximum size for embedding cache
            l3_size: Maximum size for prototype cache
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
        """
        self.ttl = ttl_seconds
        
        # L1: Expression results cache
        self.l1_cache = TTLCache(l1_size, ttl_seconds)
        
        # L2: Embedding cache (inherited from existing SemanticCache)
        self.l2_cache = TTLCache(l2_size, ttl_seconds)
        
        # L3: Prototype embeddings cache
        self.l3_cache = TTLCache(l3_size, ttl_seconds)
        
        # Statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0
        }
        
    def get_expression_result(self, expression: str, text: str) -> Optional[Any]:
        """Get cached expression match result"""
        key = f"{expression}::{text}"
        result = self.l1_cache.get(key)
        
        if result is not None:
            self.stats['l1_hits'] += 1
        else:
            self.stats['l1_misses'] += 1
            
        return result
    
    def put_expression_result(self, expression: str, text: str, result: Any):
        """Cache expression match result"""
        key = f"{expression}::{text}"
        self.l1_cache.put(key, result)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        result = self.l2_cache.get(text)
        
        if result is not None:
            self.stats['l2_hits'] += 1
        else:
            self.stats['l2_misses'] += 1
            
        return result
    
    def put_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        self.l2_cache.put(text, embedding)
    
    def get_batch_embeddings(self, texts: List[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Get batch embeddings, returning uncached texts and cached results"""
        uncached = []
        cached = {}
        
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding is not None:
                cached[text] = embedding
            else:
                uncached.append(text)
                
        return uncached, cached
    
    def put_batch_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Cache multiple embeddings"""
        for text, embedding in embeddings.items():
            self.put_embedding(text, embedding)
    
    def get_prototype_embeddings(self, type_name: str) -> Optional[List[np.ndarray]]:
        """Get cached prototype embeddings for a type"""
        result = self.l3_cache.get(type_name)
        
        if result is not None:
            self.stats['l3_hits'] += 1
        else:
            self.stats['l3_misses'] += 1
            
        return result
    
    def put_prototype_embeddings(self, type_name: str, embeddings: List[np.ndarray]):
        """Cache prototype embeddings for a type"""
        self.l3_cache.put(type_name, embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']
        total_misses = self.stats['l1_misses'] + self.stats['l2_misses'] + self.stats['l3_misses']
        
        return {
            **self.stats,
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'l3_size': len(self.l3_cache),
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        }
    
    def clear(self):
        """Clear all caches"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        
        # Reset stats
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0
        }


class TTLCache:
    """Thread-safe LRU cache with optional TTL (time-to-live)"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = threading.RLock()  # Use RLock to allow re-entrant locking
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
                
            value, timestamp = self.cache[key]
            
            # Check TTL
            if self.ttl and (time.time() - timestamp) > self.ttl:
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return value
    
    def put(self, key: str, value: Any):
        with self.lock:
            # Remove if already exists to update timestamp
            if key in self.cache:
                del self.cache[key]
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            # Add with current timestamp
            self.cache[key] = (value, time.time())
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def __len__(self):
        with self.lock:
            # Clean expired entries
            if self.ttl:
                current_time = time.time()
                expired_keys = [
                    k for k, (_, ts) in self.cache.items() 
                    if (current_time - ts) > self.ttl
                ]
                for k in expired_keys:
                    del self.cache[k]
            
            return len(self.cache)


# Global multi-level cache instance
_global_cache: Optional[MultiLevelCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> MultiLevelCache:
    """Get or create the global multi-level cache instance"""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = MultiLevelCache()
    
    return _global_cache


def clear_global_cache():
    """Clear the global cache"""
    if _global_cache is not None:
        _global_cache.clear()
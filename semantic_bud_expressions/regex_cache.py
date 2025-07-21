"""
Regex compilation cache for improved performance.
Caches compiled regex patterns to avoid recompilation overhead.
"""
import re
from typing import Dict, Pattern, Optional, Tuple
import threading
from collections import OrderedDict


class RegexCache:
    """Thread-safe LRU cache for compiled regex patterns"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_size: int = 1000):
        if self._initialized:
            return
            
        self.max_size = max_size
        self.cache: OrderedDict[str, Pattern] = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self._initialized = True
        
    def compile(self, pattern: str, flags: int = 0) -> Pattern:
        """
        Compile a regex pattern with caching.
        
        Args:
            pattern: The regex pattern string
            flags: Optional regex flags
            
        Returns:
            Compiled regex pattern
        """
        cache_key = (pattern, flags)
        
        with self._lock:
            if cache_key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                self.stats['hits'] += 1
                return self.cache[cache_key]
            
            # Cache miss - compile the pattern
            self.stats['misses'] += 1
            
            try:
                compiled_pattern = re.compile(pattern, flags)
            except re.error as e:
                # Re-raise with more context
                raise re.error(f"Invalid regex pattern '{pattern}': {e}")
            
            # Add to cache
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1
                
            self.cache[cache_key] = compiled_pattern
            return compiled_pattern
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
            return {
                **self.stats,
                'size': len(self.cache),
                'hit_rate': hit_rate
            }
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self.cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0
            }
    
    def precompile_common_patterns(self):
        """Precompile commonly used regex patterns"""
        common_patterns = [
            # Basic patterns
            (r'\w+', 0),  # Word characters
            (r'\d+', 0),  # Digits
            (r'\s+', 0),  # Whitespace
            (r'.+', 0),   # Any character
            (r'.*', 0),   # Any character (0 or more)
            
            # Common bud expression patterns
            (r'[^{}]+', 0),  # Non-brace characters
            (r'\{(\w+)\}', 0),  # Basic parameter
            (r'\{(\w+):(\w+)\}', 0),  # Parameter with type hint
            
            # Math patterns
            (r'[0-9+\-*/().,\s\w^=<>≤≥]+', 0),  # Math expressions
            
            # Phrase patterns
            (r'[\w\s]+', 0),  # Words and spaces
            (r'[^\s]+(?:\s+[^\s]+)*', 0),  # Multi-word phrases
        ]
        
        for pattern, flags in common_patterns:
            try:
                self.compile(pattern, flags)
            except re.error:
                # Skip invalid patterns
                pass


# Global regex cache instance
regex_cache = RegexCache()
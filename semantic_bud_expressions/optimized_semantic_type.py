"""
Optimized semantic parameter type that supports batch embedding computation.
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from .semantic_parameter_type import SemanticParameterType
from .batch_matcher import BatchMatcher


class OptimizedSemanticParameterType(SemanticParameterType):
    """
    Semantic parameter type optimized for batch processing.
    Reuses embeddings computed for the entire input text.
    """
    
    # Class-level batch matcher shared across instances
    _batch_matcher: Optional[BatchMatcher] = None
    _current_text: Optional[str] = None
    _text_embeddings: Optional[Dict[str, np.ndarray]] = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if OptimizedSemanticParameterType._batch_matcher is None:
            OptimizedSemanticParameterType._batch_matcher = BatchMatcher(self.model_manager)
    
    @classmethod
    def prepare_text(cls, text: str, parameter_types: List['SemanticParameterType']):
        """
        Pre-compute embeddings for all phrases in the text.
        This should be called once before matching multiple parameters.
        
        Args:
            text: The input text to prepare
            parameter_types: All parameter types that will be matched
        """
        if cls._batch_matcher is None:
            return
            
        # Only recompute if text has changed
        if cls._current_text != text:
            cls._current_text = text
            cls._text_embeddings, _ = cls._batch_matcher.match_with_cache(
                text, parameter_types
            )
    
    def matches_semantically_optimized(self, phrase: str) -> Tuple[bool, float, str]:
        """
        Optimized semantic matching using pre-computed embeddings.
        
        Args:
            phrase: The phrase to match
            
        Returns:
            Tuple of (matches, similarity_score, closest_prototype)
        """
        # Ensure prototype embeddings are available
        self._ensure_embeddings()
        
        # Check if we have pre-computed embeddings for this phrase
        if (self._text_embeddings is not None and 
            phrase in self._text_embeddings):
            phrase_embedding = self._text_embeddings[phrase]
        else:
            # Fall back to computing embedding on demand
            phrase_embedding = self.model_manager.embed_sync([phrase])[0]
        
        # Calculate similarities with all prototypes
        max_similarity = 0.0
        closest_prototype = None
        
        for i, prototype in enumerate(self.prototypes):
            similarity = self.model_manager.cosine_similarity(
                phrase_embedding, 
                self._prototype_embeddings[i]
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                closest_prototype = prototype
        
        matches = max_similarity >= self.similarity_threshold
        return matches, max_similarity, closest_prototype
    
    def matches_semantically(self, text: str) -> Tuple[bool, float, str]:
        """
        Override to use optimized matching when available.
        
        Args:
            text: The text to match
            
        Returns:
            Tuple of (matches, similarity_score, closest_prototype)
        """
        # Try optimized matching first
        if self._text_embeddings is not None and text in self._text_embeddings:
            return self.matches_semantically_optimized(text)
        
        # Fall back to parent implementation
        return super().matches_semantically(text)
    
    @classmethod
    def clear_text_cache(cls):
        """Clear the cached text embeddings"""
        cls._current_text = None
        cls._text_embeddings = None
        if cls._batch_matcher:
            cls._batch_matcher.clear_cache()


class OptimizedDynamicSemanticParameterType(OptimizedSemanticParameterType):
    """
    Optimized version of dynamic semantic parameter type.
    """
    
    def __init__(self, name: str, similarity_threshold: float = 0.6):
        """
        Create an optimized dynamic semantic parameter type.
        
        Args:
            name: The name of the parameter (e.g., 'cars', 'vehicle')
            similarity_threshold: Minimum similarity score (0-1)
        """
        super().__init__(
            name=name,
            prototypes=[name],  # The name itself is the semantic reference
            similarity_threshold=similarity_threshold,
            use_for_snippets=True,
            prefer_for_regexp_match=False
        )
        self.is_dynamic = True
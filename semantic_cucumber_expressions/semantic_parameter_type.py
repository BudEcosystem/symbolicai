# semantic_cucumber_expressions/semantic_parameter_type.py
from typing import List, Optional, Callable, Dict, Any
import re
from .parameter_type import ParameterType
from .model_manager import Model2VecManager
import numpy as np

class SemanticParameterType(ParameterType):
    """Parameter type that uses semantic similarity for matching"""
    
    def __init__(
        self,
        name: str,
        prototypes: List[str],
        type: type = str,
        transformer: Optional[Callable] = None,
        similarity_threshold: float = 0.7,
        use_for_snippets: bool = True,
        prefer_for_regexp_match: bool = False,
        fallback_regexp: Optional[str] = None,
        context_hints: Optional[Dict[str, Any]] = None
    ):
        """
        Create a semantic parameter type
        
        Args:
            name: Name of the parameter type (e.g., 'fruit', 'greeting')
            prototypes: List of prototype examples (e.g., ['apple', 'banana', 'orange'])
            type: The type to transform to
            transformer: Custom transformation function
            similarity_threshold: Minimum similarity score (0-1)
            use_for_snippets: Whether to use for snippet generation
            prefer_for_regexp_match: Whether to prefer for regexp matching
            fallback_regexp: Optional regex pattern as fallback
            context_hints: Additional context for semantic matching
        """
        # Use a broad regex that captures words
        base_regexp = fallback_regexp or r'\w+'
        
        super().__init__(
            name=name,
            regexp=base_regexp,
            type=type,
            transformer=transformer,
            use_for_snippets=use_for_snippets,
            prefer_for_regexp_match=prefer_for_regexp_match
        )
        
        self.prototypes = prototypes
        self.similarity_threshold = similarity_threshold
        self.context_hints = context_hints or {}
        self.model_manager = Model2VecManager()
        self._prototype_embeddings = None
        
    def _ensure_embeddings(self):
        """Ensure prototype embeddings are computed"""
        if self._prototype_embeddings is None:
            self._prototype_embeddings = self.model_manager.embed_sync(self.prototypes)
    
    def matches_semantically(self, text: str) -> tuple[bool, float, str]:
        """
        Check if text matches semantically
        
        Returns:
            Tuple of (matches, similarity_score, closest_prototype)
        """
        self._ensure_embeddings()
        
        # Get embedding for input text
        text_embedding = self.model_manager.embed_sync([text])[0]
        
        # Calculate similarities with all prototypes
        max_similarity = 0.0
        closest_prototype = None
        
        for i, prototype in enumerate(self.prototypes):
            similarity = self.model_manager.cosine_similarity(
                text_embedding, 
                self._prototype_embeddings[i]
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                closest_prototype = prototype
        
        matches = max_similarity >= self.similarity_threshold
        return matches, max_similarity, closest_prototype
    
    def transform(self, group_values: List[str]) -> Any:
        """Transform with semantic validation"""
        if not group_values:
            return None
            
        value = group_values[0]
        
        # Check semantic match
        matches, similarity, closest = self.matches_semantically(value)
        
        if not matches:
            raise ValueError(
                f"'{value}' does not semantically match parameter type '{self.name}' "
                f"(similarity: {similarity:.2f}, threshold: {self.similarity_threshold})"
            )
        
        # Apply transformer - use parent's transform method which handles the default case
        return super().transform(group_values)

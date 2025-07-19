# semantic_cucumber_expressions/unified_parameter_type.py
from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Union, Pattern
from enum import Enum
import re

from .parameter_type import ParameterType
from .model_manager import Model2VecManager


class ParameterTypeHint(Enum):
    """Enumeration of supported parameter type hints"""
    STANDARD = "standard"      # Regular cucumber parameter
    SEMANTIC = "semantic"      # Semantic similarity matching
    DYNAMIC = "dynamic"        # Dynamic semantic matching
    REGEX = "regex"           # Custom regex pattern
    PHRASE = "phrase"         # Multi-word phrase matching
    MATH = "math"             # Mathematical expression
    QUOTED = "quoted"         # Quoted string matching


class UnifiedParameterType(ParameterType):
    """
    A unified parameter type that can handle multiple matching strategies
    based on type hints (e.g., {param:semantic}, {param:phrase}, etc.)
    """
    
    def __init__(
        self,
        name: str,
        type_hint: ParameterTypeHint = ParameterTypeHint.STANDARD,
        regexp: Optional[Union[str, List[str], Pattern]] = None,
        type: type = str,
        transformer: Optional[Callable] = None,
        use_for_snippets: bool = True,
        prefer_for_regexp_match: bool = False,
        # Semantic-specific parameters
        prototypes: Optional[List[str]] = None,
        similarity_threshold: float = 0.3,
        # Phrase-specific parameters
        phrase_delimiters: Optional[List[str]] = None,
        max_phrase_length: int = 10,
        # Custom regex parameters
        custom_pattern: Optional[str] = None,
        # Context hints
        context_hints: Optional[Dict[str, Any]] = None
    ):
        """
        Create a unified parameter type that can handle different matching strategies.
        
        Args:
            name: Name of the parameter type
            type_hint: The type hint that determines matching strategy
            regexp: Base regex pattern (auto-generated if not provided)
            type: The type to transform to
            transformer: Custom transformation function
            use_for_snippets: Whether to use for snippet generation
            prefer_for_regexp_match: Whether to prefer for regexp matching
            prototypes: List of prototype examples for semantic matching
            similarity_threshold: Minimum similarity score for semantic matching
            phrase_delimiters: List of delimiters for phrase boundary detection
            max_phrase_length: Maximum number of words in a phrase
            custom_pattern: Custom regex pattern for REGEX type hint
            context_hints: Additional context for matching
        """
        self.type_hint = type_hint
        self.prototypes = prototypes or []
        self.similarity_threshold = similarity_threshold
        self.phrase_delimiters = phrase_delimiters or ['.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}']
        self.max_phrase_length = max_phrase_length
        self.custom_pattern = custom_pattern
        self.context_hints = context_hints or {}
        
        # Initialize model manager for semantic matching
        self.model_manager = Model2VecManager() if type_hint in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC] else None
        self._prototype_embeddings = None
        
        # Generate appropriate regex based on type hint
        if regexp is None:
            regexp = self._generate_regexp()
        
        super().__init__(
            name=name,
            regexp=regexp,
            type=type,
            transformer=transformer,
            use_for_snippets=use_for_snippets,
            prefer_for_regexp_match=prefer_for_regexp_match
        )
    
    def _generate_regexp(self) -> str:
        """Generate appropriate regex pattern based on type hint"""
        if self.type_hint == ParameterTypeHint.STANDARD:
            return r'\w+'
        elif self.type_hint == ParameterTypeHint.SEMANTIC:
            return r'\w+'
        elif self.type_hint == ParameterTypeHint.DYNAMIC:
            return r'\w+'
        elif self.type_hint == ParameterTypeHint.REGEX:
            return self.custom_pattern or r'\w+'
        elif self.type_hint == ParameterTypeHint.PHRASE:
            # Match multiple words with spaces, but stop at delimiters
            delimiter_pattern = '|'.join(re.escape(d) for d in self.phrase_delimiters)
            return rf'(?:(?!{delimiter_pattern})\S+(?:\s+(?!{delimiter_pattern})\S+)*)'
        elif self.type_hint == ParameterTypeHint.MATH:
            # Match mathematical expressions
            return r'[0-9+\-*/()=\s\w]+'
        elif self.type_hint == ParameterTypeHint.QUOTED:
            # Match quoted strings
            return r'(?:"[^"]*"|\'[^\']*\'|[^\s]+)'
        else:
            return r'\w+'
    
    def _ensure_embeddings(self):
        """Ensure prototype embeddings are computed for semantic matching"""
        if self.model_manager and self.prototypes and self._prototype_embeddings is None:
            self._prototype_embeddings = self.model_manager.embed_sync(self.prototypes)
    
    def matches_semantically(self, text: str) -> tuple[bool, float, str]:
        """
        Check if text matches semantically against prototypes or parameter name.
        
        Returns:
            Tuple of (matches, similarity_score, closest_prototype)
        """
        if self.type_hint not in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC]:
            return False, 0.0, ""
        
        if not self.model_manager:
            return False, 0.0, ""
        
        # For dynamic matching, use parameter name as prototype
        if self.type_hint == ParameterTypeHint.DYNAMIC:
            if not self.prototypes:
                self.prototypes = [self.name]
                self._prototype_embeddings = None
        
        self._ensure_embeddings()
        
        if not self._prototype_embeddings:
            return False, 0.0, ""
        
        # Get embedding for input text
        text_embedding = self.model_manager.embed_sync([text])[0]
        
        # Calculate similarities with all prototypes
        max_similarity = 0.0
        closest_prototype = ""
        
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
        """Transform with type-specific validation and processing"""
        if not group_values:
            return None
        
        value = group_values[0]
        
        # Handle semantic/dynamic matching validation
        if self.type_hint in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC]:
            matches, similarity, closest = self.matches_semantically(value)
            
            if not matches:
                raise ValueError(
                    f"'{value}' does not semantically match parameter type '{self.name}' "
                    f"(similarity: {similarity:.2f}, threshold: {self.similarity_threshold})"
                )
            
            # For semantic matching, might want to include metadata
            if self.transformer:
                return self.transformer(value, similarity=similarity, closest_prototype=closest)
            else:
                return value
        
        # Handle phrase processing
        elif self.type_hint == ParameterTypeHint.PHRASE:
            # Clean up phrase (remove extra spaces, etc.)
            cleaned_value = re.sub(r'\s+', ' ', value.strip())
            
            # Validate phrase length
            word_count = len(cleaned_value.split())
            if word_count > self.max_phrase_length:
                raise ValueError(
                    f"Phrase '{cleaned_value}' has {word_count} words, "
                    f"exceeding maximum of {self.max_phrase_length}"
                )
            
            if self.transformer:
                return self.transformer(cleaned_value)
            else:
                return cleaned_value
        
        # Handle quoted string processing
        elif self.type_hint == ParameterTypeHint.QUOTED:
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            if self.transformer:
                return self.transformer(value)
            else:
                return value
        
        # Handle mathematical expressions
        elif self.type_hint == ParameterTypeHint.MATH:
            # Import here to avoid circular imports
            try:
                from .math_parameter_type import MathExpression
                return MathExpression(value)
            except ImportError:
                # Fall back to string if math module not available
                pass
        
        # Standard transformation
        if self.transformer:
            return self.transformer(value)
        else:
            return self.type(value)
    
    def get_phrase_boundaries(self, text: str, match_start: int, match_end: int) -> tuple[int, int]:
        """
        Determine the optimal phrase boundaries for multi-word matching.
        
        Args:
            text: The full text being matched
            match_start: Start position of the current match
            match_end: End position of the current match
            
        Returns:
            Tuple of (actual_start, actual_end) positions
        """
        if self.type_hint != ParameterTypeHint.PHRASE:
            return match_start, match_end
        
        # Look for phrase boundaries around the match
        # This is a simplified implementation - could be enhanced with NLP
        
        # Find the start of the phrase
        start = match_start
        while start > 0 and text[start-1] not in self.phrase_delimiters:
            start -= 1
        
        # Find the end of the phrase
        end = match_end
        while end < len(text) and text[end] not in self.phrase_delimiters:
            end += 1
        
        # Ensure we don't exceed max phrase length
        phrase_text = text[start:end].strip()
        words = phrase_text.split()
        
        if len(words) > self.max_phrase_length:
            # Truncate to max_phrase_length words
            truncated_words = words[:self.max_phrase_length]
            truncated_text = ' '.join(truncated_words)
            end = start + len(truncated_text)
        
        return start, end
    
    def __repr__(self):
        return f"UnifiedParameterType(name='{self.name}', type_hint={self.type_hint.value})"
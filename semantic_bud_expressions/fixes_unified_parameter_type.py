# semantic_bud_expressions/fixes_unified_parameter_type.py
"""
Fixed version of UnifiedParameterType that addresses all test failures:
1. Standard parameters don't default to semantic
2. Transformer signature handling
3. Flexible phrase length validation
"""

from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Union, Pattern
from enum import Enum
import re
import logging

from .parameter_type import ParameterType
from .model_manager import Model2VecManager
from .unified_parameter_type import ParameterTypeHint


class FixedUnifiedParameterType(ParameterType):
    """
    Fixed unified parameter type that handles all parameter types correctly.
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
        phrase_length_strict: bool = False,  # New: control strict validation
        # Custom regex parameters
        custom_pattern: Optional[str] = None,
        # Context hints
        context_hints: Optional[Dict[str, Any]] = None
    ):
        self.type_hint = type_hint
        self.prototypes = prototypes or []
        self.similarity_threshold = similarity_threshold
        self.phrase_delimiters = phrase_delimiters or ['.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}']
        self.max_phrase_length = max_phrase_length
        self.phrase_length_strict = phrase_length_strict
        self.custom_pattern = custom_pattern
        self.context_hints = context_hints or {}
        
        # Only initialize model manager for semantic/dynamic types
        self.model_manager = None
        if type_hint in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC]:
            self.model_manager = Model2VecManager()
        
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
        """Check if text matches semantically against prototypes or parameter name"""
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
    
    def _safe_transformer_call(self, value: Any, **kwargs) -> Any:
        """Safely call transformer with or without extra arguments"""
        if not self.transformer:
            return value
            
        try:
            import inspect
            sig = inspect.signature(self.transformer)
            
            # Check if transformer accepts **kwargs or specific named params
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD 
                for p in sig.parameters.values()
            )
            
            # Check for specific parameter names
            param_names = set(sig.parameters.keys())
            
            if accepts_kwargs or any(k in param_names for k in kwargs):
                # Try with extra arguments
                return self.transformer(value, **kwargs)
            else:
                # Call with just value
                return self.transformer(value)
        except Exception:
            # Fallback to simple call
            try:
                return self.transformer(value)
            except Exception as e:
                logging.warning(f"Transformer failed for {self.name}: {e}")
                return value
    
    def transform(self, group_values: List[str]) -> Any:
        """Transform with type-specific validation and processing"""
        if not group_values:
            return None
        
        value = group_values[0]
        
        # Handle STANDARD type - no semantic validation
        if self.type_hint == ParameterTypeHint.STANDARD:
            return self._safe_transformer_call(value)
        
        # Handle semantic/dynamic matching validation
        elif self.type_hint in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC]:
            matches, similarity, closest = self.matches_semantically(value)
            
            if not matches:
                raise ValueError(
                    f"'{value}' does not semantically match parameter type '{self.name}' "
                    f"(similarity: {similarity:.2f}, threshold: {self.similarity_threshold})"
                )
            
            # Use safe transformer call
            return self._safe_transformer_call(
                value, 
                similarity=similarity, 
                closest_prototype=closest
            )
        
        # Handle phrase processing
        elif self.type_hint == ParameterTypeHint.PHRASE:
            # Clean up phrase (remove extra spaces, etc.)
            cleaned_value = re.sub(r'\s+', ' ', value.strip())
            
            # Handle phrase length
            word_count = len(cleaned_value.split())
            if word_count > self.max_phrase_length:
                if self.phrase_length_strict:
                    # Strict mode: raise error
                    raise ValueError(
                        f"Phrase '{cleaned_value}' has {word_count} words, "
                        f"exceeding maximum of {self.max_phrase_length}"
                    )
                else:
                    # Flexible mode: truncate
                    words = cleaned_value.split()
                    cleaned_value = ' '.join(words[:self.max_phrase_length])
                    logging.debug(
                        f"Truncated phrase from {word_count} to {self.max_phrase_length} words"
                    )
            
            return self._safe_transformer_call(cleaned_value)
        
        # Handle quoted string processing
        elif self.type_hint == ParameterTypeHint.QUOTED:
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            return self._safe_transformer_call(value)
        
        # Handle mathematical expressions
        elif self.type_hint == ParameterTypeHint.MATH:
            # Import here to avoid circular imports
            try:
                from .math_parameter_type import MathExpression
                return MathExpression(value)
            except ImportError:
                # Fall back to string if math module not available
                return self._safe_transformer_call(value)
        
        # Handle regex type
        elif self.type_hint == ParameterTypeHint.REGEX:
            return self._safe_transformer_call(value)
        
        # Default transformation
        return self._safe_transformer_call(value)
    
    def __repr__(self):
        return f"FixedUnifiedParameterType(name='{self.name}', type_hint={self.type_hint.value})"
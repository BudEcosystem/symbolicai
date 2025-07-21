# semantic_bud_expressions/enhanced_unified_parameter_type.py
from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Union, Pattern, Tuple
from enum import Enum
import re
import logging

from .unified_parameter_type import UnifiedParameterType, ParameterTypeHint
from .faiss_phrase_matcher import FAISSPhraseMatcher, PhraseCandidate
from .model_manager import Model2VecManager
from .faiss_manager import FAISSManager
import numpy as np


class EnhancedUnifiedParameterType(UnifiedParameterType):
    """
    Enhanced unified parameter type with FAISS integration for better multi-word phrase matching.
    
    Key improvements:
    - FAISS-powered phrase matching for large vocabularies
    - Intelligent phrase boundary detection
    - Semantic phrase matching with similarity scoring
    - Context-aware phrase extraction
    """
    
    # Class-level FAISS phrase matcher (shared across instances)
    _phrase_matcher: Optional[FAISSPhraseMatcher] = None
    _faiss_manager: Optional[FAISSManager] = None
    
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
        # FAISS-specific parameters
        use_faiss: bool = True,
        faiss_auto_threshold: int = 100,
        phrase_similarity_threshold: float = 0.4,
        use_context_scoring: bool = True,
        # Custom regex parameters
        custom_pattern: Optional[str] = None,
        # Context hints
        context_hints: Optional[Dict[str, Any]] = None
    ):
        """
        Create an enhanced unified parameter type with FAISS support.
        
        Additional Args:
            use_faiss: Whether to use FAISS for similarity search
            faiss_auto_threshold: Number of prototypes above which to auto-enable FAISS
            phrase_similarity_threshold: Minimum similarity for phrase matching
            use_context_scoring: Whether to use context scoring for phrases
        """
        self.use_faiss = use_faiss
        self.faiss_auto_threshold = faiss_auto_threshold
        self.phrase_similarity_threshold = phrase_similarity_threshold
        self.use_context_scoring = use_context_scoring
        
        # Initialize parent class
        super().__init__(
            name=name,
            type_hint=type_hint,
            regexp=regexp,
            type=type,
            transformer=transformer,
            use_for_snippets=use_for_snippets,
            prefer_for_regexp_match=prefer_for_regexp_match,
            prototypes=prototypes,
            similarity_threshold=similarity_threshold,
            phrase_delimiters=phrase_delimiters,
            max_phrase_length=max_phrase_length,
            custom_pattern=custom_pattern,
            context_hints=context_hints
        )
        
        # Initialize FAISS components if needed
        self._init_faiss_components()
        
        # For semantic types with many prototypes, use FAISS
        if self.type_hint in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.PHRASE]:
            self._setup_faiss_index()
    
    def _init_faiss_components(self):
        """Initialize FAISS components (class-level)"""
        if not EnhancedUnifiedParameterType._faiss_manager and self.use_faiss:
            EnhancedUnifiedParameterType._faiss_manager = FAISSManager()
            
        if not EnhancedUnifiedParameterType._phrase_matcher and self.model_manager:
            EnhancedUnifiedParameterType._phrase_matcher = FAISSPhraseMatcher(
                model_manager=self.model_manager,
                max_phrase_length=self.max_phrase_length,
                similarity_threshold=self.phrase_similarity_threshold,
                use_context_scoring=self.use_context_scoring,
                phrase_delimiters=self.phrase_delimiters
            )
    
    def _setup_faiss_index(self):
        """Setup FAISS index for prototypes if applicable"""
        if not self.use_faiss or not self._faiss_manager:
            return
            
        # For semantic types with many prototypes
        if (self.type_hint == ParameterTypeHint.SEMANTIC and 
            self.prototypes and 
            len(self.prototypes) >= self.faiss_auto_threshold):
            
            # Create FAISS index for semantic prototypes
            dimension = 256  # Model2Vec dimension
            self._faiss_index = self._faiss_manager.create_auto_index(
                dimension, len(self.prototypes)
            )
            
            # Add prototype embeddings
            self._ensure_embeddings()
            if self._prototype_embeddings is not None:
                self._faiss_manager.add_embeddings(
                    self._faiss_index,
                    self._prototype_embeddings
                )
                
                logging.info(f"Created FAISS index for {self.name} with {len(self.prototypes)} prototypes")
        
        # For phrase types, add known phrases to matcher
        elif self.type_hint == ParameterTypeHint.PHRASE and self.prototypes:
            if self._phrase_matcher:
                self._phrase_matcher.add_known_phrases(
                    self.prototypes,
                    categories=[self.name] * len(self.prototypes)
                )
    
    def _generate_regexp(self) -> str:
        """Generate enhanced regex pattern based on type hint"""
        if self.type_hint == ParameterTypeHint.PHRASE:
            # Enhanced phrase regex that's more flexible
            # This allows the matcher to capture a wider range initially
            # and then use FAISS to find the best phrase boundaries
            delimiter_pattern = '|'.join(re.escape(d) for d in self.phrase_delimiters)
            
            # Match sequences of non-delimiter characters and spaces
            # This is more permissive to allow FAISS phrase matcher to work
            return rf'(?:(?!(?:{delimiter_pattern}))[^\s])+(?:\s+(?:(?!(?:{delimiter_pattern}))[^\s])+)*'
        else:
            # Use parent implementation for other types
            return super()._generate_regexp()
    
    def matches_semantically(self, text: str) -> tuple[bool, float, str]:
        """
        Enhanced semantic matching with FAISS support.
        
        Returns:
            Tuple of (matches, similarity_score, closest_prototype)
        """
        if self.type_hint not in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC]:
            return False, 0.0, ""
        
        if not self.model_manager:
            return False, 0.0, ""
        
        # Use FAISS if available and applicable
        if (self.use_faiss and 
            hasattr(self, '_faiss_index') and 
            self._faiss_index is not None):
            
            # Get embedding for input text
            text_embedding = self.model_manager.embed_sync([text])[0]
            
            # Search in FAISS
            distances, indices = self._faiss_manager.search(
                self._faiss_index,
                text_embedding.reshape(1, -1),
                k=1
            )
            
            # Get best match
            best_idx = indices[0, 0]
            best_similarity = float(distances[0, 0])
            closest_prototype = self.prototypes[best_idx] if best_idx < len(self.prototypes) else ""
            
            matches = best_similarity >= self.similarity_threshold
            return matches, best_similarity, closest_prototype
        else:
            # Fall back to parent implementation
            return super().matches_semantically(text)
    
    def match_phrase(self, text: str, start_pos: int = 0) -> Optional[Tuple[str, int, int]]:
        """
        Match a phrase using FAISS phrase matcher.
        
        Args:
            text: Text to match in
            start_pos: Starting position
            
        Returns:
            Tuple of (matched_phrase, start, end) or None
        """
        if not self._phrase_matcher:
            return None
            
        result = self._phrase_matcher.match_phrase_in_text(
            text,
            start_pos,
            parameter_name=self.name
        )
        
        return result
    
    def transform(self, group_values: List[str]) -> Any:
        """Enhanced transform with better phrase handling"""
        if not group_values:
            return None
        
        value = group_values[0]
        
        # Handle phrase processing with FAISS
        if self.type_hint == ParameterTypeHint.PHRASE:
            # If we have a phrase matcher, try to refine the match
            if self._phrase_matcher and hasattr(self, '_last_match_context'):
                # Try to get better phrase boundaries
                context = self._last_match_context
                refined = self._phrase_matcher.find_best_phrase_match(
                    context['full_text'],
                    context['start_pos'],
                    parameter_name=self.name
                )
                
                if refined:
                    value = refined.text
            
            # Clean up phrase
            cleaned_value = re.sub(r'\s+', ' ', value.strip())
            
            # Don't enforce strict length limits if using FAISS
            # The phrase matcher already handles this intelligently
            if self.use_faiss and self._phrase_matcher:
                # Just log if it's longer than expected
                word_count = len(cleaned_value.split())
                if word_count > self.max_phrase_length:
                    logging.debug(
                        f"Phrase '{cleaned_value}' has {word_count} words "
                        f"(soft limit: {self.max_phrase_length})"
                    )
            else:
                # Strict validation without FAISS
                word_count = len(cleaned_value.split())
                if word_count > self.max_phrase_length:
                    # Truncate to max length
                    words = cleaned_value.split()
                    cleaned_value = ' '.join(words[:self.max_phrase_length])
            
            if self.transformer:
                return self.transformer(cleaned_value)
            else:
                return cleaned_value
        
        # For semantic/dynamic types, fix the transformer call
        elif self.type_hint in [ParameterTypeHint.SEMANTIC, ParameterTypeHint.DYNAMIC]:
            matches, similarity, closest = self.matches_semantically(value)
            
            if not matches:
                raise ValueError(
                    f"'{value}' does not semantically match parameter type '{self.name}' "
                    f"(similarity: {similarity:.2f}, threshold: {self.similarity_threshold})"
                )
            
            # Fix: Check if transformer accepts additional arguments
            if self.transformer:
                # Try to call with extra args, fall back to simple call
                try:
                    import inspect
                    sig = inspect.signature(self.transformer)
                    if len(sig.parameters) > 1:
                        return self.transformer(value, similarity=similarity, closest_prototype=closest)
                    else:
                        return self.transformer(value)
                except:
                    return self.transformer(value)
            else:
                return value
        
        # Use parent implementation for other types
        else:
            return super().transform(group_values)
    
    def get_phrase_boundaries(self, text: str, match_start: int, match_end: int) -> tuple[int, int]:
        """
        Enhanced phrase boundary detection using FAISS.
        
        Args:
            text: The full text being matched
            match_start: Start position of the current match
            match_end: End position of the current match
            
        Returns:
            Tuple of (actual_start, actual_end) positions
        """
        if self.type_hint != ParameterTypeHint.PHRASE:
            return match_start, match_end
        
        # Use FAISS phrase matcher if available
        if self._phrase_matcher and self.use_faiss:
            result = self._phrase_matcher.match_phrase_in_text(
                text,
                match_start,
                parameter_name=self.name
            )
            
            if result:
                _, start, end = result
                return start, end
        
        # Fall back to parent implementation
        return super().get_phrase_boundaries(text, match_start, match_end)
    
    def set_match_context(self, full_text: str, start_pos: int, end_pos: int):
        """Store context for later phrase refinement"""
        self._last_match_context = {
            'full_text': full_text,
            'start_pos': start_pos,
            'end_pos': end_pos
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parameter type statistics"""
        stats = {
            'name': self.name,
            'type_hint': self.type_hint.value,
            'use_faiss': self.use_faiss,
            'num_prototypes': len(self.prototypes) if self.prototypes else 0,
        }
        
        if hasattr(self, '_faiss_index') and self._faiss_index:
            stats['faiss_index'] = {
                'type': self._faiss_manager.get_index_type(self._faiss_index),
                'num_vectors': self._faiss_index.ntotal if hasattr(self._faiss_index, 'ntotal') else 0
            }
        
        if self._phrase_matcher:
            stats['phrase_matcher'] = self._phrase_matcher.get_statistics()
        
        return stats
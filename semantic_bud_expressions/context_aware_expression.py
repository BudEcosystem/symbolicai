"""
Context-aware expression matching that validates surrounding text semantically.
"""

import re
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
from .unified_expression import UnifiedBudExpression
from .unified_registry import UnifiedParameterTypeRegistry
from .model_manager import Model2VecManager


class ContextMatch:
    """Represents a context-aware match result"""
    
    def __init__(
        self,
        matched_text: str,
        parameters: Dict[str, Any],
        start_pos: int,
        end_pos: int,
        context_text: str,
        context_similarity: float,
        full_text: str
    ):
        self.matched_text = matched_text
        self.parameters = parameters
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.context_text = context_text
        self.context_similarity = context_similarity
        self.full_text = full_text
    
    def __repr__(self):
        return (
            f"ContextMatch(text='{self.matched_text}', "
            f"similarity={self.context_similarity:.2f}, "
            f"context='{self.context_text[:50]}...')"
        )


class ContextAwareExpression:
    """
    Expression that matches based on both pattern and semantic context.
    
    First finds expression matches, then validates that the preceding context
    semantically matches the expected context.
    """
    
    def __init__(
        self,
        expression: str,
        expected_context: str,
        context_threshold: float = 0.5,
        context_window: Union[int, str] = 50,
        context_comparison: str = 'direct',
        registry: Optional[UnifiedParameterTypeRegistry] = None
    ):
        """
        Initialize context-aware expression.
        
        Args:
            expression: The pattern expression (e.g., "I {emotion} {vehicle}")
            expected_context: Expected semantic context (e.g., "cars and automotive")
            context_threshold: Minimum similarity for context match (0-1)
            context_window: How much context to extract:
                - int: Number of words before match
                - 'sentence': Previous sentence
                - 'paragraph': Current paragraph
                - 'auto': Automatically determine
            context_comparison: How to compare contexts:
                - 'direct': Direct embedding comparison
                - 'chunked_mean': Average of sentence embeddings
            registry: Parameter type registry (creates default if None)
        """
        self.expected_context = expected_context
        self.context_threshold = context_threshold
        self.context_window = context_window
        self.context_comparison = context_comparison
        
        # Initialize registry and model
        self.registry = registry or UnifiedParameterTypeRegistry()
        if not self.registry.model_manager:
            self.registry.initialize_model()
        
        # Create the underlying expression
        self.expression = UnifiedBudExpression(expression, self.registry)
        self.model_manager = self.registry.model_manager
        
        # Cache for embeddings
        self._expected_context_embedding = None
        self._context_cache = {}
        self._last_extracted_context = None
    
    def _get_expected_embedding(self) -> np.ndarray:
        """Get or compute expected context embedding"""
        if self._expected_context_embedding is None:
            self._expected_context_embedding = self.model_manager.embed_sync(
                [self.expected_context]
            )[0]
        return self._expected_context_embedding
    
    def extract_context(
        self,
        text: str,
        match_start: Optional[int] = None,
        position: str = 'before',
        window: Optional[Union[int, str]] = None
    ) -> str:
        """
        Extract context from text relative to match position.
        
        Args:
            text: Full text
            match_start: Start position of match (optional for some uses)
            position: Where to extract context ('before', 'sentence', 'auto')
            window: Override for context window
            
        Returns:
            Extracted context text
        """
        # Debug prints
        # print(f"DEBUG extract_context: text='{text[:50]}...', match_start={match_start}, position='{position}'")
        
        if match_start is None and position in ['before', 'auto']:
            # If no position given for before/auto, assume we want context from start
            match_start = len(text)
            
        window = window or self.context_window
        
        if position == 'before':
            # Extract text before the match
            before_text = text[:match_start].strip()
            
            if isinstance(window, int):
                # Extract N words
                words = before_text.split()
                context = ' '.join(words[-window:]) if words else ""
                
            elif window == 'sentence':
                # Extract previous sentence
                sentences = re.split(r'[.!?]+', before_text)
                context = sentences[-2].strip() if len(sentences) > 1 else sentences[-1].strip()
                
            elif window == 'paragraph':
                # Extract current paragraph
                paragraphs = before_text.split('\n\n')
                context = paragraphs[-1].strip()
                
            elif window == 'auto':
                # Intelligently determine context
                # Try sentence first, fallback to words
                sentences = re.split(r'[.!?]+', before_text)
                if len(sentences) > 1 and len(sentences[-2].strip()) > 20:
                    context = sentences[-2].strip()
                else:
                    words = before_text.split()
                    context = ' '.join(words[-30:]) if words else ""
            else:
                context = before_text
                
        elif position == 'sentence':
            # Extract containing or previous sentence
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if match_start is None:
                # No position given - return the previous sentence if multiple exist
                if len(sentences) >= 2:
                    # Return the previous sentence (second-to-last)
                    context = sentences[-2]
                elif len(sentences) >= 1:
                    context = sentences[0]
                else:
                    context = ""
            else:
                # Find the sentence before the one containing the match
                current_pos = 0
                match_sentence_idx = -1
                
                for i, sentence in enumerate(sentences):
                    sentence_end = current_pos + len(sentence) + 1  # +1 for punctuation
                    if current_pos <= match_start <= sentence_end:
                        match_sentence_idx = i
                        break
                    current_pos = sentence_end + 1  # +1 for space
                
                # Return the previous sentence or the matching sentence if it's the first
                if match_sentence_idx > 0:
                    context = sentences[match_sentence_idx - 1]
                elif match_sentence_idx == 0 and len(sentences) > 1:
                    context = sentences[1]  # Use next sentence as context
                elif len(sentences) > 0:
                    context = sentences[0]
                else:
                    context = ""
                
        elif position == 'auto':
            # Try different strategies
            context = self.extract_context(text, match_start, 'before', 'auto')
            
        else:
            context = ""
            
        return context
    
    def compare_context_direct(
        self,
        extracted_context: str,
        expected_context: str
    ) -> float:
        """
        Directly compare context embeddings with fallback strategies.
        
        Args:
            extracted_context: Extracted context text
            expected_context: Expected context text
            
        Returns:
            Similarity score (0-1)
        """
        if not extracted_context or not extracted_context.strip():
            return 0.0
        
        # Clean the context
        extracted_context = extracted_context.strip()
        
        try:
            # Get embeddings
            extracted_embedding = self.model_manager.embed_sync([extracted_context])[0]
            expected_embedding = self._get_expected_embedding()
            
            # Compute similarity
            similarity = self.model_manager.cosine_similarity(
                extracted_embedding,
                expected_embedding
            )
            
            # If similarity is low, try keyword matching as fallback
            if similarity < 0.3:
                # Check for keyword overlap
                extracted_words = set(extracted_context.lower().split())
                expected_words = set(expected_context.lower().split())
                
                if extracted_words & expected_words:  # If there's any overlap
                    # Boost similarity based on keyword overlap
                    overlap_ratio = len(extracted_words & expected_words) / len(expected_words)
                    similarity = max(similarity, overlap_ratio * 0.6)
            
            return float(similarity)
            
        except Exception as e:
            # Fallback to keyword matching if embedding fails
            extracted_words = set(extracted_context.lower().split())
            expected_words = set(expected_context.lower().split())
            
            if not expected_words:
                return 0.0
                
            overlap_ratio = len(extracted_words & expected_words) / len(expected_words)
            return overlap_ratio
    
    def compare_context_chunked(
        self,
        extracted_context: str,
        expected_context: str
    ) -> float:
        """
        Compare contexts by chunking into sentences and averaging.
        
        Args:
            extracted_context: Extracted context text
            expected_context: Expected context text
            
        Returns:
            Similarity score (0-1)
        """
        if not extracted_context:
            return 0.0
            
        # Split into sentences
        sentences = re.split(r'[.!?]+', extracted_context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return self.compare_context_direct(extracted_context, expected_context)
        
        # Get embeddings for each sentence
        sentence_embeddings = self.model_manager.embed_sync(sentences)
        
        # Average embeddings
        avg_embedding = np.mean(sentence_embeddings, axis=0)
        
        # Compare with expected
        expected_embedding = self._get_expected_embedding()
        similarity = self.model_manager.cosine_similarity(
            avg_embedding,
            expected_embedding
        )
        
        return float(similarity)
    
    def _compare_context(self, extracted_context: str) -> float:
        """Compare extracted context with expected context"""
        # Store last extracted context for debugging
        self._last_extracted_context = extracted_context
        
        if self.context_comparison == 'chunked_mean':
            return self.compare_context_chunked(extracted_context, self.expected_context)
        else:
            return self.compare_context_direct(extracted_context, self.expected_context)
    
    def _extract_parameters(self, result: List) -> Dict[str, Any]:
        """Extract parameters from match result"""
        parameters = {}
        
        # Get parameter names from the expression pattern
        param_names = re.findall(r'\{(\w+)(?::\w+)?\}', self.expression.original_expression)
        
        # Map arguments to parameter names
        for i, arg in enumerate(result):
            if i < len(param_names):
                param_name = param_names[i]
                try:
                    parameters[param_name] = arg.value
                except Exception:
                    # If value extraction fails, try to get the matched text
                    if hasattr(arg, 'group') and arg.group:
                        parameters[param_name] = arg.group.value
                    else:
                        parameters[param_name] = None
        
        return parameters
    
    def match_with_context(self, text: str) -> Optional[ContextMatch]:
        """
        Find first match that also matches context semantically.
        
        Args:
            text: Text to search in
            
        Returns:
            ContextMatch if found with matching context, None otherwise
        """
        if not text or not text.strip():
            return None
            
        try:
            # Use regex to find all potential matches
            import re
            pattern = self.expression.regexp
            
            # Remove anchors if present to allow finding matches within text
            if pattern.startswith('^'):
                pattern = pattern[1:]
            if pattern.endswith('$'):
                pattern = pattern[:-1]
            
            for match_obj in re.finditer(pattern, text):
                match_start = match_obj.start()
                match_text = match_obj.group(0)
                
                # Try to parse this match
                result = self.expression.match(match_text)
                if not result:
                    continue
                    
                # Extract context before this match
                context = self.extract_context(text, match_start)
                
                # Check context similarity
                similarity = self._compare_context(context)
                
                if similarity >= self.context_threshold:
                    # Build parameters dict
                    parameters = self._extract_parameters(result)
                    
                    return ContextMatch(
                        matched_text=match_text,
                        parameters=parameters,
                        start_pos=match_start,
                        end_pos=match_obj.end(),
                        context_text=context,
                        context_similarity=similarity,
                        full_text=text
                    )
            
            return None
            
        except Exception as e:
            # Handle any errors gracefully
            return None
    
    def find_all_with_context(self, text: str) -> List[ContextMatch]:
        """
        Find all matches that also match context semantically.
        
        Args:
            text: Text to search in
            
        Returns:
            List of ContextMatch objects
        """
        if not text or not text.strip():
            return []
            
        results = []
        
        try:
            # Use regex to find all potential matches
            import re
            pattern = self.expression.regexp
            
            # Remove anchors if present to allow finding matches within text
            if pattern.startswith('^'):
                pattern = pattern[1:]
            if pattern.endswith('$'):
                pattern = pattern[:-1]
            
            for match_obj in re.finditer(pattern, text):
                match_start = match_obj.start()
                match_text = match_obj.group(0)
                
                # Try to parse this match
                result = self.expression.match(match_text)
                if not result:
                    continue
                    
                # Extract and check context
                context = self.extract_context(text, match_start)
                similarity = self._compare_context(context)
                
                if similarity >= self.context_threshold:
                    # Build parameters dict
                    parameters = self._extract_parameters(result)
                    
                    results.append(ContextMatch(
                        matched_text=match_text,
                        parameters=parameters,
                        start_pos=match_start,
                        end_pos=match_obj.end(),
                        context_text=context,
                        context_similarity=similarity,
                        full_text=text
                    ))
            
            return results
            
        except Exception as e:
            # Handle any errors gracefully
            return results
    
    def get_extracted_context(self) -> Optional[str]:
        """Get the last extracted context (for debugging)"""
        return self._last_extracted_context
    
    def match(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Backward compatibility method.
        
        Args:
            text: Text to match
            
        Returns:
            Parameters dict if match found, None otherwise
        """
        result = self.match_with_context(text)
        return result.parameters if result else None
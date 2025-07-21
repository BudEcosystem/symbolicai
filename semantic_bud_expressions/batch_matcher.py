"""
Batch matching optimization for semantic expressions.
Computes embeddings for entire input text once and reuses them.
"""
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from .model_manager import Model2VecManager
from .semantic_cache import SemanticCache


class BatchMatcher:
    """Optimized matcher that batch-computes embeddings for efficiency"""
    
    def __init__(self, model_manager: Model2VecManager):
        self.model_manager = model_manager
        self.phrase_cache = {}  # Cache for phrase embeddings within a match
        
    def extract_all_phrases(self, text: str, max_phrase_length: int = 10) -> List[str]:
        """
        Extract all possible phrases from the text.
        
        Args:
            text: The input text
            max_phrase_length: Maximum number of words in a phrase
            
        Returns:
            List of all possible phrases
        """
        words = text.split()
        phrases = []
        
        # Extract all possible phrases up to max_phrase_length
        for start in range(len(words)):
            for length in range(1, min(max_phrase_length + 1, len(words) - start + 1)):
                phrase = ' '.join(words[start:start + length])
                phrases.append(phrase)
                
        # Also add the full text if not already included
        if text not in phrases:
            phrases.append(text)
            
        return phrases
    
    def batch_compute_embeddings(self, phrases: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for all phrases in a single batch.
        
        Args:
            phrases: List of phrases to compute embeddings for
            
        Returns:
            Dictionary mapping phrases to their embeddings
        """
        if not phrases:
            return {}
            
        # Remove duplicates while preserving order
        unique_phrases = list(dict.fromkeys(phrases))
        
        # Batch compute embeddings
        embeddings = self.model_manager.embed_sync(unique_phrases)
        
        # Create mapping
        phrase_embeddings = {
            phrase: embedding 
            for phrase, embedding in zip(unique_phrases, embeddings)
        }
        
        return phrase_embeddings
    
    def match_with_cache(
        self, 
        text: str, 
        parameter_types: List['SemanticParameterType'],
        max_phrase_length: int = 10
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[str, float]]]]:
        """
        Match text against parameter types using batch-computed embeddings.
        
        Args:
            text: The input text to match
            parameter_types: List of semantic parameter types to match against
            max_phrase_length: Maximum phrase length to consider
            
        Returns:
            Tuple of (phrase_embeddings, matches_by_type)
            where matches_by_type maps parameter type names to lists of (phrase, similarity) tuples
        """
        # Extract all possible phrases
        phrases = self.extract_all_phrases(text, max_phrase_length)
        
        # Batch compute embeddings for all phrases
        phrase_embeddings = self.batch_compute_embeddings(phrases)
        
        # Store in cache for reuse
        self.phrase_cache.update(phrase_embeddings)
        
        # Match against each parameter type
        matches_by_type = {}
        
        for param_type in parameter_types:
            if not hasattr(param_type, '_prototype_embeddings') or param_type._prototype_embeddings is None:
                continue
                
            type_matches = []
            
            # Check each phrase against this parameter type
            for phrase, phrase_embedding in phrase_embeddings.items():
                max_similarity = 0.0
                best_prototype = None
                
                # Compare with all prototypes
                for i, prototype_embedding in enumerate(param_type._prototype_embeddings):
                    similarity = self.model_manager.cosine_similarity(
                        phrase_embedding, 
                        prototype_embedding
                    )
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_prototype = param_type.prototypes[i] if hasattr(param_type, 'prototypes') else None
                
                # Store if above threshold
                if max_similarity >= getattr(param_type, 'similarity_threshold', 0.7):
                    type_matches.append((phrase, max_similarity, best_prototype))
            
            # Sort by similarity score
            type_matches.sort(key=lambda x: x[1], reverse=True)
            matches_by_type[param_type.name] = type_matches
        
        return phrase_embeddings, matches_by_type
    
    def clear_cache(self):
        """Clear the phrase cache"""
        self.phrase_cache.clear()


class OptimizedSemanticMatcher:
    """
    Optimized semantic matcher that uses batch computation
    and caching for improved performance.
    """
    
    def __init__(self, model_manager: Model2VecManager):
        self.model_manager = model_manager
        self.batch_matcher = BatchMatcher(model_manager)
        self._match_cache = {}  # Cache full expression matches
        
    def match_expression(
        self, 
        expression_pattern: str,
        input_text: str,
        parameter_types: List['SemanticParameterType']
    ) -> Optional[Dict[str, str]]:
        """
        Match an expression pattern against input text using batch optimization.
        
        Args:
            expression_pattern: Pattern like "I love {fruit}"
            input_text: Text to match like "I love apples"
            parameter_types: List of semantic parameter types
            
        Returns:
            Dictionary of matched parameters or None if no match
        """
        # Check cache first
        cache_key = (expression_pattern, input_text)
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]
        
        # Extract parameter names from pattern
        import re
        param_pattern = re.compile(r'\{(\w+)(?::\w+)?\}')
        param_names = param_pattern.findall(expression_pattern)
        
        # Get relevant parameter types
        relevant_types = [
            pt for pt in parameter_types 
            if hasattr(pt, 'name') and pt.name in param_names
        ]
        
        if not relevant_types:
            return None
        
        # Batch compute embeddings for all phrases
        phrase_embeddings, matches_by_type = self.batch_matcher.match_with_cache(
            input_text, relevant_types
        )
        
        # Try to construct a match
        # This is a simplified version - in practice, you'd need more sophisticated matching
        result = {}
        for param_name in param_names:
            type_matches = matches_by_type.get(param_name, [])
            if type_matches:
                # Use the best match
                result[param_name] = type_matches[0][0]  # phrase
            else:
                # No match found
                self._match_cache[cache_key] = None
                return None
        
        self._match_cache[cache_key] = result
        return result
    
    def clear_caches(self):
        """Clear all caches"""
        self.batch_matcher.clear_cache()
        self._match_cache.clear()
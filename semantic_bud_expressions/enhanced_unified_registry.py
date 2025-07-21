# semantic_bud_expressions/enhanced_unified_registry.py
from typing import Dict, List, Optional, Union
from .unified_registry import UnifiedParameterTypeRegistry
from .enhanced_unified_parameter_type import EnhancedUnifiedParameterType
from .unified_parameter_type import ParameterTypeHint
from .parameter_type import ParameterType
import logging


class EnhancedUnifiedParameterTypeRegistry(UnifiedParameterTypeRegistry):
    """
    Enhanced registry with FAISS integration for better multi-word phrase matching.
    
    Key improvements:
    - Automatic FAISS enabling for large prototype sets
    - Enhanced phrase matching with semantic similarity
    - Shared FAISS resources across parameter types
    - Better performance for large vocabularies
    """
    
    def __init__(
        self,
        use_faiss: bool = True,
        faiss_auto_threshold: int = 100,
        phrase_similarity_threshold: float = 0.4
    ):
        """
        Initialize enhanced registry.
        
        Args:
            use_faiss: Whether to use FAISS for similarity search
            faiss_auto_threshold: Auto-enable FAISS above this prototype count
            phrase_similarity_threshold: Minimum similarity for phrase matching
        """
        super().__init__()
        self.use_faiss = use_faiss
        self.faiss_auto_threshold = faiss_auto_threshold
        self.phrase_similarity_threshold = phrase_similarity_threshold
        
        # Track FAISS-enabled types
        self._faiss_enabled_types = set()
    
    def create_semantic_parameter_type(
        self,
        name: str,
        prototypes: List[str],
        similarity_threshold: float = 0.7,
        use_faiss: Optional[bool] = None,
        **kwargs
    ) -> EnhancedUnifiedParameterType:
        """Create an enhanced semantic parameter type with FAISS support"""
        # Auto-enable FAISS for large prototype sets
        if use_faiss is None:
            use_faiss = self.use_faiss and len(prototypes) >= self.faiss_auto_threshold
        
        unified_type = EnhancedUnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.SEMANTIC,
            prototypes=prototypes,
            similarity_threshold=similarity_threshold,
            use_faiss=use_faiss,
            faiss_auto_threshold=self.faiss_auto_threshold,
            **kwargs
        )
        
        self.define_unified_parameter_type(unified_type)
        
        if use_faiss:
            self._faiss_enabled_types.add(name)
            logging.info(f"FAISS enabled for semantic type '{name}' with {len(prototypes)} prototypes")
        
        return unified_type
    
    def create_phrase_parameter_type(
        self,
        name: str,
        max_phrase_length: int = 10,
        phrase_delimiters: Optional[List[str]] = None,
        known_phrases: Optional[List[str]] = None,
        use_faiss: Optional[bool] = None,
        use_context_scoring: bool = True,
        **kwargs
    ) -> EnhancedUnifiedParameterType:
        """
        Create an enhanced phrase parameter type with FAISS support.
        
        Args:
            name: Parameter type name
            max_phrase_length: Maximum words in a phrase
            phrase_delimiters: Custom phrase boundary delimiters
            known_phrases: List of known valid phrases
            use_faiss: Whether to use FAISS (auto-enabled if known_phrases provided)
            use_context_scoring: Whether to use context for phrase scoring
            **kwargs: Additional arguments
        """
        if phrase_delimiters is None:
            phrase_delimiters = self._phrase_config['phrase_delimiters']
        
        # Auto-enable FAISS if known phrases are provided
        if use_faiss is None:
            use_faiss = self.use_faiss and known_phrases is not None
        
        unified_type = EnhancedUnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.PHRASE,
            max_phrase_length=max_phrase_length,
            phrase_delimiters=phrase_delimiters,
            prototypes=known_phrases,  # Use as prototypes for FAISS
            use_faiss=use_faiss,
            phrase_similarity_threshold=self.phrase_similarity_threshold,
            use_context_scoring=use_context_scoring,
            **kwargs
        )
        
        self.define_unified_parameter_type(unified_type)
        
        if use_faiss and known_phrases:
            self._faiss_enabled_types.add(name)
            logging.info(f"FAISS enabled for phrase type '{name}' with {len(known_phrases)} known phrases")
        
        return unified_type
    
    def create_semantic_phrase_parameter_type(
        self,
        name: str,
        semantic_categories: List[str],
        max_phrase_length: int = 10,
        similarity_threshold: float = 0.5,
        **kwargs
    ) -> EnhancedUnifiedParameterType:
        """
        Create a parameter type that combines semantic and phrase matching.
        E.g., matches "red sports car" where "red" is semantic and the whole is a phrase.
        
        Args:
            name: Parameter type name
            semantic_categories: Semantic categories for the phrases
            max_phrase_length: Maximum phrase length
            similarity_threshold: Semantic similarity threshold
            **kwargs: Additional arguments
        """
        # This is a phrase type with semantic matching enabled
        unified_type = EnhancedUnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.PHRASE,
            prototypes=semantic_categories,
            max_phrase_length=max_phrase_length,
            similarity_threshold=similarity_threshold,
            use_faiss=True,  # Always use FAISS for semantic phrases
            use_context_scoring=True,
            **kwargs
        )
        
        self.define_unified_parameter_type(unified_type)
        self._faiss_enabled_types.add(name)
        
        return unified_type
    
    def create_dynamic_parameter_type(
        self,
        name: str,
        similarity_threshold: Optional[float] = None,
        use_faiss: Optional[bool] = None,
        **kwargs
    ) -> EnhancedUnifiedParameterType:
        """Create an enhanced dynamic parameter type"""
        if similarity_threshold is None:
            similarity_threshold = self._dynamic_threshold
        
        if use_faiss is None:
            use_faiss = self.use_faiss
        
        unified_type = EnhancedUnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.DYNAMIC,
            similarity_threshold=similarity_threshold,
            use_faiss=use_faiss,
            **kwargs
        )
        
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def lookup_by_type_name(self, name: str) -> Optional[ParameterType]:
        """Enhanced lookup that creates enhanced parameter types"""
        # First check unified cache
        if name in self._unified_types_cache:
            return self._unified_types_cache[name]
        
        # Then try the standard lookup
        param_type = super().lookup_by_type_name(name)
        
        if param_type is not None:
            return param_type
        
        # If not found and dynamic matching is enabled, create enhanced dynamic type
        if self._dynamic_matching_enabled:
            # Create new enhanced dynamic parameter type
            unified_type = EnhancedUnifiedParameterType(
                name=name,
                type_hint=ParameterTypeHint.DYNAMIC,
                similarity_threshold=self._dynamic_threshold,
                use_faiss=self.use_faiss
            )
            
            # Cache and register it
            self._unified_types_cache[name] = unified_type
            self.define_parameter_type(unified_type)
            
            return unified_type
        
        return None
    
    def get_faiss_statistics(self) -> Dict[str, any]:
        """Get statistics about FAISS usage"""
        stats = {
            'faiss_enabled': self.use_faiss,
            'faiss_auto_threshold': self.faiss_auto_threshold,
            'faiss_enabled_types': list(self._faiss_enabled_types),
            'total_faiss_types': len(self._faiss_enabled_types)
        }
        
        # Get detailed stats for each FAISS-enabled type
        for type_name in self._faiss_enabled_types:
            param_type = self.lookup_by_type_name(type_name)
            if param_type and hasattr(param_type, 'get_statistics'):
                stats[f'type_{type_name}'] = param_type.get_statistics()
        
        return stats
    
    def optimize_for_performance(self):
        """Optimize registry for performance by pre-computing indices"""
        logging.info("Optimizing registry for performance...")
        
        # Force FAISS index creation for all applicable types
        for name, param_type in self._unified_types_cache.items():
            if isinstance(param_type, EnhancedUnifiedParameterType):
                if hasattr(param_type, '_setup_faiss_index'):
                    param_type._setup_faiss_index()
    
    def add_phrases_to_type(
        self,
        type_name: str,
        phrases: List[str],
        categories: Optional[List[str]] = None
    ):
        """Add phrases to an existing phrase parameter type"""
        param_type = self.lookup_by_type_name(type_name)
        
        if (param_type and 
            isinstance(param_type, EnhancedUnifiedParameterType) and
            param_type.type_hint == ParameterTypeHint.PHRASE):
            
            # Add to prototypes
            if param_type.prototypes:
                param_type.prototypes.extend(phrases)
            else:
                param_type.prototypes = phrases
            
            # Update FAISS index
            if param_type._phrase_matcher:
                param_type._phrase_matcher.add_known_phrases(
                    phrases,
                    categories or [type_name] * len(phrases)
                )
            
            logging.info(f"Added {len(phrases)} phrases to type '{type_name}'")
        else:
            logging.warning(f"Type '{type_name}' is not a phrase parameter type")
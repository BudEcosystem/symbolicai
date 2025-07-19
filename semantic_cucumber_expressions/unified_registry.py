# semantic_cucumber_expressions/unified_registry.py
from typing import Dict, List, Optional, Union
from .semantic_registry import SemanticParameterTypeRegistry
from .unified_parameter_type import UnifiedParameterType, ParameterTypeHint
from .parameter_type import ParameterType
from .parameter_type_registry import ParameterTypeRegistry


class UnifiedParameterTypeRegistry(SemanticParameterTypeRegistry):
    """
    Enhanced parameter type registry that supports unified parameter types
    with type hints and advanced matching strategies.
    """
    
    def __init__(self):
        super().__init__()
        self._unified_types_cache = {}  # Cache for unified parameter types
        self._phrase_config = {
            'max_phrase_length': 10,
            'phrase_delimiters': ['.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}']
        }
    
    def define_unified_parameter_type(self, unified_type: UnifiedParameterType):
        """Define a unified parameter type"""
        self.define_parameter_type(unified_type)
        self._unified_types_cache[unified_type.name] = unified_type
    
    def create_semantic_parameter_type(
        self,
        name: str,
        prototypes: List[str],
        similarity_threshold: float = 0.7,
        **kwargs
    ) -> UnifiedParameterType:
        """Create a semantic parameter type using the unified system"""
        unified_type = UnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.SEMANTIC,
            prototypes=prototypes,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def create_phrase_parameter_type(
        self,
        name: str,
        max_phrase_length: int = 10,
        phrase_delimiters: Optional[List[str]] = None,
        **kwargs
    ) -> UnifiedParameterType:
        """Create a phrase parameter type for multi-word matching"""
        if phrase_delimiters is None:
            phrase_delimiters = self._phrase_config['phrase_delimiters']
        
        unified_type = UnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.PHRASE,
            max_phrase_length=max_phrase_length,
            phrase_delimiters=phrase_delimiters,
            **kwargs
        )
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def create_dynamic_parameter_type(
        self,
        name: str,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> UnifiedParameterType:
        """Create a dynamic semantic parameter type"""
        if similarity_threshold is None:
            similarity_threshold = self._dynamic_threshold
        
        unified_type = UnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.DYNAMIC,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def create_regex_parameter_type(
        self,
        name: str,
        pattern: str,
        **kwargs
    ) -> UnifiedParameterType:
        """Create a regex parameter type with custom pattern"""
        unified_type = UnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.REGEX,
            custom_pattern=pattern,
            **kwargs
        )
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def create_quoted_parameter_type(
        self,
        name: str,
        **kwargs
    ) -> UnifiedParameterType:
        """Create a quoted string parameter type"""
        unified_type = UnifiedParameterType(
            name=name,
            type_hint=ParameterTypeHint.QUOTED,
            **kwargs
        )
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def lookup_by_type_name(self, name: str) -> Optional[ParameterType]:
        """
        Enhanced lookup that supports unified parameter type creation.
        """
        # First check unified cache
        if name in self._unified_types_cache:
            return self._unified_types_cache[name]
        
        # Then try the standard lookup (but skip the parent's dynamic creation)
        param_type = ParameterTypeRegistry.lookup_by_type_name(self, name)
        
        if param_type is not None:
            return param_type
        
        # If not found and dynamic matching is enabled, create unified dynamic type
        if self._dynamic_matching_enabled:
            # Create new unified dynamic parameter type
            unified_type = UnifiedParameterType(
                name=name,
                type_hint=ParameterTypeHint.DYNAMIC,
                similarity_threshold=self._dynamic_threshold
            )
            
            # Cache and register it
            self._unified_types_cache[name] = unified_type
            self.define_parameter_type(unified_type)
            
            return unified_type
        
        return None
    
    def set_phrase_config(self, max_phrase_length: int = 10, phrase_delimiters: Optional[List[str]] = None):
        """Configure phrase matching parameters"""
        self._phrase_config['max_phrase_length'] = max_phrase_length
        if phrase_delimiters is not None:
            self._phrase_config['phrase_delimiters'] = phrase_delimiters
    
    def get_phrase_config(self) -> Dict[str, Union[int, List[str]]]:
        """Get current phrase matching configuration"""
        return self._phrase_config.copy()
    
    def create_parameter_type_from_hint(
        self,
        name: str,
        type_hint: ParameterTypeHint,
        **kwargs
    ) -> UnifiedParameterType:
        """Create a unified parameter type from a type hint"""
        unified_type = UnifiedParameterType(
            name=name,
            type_hint=type_hint,
            **kwargs
        )
        self.define_unified_parameter_type(unified_type)
        return unified_type
    
    def enable_advanced_phrase_matching(self, enabled: bool = True):
        """Enable or disable advanced phrase matching features"""
        # This could be extended to enable/disable specific phrase matching features
        pass
    
    def get_unified_parameter_types(self) -> Dict[str, UnifiedParameterType]:
        """Get all unified parameter types"""
        return self._unified_types_cache.copy()
    
    def clear_unified_cache(self):
        """Clear the unified parameter types cache"""
        self._unified_types_cache.clear()
    
    def migrate_semantic_type_to_unified(self, name: str) -> Optional[UnifiedParameterType]:
        """Migrate an existing semantic parameter type to unified system"""
        existing_type = self.lookup_by_type_name(name)
        
        if existing_type and hasattr(existing_type, 'prototypes'):
            # Create unified version
            unified_type = UnifiedParameterType(
                name=name,
                type_hint=ParameterTypeHint.SEMANTIC,
                prototypes=existing_type.prototypes,
                similarity_threshold=getattr(existing_type, 'similarity_threshold', 0.7)
            )
            
            # Replace the existing type
            self.define_unified_parameter_type(unified_type)
            return unified_type
        
        return None
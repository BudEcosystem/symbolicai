# semantic_bud_expressions/enhanced_unified_expression.py
from typing import Optional
from .unified_expression import UnifiedBudExpression
from .enhanced_unified_parameter_type import EnhancedUnifiedParameterType
from .unified_parameter_type import ParameterTypeHint


class EnhancedUnifiedBudExpression(UnifiedBudExpression):
    """
    Enhanced unified expression that properly handles parameter type creation
    to avoid treating standard parameters as semantic.
    """
    
    def _create_unified_parameter_type(self, param_name: str, type_hint: ParameterTypeHint) -> Optional[EnhancedUnifiedParameterType]:
        """Create an enhanced unified parameter type based on the type hint"""
        # Check if we have a predefined type that matches
        existing_type = self.parameter_type_registry.lookup_by_type_name(param_name)
        
        if type_hint == ParameterTypeHint.STANDARD:
            # For standard parameters, NEVER create dynamic/semantic types
            if existing_type:
                # If it's already a proper type, use it
                if not hasattr(existing_type, 'type_hint') or \
                   (hasattr(existing_type, 'type_hint') and 
                    existing_type.type_hint in [ParameterTypeHint.STANDARD, None]):
                    return None  # Use existing type
            
            # Always create a true STANDARD type, not DYNAMIC
            return EnhancedUnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.STANDARD,
                use_faiss=False  # Standard types don't need FAISS
            )
        
        elif type_hint == ParameterTypeHint.SEMANTIC:
            # Look for existing semantic parameter type
            if existing_type and hasattr(existing_type, 'prototypes'):
                # Convert existing semantic type to unified type
                return EnhancedUnifiedParameterType(
                    name=param_name,
                    type_hint=ParameterTypeHint.SEMANTIC,
                    prototypes=existing_type.prototypes,
                    similarity_threshold=getattr(existing_type, 'similarity_threshold', 0.7),
                    use_faiss=True
                )
            else:
                # Create new semantic type with parameter name as prototype
                return EnhancedUnifiedParameterType(
                    name=param_name,
                    type_hint=ParameterTypeHint.SEMANTIC,
                    prototypes=[param_name],
                    similarity_threshold=0.3,
                    use_faiss=False  # Single prototype doesn't need FAISS
                )
        
        elif type_hint == ParameterTypeHint.DYNAMIC:
            # Dynamic parameters should use semantic matching with the parameter name
            return EnhancedUnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.DYNAMIC,
                similarity_threshold=getattr(self.parameter_type_registry, '_dynamic_threshold', 0.3),
                use_faiss=False
            )
        
        elif type_hint == ParameterTypeHint.PHRASE:
            # Create phrase parameter type
            return EnhancedUnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.PHRASE,
                max_phrase_length=getattr(self.parameter_type_registry, '_phrase_config', {}).get('max_phrase_length', 10),
                phrase_delimiters=getattr(self.parameter_type_registry, '_phrase_config', {}).get('phrase_delimiters'),
                use_faiss=True
            )
        
        elif type_hint == ParameterTypeHint.REGEX:
            # Custom regex parameter
            return EnhancedUnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.REGEX,
                custom_pattern=r'\w+',  # Default pattern
                use_faiss=False
            )
        
        elif type_hint == ParameterTypeHint.QUOTED:
            # Quoted string parameter
            return EnhancedUnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.QUOTED,
                use_faiss=False
            )
        
        elif type_hint == ParameterTypeHint.MATH:
            # Math expression parameter
            return EnhancedUnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.MATH,
                use_faiss=False
            )
        
        # Default to standard
        return EnhancedUnifiedParameterType(
            name=param_name,
            type_hint=ParameterTypeHint.STANDARD,
            use_faiss=False
        )
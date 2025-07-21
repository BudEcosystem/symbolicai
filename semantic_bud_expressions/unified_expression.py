# semantic_bud_expressions/unified_expression.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Pattern
import re

from .ast import Node, NodeType
from .argument import Argument
from .group import Group
from .expression import budExpression
from .unified_expression_parser_simple import SimpleUnifiedExpressionParser
from .unified_parameter_type import UnifiedParameterType, ParameterTypeHint
from .parameter_type import ParameterType


class UnifiedBudExpression(budExpression):
    """
    A unified bud expression that supports multiple parameter types
    and matching strategies through type hints.
    
    Supports syntax like:
    - {param}            # Standard parameter
    - {param:semantic}   # Semantic parameter with similarity matching
    - {param:phrase}     # Multi-word phrase parameter
    - {param:dynamic}    # Dynamic semantic parameter
    - {param:regex}      # Custom regex parameter
    - {param:math}       # Mathematical expression
    - {param:quoted}     # Quoted string parameter
    """
    
    def __init__(self, expression: str, parameter_type_registry):
        """
        Initialize a unified bud expression.
        
        Args:
            expression: The expression string with optional type hints
            parameter_type_registry: Registry containing parameter types
        """
        self.original_expression = expression
        self.parameter_type_registry = parameter_type_registry
        self.parser = SimpleUnifiedExpressionParser()
        
        # Parse the expression to extract type hints
        self.parameter_metadata = self.parser.parse(expression)
        
        # Create unified parameter types based on type hints
        self._create_unified_parameter_types()
        
        # Initialize parent class with processed expression (type hints removed)
        processed_expression = self.parser.remove_type_hints(expression)
        super().__init__(processed_expression, parameter_type_registry)
    
    def _create_unified_parameter_types(self):
        """Create unified parameter types based on parsed type hints"""
        for param_name, metadata in self.parameter_metadata.items():
            type_hint = metadata['type_hint']
            
            # Check if parameter type already exists
            existing_type = self.parameter_type_registry.lookup_by_type_name(param_name)
            
            if existing_type is None:
                # Create new unified parameter type
                unified_type = self._create_unified_parameter_type(param_name, type_hint)
                
                if unified_type:
                    # Register the new type
                    self.parameter_type_registry.define_parameter_type(unified_type)
            elif not isinstance(existing_type, UnifiedParameterType) and type_hint != ParameterTypeHint.STANDARD:
                # Replace existing type with unified version (be careful not to override built-ins)
                if not hasattr(existing_type, 'prototypes'):  # Don't override semantic types
                    unified_type = self._create_unified_parameter_type(param_name, type_hint)
                    if unified_type:
                        # Remove existing type and add unified type
                        # Note: This is a bit hacky, but necessary for the demo
                        if hasattr(self.parameter_type_registry, '_parameter_types_by_name'):
                            self.parameter_type_registry._parameter_types_by_name[param_name] = unified_type
                        elif hasattr(self.parameter_type_registry, 'parameter_types'):
                            # Replace in list
                            for i, pt in enumerate(self.parameter_type_registry.parameter_types):
                                if pt.name == param_name:
                                    self.parameter_type_registry.parameter_types[i] = unified_type
                                    break
    
    def _create_unified_parameter_type(self, param_name: str, type_hint: ParameterTypeHint) -> Optional[UnifiedParameterType]:
        """Create a unified parameter type based on the type hint"""
        # Check if we have a predefined type that matches
        existing_type = self.parameter_type_registry.lookup_by_type_name(param_name)
        
        if type_hint == ParameterTypeHint.STANDARD:
            # For standard parameters, use existing type or create simple one
            if existing_type:
                return None  # Use existing type
            else:
                # Create a dynamic type if dynamic matching is enabled
                if hasattr(self.parameter_type_registry, '_dynamic_matching_enabled') and \
                   self.parameter_type_registry._dynamic_matching_enabled:
                    return UnifiedParameterType(
                        name=param_name,
                        type_hint=ParameterTypeHint.DYNAMIC,
                        similarity_threshold=getattr(self.parameter_type_registry, '_dynamic_threshold', 0.3)
                    )
                else:
                    return UnifiedParameterType(
                        name=param_name,
                        type_hint=ParameterTypeHint.STANDARD
                    )
        
        elif type_hint == ParameterTypeHint.SEMANTIC:
            # Look for existing semantic parameter type
            if existing_type and hasattr(existing_type, 'prototypes'):
                # Convert existing semantic type to unified type
                return UnifiedParameterType(
                    name=param_name,
                    type_hint=ParameterTypeHint.SEMANTIC,
                    prototypes=existing_type.prototypes,
                    similarity_threshold=getattr(existing_type, 'similarity_threshold', 0.7)
                )
            else:
                # Create new semantic type with parameter name as prototype
                return UnifiedParameterType(
                    name=param_name,
                    type_hint=ParameterTypeHint.SEMANTIC,
                    prototypes=[param_name],
                    similarity_threshold=0.7
                )
        
        elif type_hint == ParameterTypeHint.DYNAMIC:
            return UnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.DYNAMIC,
                similarity_threshold=getattr(self.parameter_type_registry, '_dynamic_threshold', 0.3)
            )
        
        elif type_hint == ParameterTypeHint.PHRASE:
            return UnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.PHRASE,
                max_phrase_length=10
            )
        
        elif type_hint == ParameterTypeHint.REGEX:
            # For regex type, we need custom pattern (could be enhanced to parse from expression)
            return UnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.REGEX,
                custom_pattern=r'[^\s]+'  # Default pattern
            )
        
        elif type_hint == ParameterTypeHint.MATH:
            return UnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.MATH
            )
        
        elif type_hint == ParameterTypeHint.QUOTED:
            return UnifiedParameterType(
                name=param_name,
                type_hint=ParameterTypeHint.QUOTED
            )
        
        return None
    
    
    def match(self, text: str) -> Optional[List[Argument]]:
        """
        Enhanced match method that handles phrase boundary detection
        and type-specific matching.
        """
        # First try standard matching
        match = super().match(text)
        
        if match is None:
            return None
        
        # Post-process matches for phrase parameters
        enhanced_matches = []
        
        for i, argument in enumerate(match):
            param_type = argument.parameter_type
            
            if isinstance(param_type, UnifiedParameterType) and \
               param_type.type_hint == ParameterTypeHint.PHRASE:
                # For phrase parameters, we might need to expand the match
                enhanced_argument = self._enhance_phrase_match(argument, text)
                enhanced_matches.append(enhanced_argument)
            else:
                enhanced_matches.append(argument)
        
        return enhanced_matches
    
    def _enhance_phrase_match(self, argument: Argument, full_text: str) -> Argument:
        """Enhance phrase matching to capture multi-word phrases"""
        if not argument.group:
            return argument
        
        param_type = argument.parameter_type
        if not isinstance(param_type, UnifiedParameterType):
            return argument
        
        # Get current match boundaries
        match_start = argument.group.start
        match_end = argument.group.end
        
        # Determine optimal phrase boundaries
        phrase_start, phrase_end = param_type.get_phrase_boundaries(
            full_text, match_start, match_end
        )
        
        # If boundaries changed, create new argument
        if phrase_start != match_start or phrase_end != match_end:
            phrase_text = full_text[phrase_start:phrase_end].strip()
            
            # Create new group with expanded boundaries
            new_group = Group(phrase_text, phrase_start, phrase_end, [phrase_text])
            
            # Create new argument
            return Argument(new_group, param_type)
        
        return argument
    
    def get_parameter_metadata(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific parameter"""
        return self.parameter_metadata.get(param_name)
    
    def get_all_parameter_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all parameters"""
        return self.parameter_metadata.copy()
    
    def __repr__(self):
        return f"UnifiedBudExpression('{self.original_expression}')"
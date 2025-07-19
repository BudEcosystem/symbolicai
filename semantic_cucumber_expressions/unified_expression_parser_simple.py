# semantic_cucumber_expressions/unified_expression_parser_simple.py
from __future__ import annotations
from typing import Dict, Any, Optional
import re

from .unified_parameter_type import ParameterTypeHint


class SimpleUnifiedExpressionParser:
    """
    Simplified parser that extracts type hints from parameter names
    and creates metadata without modifying the AST structure.
    """
    
    TYPE_HINT_PATTERN = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)\}')
    
    def __init__(self):
        self.parameter_metadata = {}
    
    def parse(self, expression: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse expression and extract type hints.
        
        Returns dictionary of parameter metadata.
        """
        self.parameter_metadata = {}
        
        # Find all type hint patterns
        for match in self.TYPE_HINT_PATTERN.finditer(expression):
            param_name = match.group(1)
            type_hint_str = match.group(2)
            
            # Validate type hint
            try:
                type_hint = ParameterTypeHint(type_hint_str)
            except ValueError:
                # Invalid type hint, treat as standard
                type_hint = ParameterTypeHint.STANDARD
            
            self.parameter_metadata[param_name] = {
                'type_hint': type_hint,
                'original_text': f"{param_name}:{type_hint_str}",
                'position': (match.start(), match.end())
            }
        
        # Also find standard parameters without type hints
        standard_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')
        for match in standard_pattern.finditer(expression):
            param_name = match.group(1)
            
            # Only add if not already processed with type hint
            if param_name not in self.parameter_metadata:
                self.parameter_metadata[param_name] = {
                    'type_hint': ParameterTypeHint.STANDARD,
                    'original_text': param_name,
                    'position': (match.start(), match.end())
                }
        
        return self.parameter_metadata
    
    def get_parameter_metadata(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific parameter"""
        return self.parameter_metadata.get(param_name)
    
    def get_all_parameter_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all parameters"""
        return self.parameter_metadata.copy()
    
    def remove_type_hints(self, expression: str) -> str:
        """Remove type hints from expression for compatibility"""
        # Replace {param:type} with {param}
        return self.TYPE_HINT_PATTERN.sub(r'{\1}', expression)
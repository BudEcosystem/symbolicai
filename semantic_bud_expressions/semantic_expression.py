# semantic_bud_expressions/semantic_expression.py
from typing import Optional, List, Dict, Any
from .expression import budExpression
from .semantic_parameter_type import SemanticParameterType
from .math_parameter_type import MathParameterType
from .argument import Argument
from .model_manager import Model2VecManager
from .regex_cache import regex_cache
from .multi_level_cache import get_global_cache
import re

class SemanticBudExpression(budExpression):
    """Enhanced bud Expression with semantic matching capabilities"""
    
    def __init__(self, expression: str, parameter_type_registry):
        super().__init__(expression, parameter_type_registry)
        self.model_manager = Model2VecManager()
        self._semantic_patterns = self._extract_semantic_patterns()
        self._cache = get_global_cache()  # Use multi-level cache
        
    def _extract_semantic_patterns(self) -> List[Dict[str, Any]]:
        """Extract semantic pattern markers from expression"""
        patterns = []
        
        # Look for {{category}} or {{category:type}} patterns
        semantic_pattern = regex_cache.compile(r'\{\{(\w+)(?::(\w+))?\}\}')
        
        for match in semantic_pattern.finditer(self.expression):
            patterns.append({
                'full_match': match.group(0),
                'category': match.group(1),
                'type': match.group(2) or 'semantic',
                'start': match.start(),
                'end': match.end()
            })
            
        return patterns
    
    def match(self, text: str) -> Optional[List[Argument]]:
        """Enhanced match with semantic support"""
        # Check L1 cache first
        cached_result = self._cache.get_expression_result(self.expression, text)
        if cached_result is not None:
            return cached_result
        
        # First try standard matching
        standard_match = super().match(text)
        if standard_match:
            self._cache.put_expression_result(self.expression, text, standard_match)
            return standard_match
            
        # If no standard match and we have semantic patterns, try semantic matching
        if self._semantic_patterns:
            result = self._semantic_match(text)
            self._cache.put_expression_result(self.expression, text, result)
            return result
            
        self._cache.put_expression_result(self.expression, text, None)
        return None
    
    def _semantic_match(self, text: str) -> Optional[List[Argument]]:
        """Perform semantic matching"""
        # Build regex pattern with capture groups for semantic parts
        pattern = self.expression
        capture_groups = []
        
        # Replace semantic patterns with capture groups
        for sp in reversed(self._semantic_patterns):  # Reverse to maintain indices
            pattern = pattern[:sp['start']] + '(.+?)' + pattern[sp['end']:]
            capture_groups.insert(0, sp)  # Insert at beginning due to reverse
            
        # Try to match the pattern
        regex = regex_cache.compile(f'^{pattern}$')
        match = regex.match(text)
        
        if not match:
            return None
            
        # Validate semantic matches
        arguments = []
        for i, sp in enumerate(capture_groups):
            captured_text = match.group(i + 1)
            
            # Get the semantic parameter type
            param_type = self.parameter_type_registry.lookup_by_type_name(sp['category'])
            
            if isinstance(param_type, SemanticParameterType):
                # Check semantic match
                matches, similarity, _ = param_type.matches_semantically(captured_text)
                if not matches:
                    return None  # Semantic validation failed
                    
            # Create argument
            from .group import Group
            group = Group(captured_text, match.start(i + 1), match.end(i + 1), [])
            arguments.append(Argument(group, param_type))
            
        return arguments

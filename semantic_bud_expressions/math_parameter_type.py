from typing import List, Optional, Callable, Any
import sympy
from sympy import symbols, simplify, expand
from sympy.parsing.sympy_parser import parse_expr
from .parameter_type import ParameterType

class MathParameterType(ParameterType):
    """Parameter type for mathematical expressions"""
    
    def __init__(
        self,
        name: str = "math",
        type: type = str,
        transformer: Optional[Callable] = None,
        use_for_snippets: bool = True,
        prefer_for_regexp_match: bool = False,
        simplify_expressions: bool = True
    ):
        """
        Create a mathematical expression parameter type
        
        Args:
            name: Name of the parameter type
            type: The type to transform to
            transformer: Custom transformation function
            use_for_snippets: Whether to use for snippet generation
            prefer_for_regexp_match: Whether to prefer for regexp matching
            simplify_expressions: Whether to simplify expressions before comparison
        """
        # Regex that captures mathematical expressions
        math_regexp = r'[0-9+\-*/().,\s\w^=<>≤≥]+'
        
        super().__init__(
            name=name,
            regexp=math_regexp,
            type=type,
            transformer=transformer or self._default_transformer,
            use_for_snippets=use_for_snippets,
            prefer_for_regexp_match=prefer_for_regexp_match
        )
        
        self.simplify_expressions = simplify_expressions
        
    def _default_transformer(self, *group_values) -> Any:
        """Default transformer that parses and normalizes math expressions"""
        if not group_values or not group_values[0]:
            return None
            
        expr_str = group_values[0].strip()
        
        try:
            # Parse the expression
            expr = parse_expr(expr_str, evaluate=False)
            
            # Simplify if requested
            if self.simplify_expressions:
                expr = simplify(expand(expr))
            
            return MathExpression(original=expr_str, parsed=expr, simplified=str(expr))
            
        except Exception as e:
            # If parsing fails, return as string
            return expr_str
    
    @staticmethod
    def are_equivalent(expr1: str, expr2: str) -> bool:
        """Check if two mathematical expressions are equivalent"""
        try:
            parsed1 = parse_expr(expr1, evaluate=False)
            parsed2 = parse_expr(expr2, evaluate=False)
            
            # Simplify both expressions
            simplified1 = simplify(expand(parsed1))
            simplified2 = simplify(expand(parsed2))
            
            # Check equivalence
            return simplified1.equals(simplified2)
            
        except Exception:
            # If parsing fails, fall back to string comparison
            return expr1 == expr2

class MathExpression:
    """Container for mathematical expressions"""
    
    def __init__(self, original: str, parsed: Any, simplified: str):
        self.original = original
        self.parsed = parsed
        self.simplified = simplified
        
    def __str__(self):
        return self.simplified
        
    def __eq__(self, other):
        if isinstance(other, MathExpression):
            return self.simplified == other.simplified
        elif isinstance(other, str):
            return MathParameterType.are_equivalent(self.simplified, other)
        return False

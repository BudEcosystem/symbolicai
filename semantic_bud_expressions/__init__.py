from .semantic_expression import SemanticBudExpression
from .semantic_parameter_type import SemanticParameterType
from .semantic_registry import SemanticParameterTypeRegistry
from .math_parameter_type import MathParameterType, MathExpression
from .model_manager import Model2VecManager
from .dynamic_semantic_parameter_type import DynamicSemanticParameterType
from .regex_cache import regex_cache
from .batch_matcher import BatchMatcher, OptimizedSemanticMatcher
from .optimized_semantic_type import OptimizedSemanticParameterType, OptimizedDynamicSemanticParameterType
from .multi_level_cache import MultiLevelCache, get_global_cache, clear_global_cache

# Initialize regex cache with common patterns
regex_cache.precompile_common_patterns()

# New unified system
from .unified_expression import UnifiedBudExpression
from .unified_parameter_type import UnifiedParameterType, ParameterTypeHint
from .unified_registry import UnifiedParameterTypeRegistry

# Also export original classes for compatibility
from .expression import budExpression
from .parameter_type import ParameterType
from .parameter_type_registry import ParameterTypeRegistry

__all__ = [
    # Original semantic system
    'SemanticBudExpression',
    'SemanticParameterType',
    'SemanticParameterTypeRegistry',
    'MathParameterType',
    'MathExpression',
    'Model2VecManager',
    'DynamicSemanticParameterType',
    
    # Optimized components
    'BatchMatcher',
    'OptimizedSemanticMatcher',
    'OptimizedSemanticParameterType',
    'OptimizedDynamicSemanticParameterType',
    'MultiLevelCache',
    'get_global_cache',
    'clear_global_cache',
    
    # New unified system
    'UnifiedBudExpression',
    'UnifiedParameterType',
    'UnifiedParameterTypeRegistry',
    'ParameterTypeHint',
    
    # Original bud expressions
    'budExpression',
    'ParameterType',
    'ParameterTypeRegistry'
]
#!/usr/bin/env python3
"""
Debug expression matching issue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    SemanticBudExpression,
    SemanticParameterTypeRegistry,
    budExpression
)


def debug_expression_matching():
    """Debug why 'coding' matches in expression"""
    print("=== Debug Expression Matching ===")
    
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    # Test with regular budExpression first
    print("\n1. Testing with regular budExpression:")
    expr1 = budExpression("I love {fruit}", registry)
    match1 = expr1.match("I love coding")
    if match1:
        print(f"  Matched! Value: {match1[0].group.value}")
    else:
        print("  No match")
    
    # Test with SemanticBudExpression
    print("\n2. Testing with SemanticBudExpression:")
    expr2 = SemanticBudExpression("I love {fruit}", registry)
    match2 = expr2.match("I love coding")
    if match2:
        print(f"  Matched! Value: {match2[0].group.value}")
        # Check if semantic validation happened
        try:
            result = match2[0].parameter_type.transform([match2[0].group.value])
            print(f"  Transform result: {result}")
        except Exception as e:
            print(f"  Transform error: {e}")
    else:
        print("  No match")
    
    # Check the parameter type
    print("\n3. Checking fruit parameter type:")
    fruit_type = registry.lookup_by_type_name('fruit')
    print(f"  Type: {type(fruit_type).__name__}")
    print(f"  Regexp: {fruit_type.regexp}")
    
    # Test regexp directly
    import re
    pattern = re.compile(fruit_type.regexp)
    if pattern.match("coding"):
        print(f"  Regexp '{fruit_type.regexp}' matches 'coding'")
    else:
        print(f"  Regexp '{fruit_type.regexp}' does not match 'coding'")
    
    # Test semantic validation
    print("\n4. Testing semantic validation:")
    matches, score, closest = fruit_type.matches_semantically("coding")
    print(f"  Semantic match: {matches}, score: {score:.3f}")
    
    # Test transform
    print("\n5. Testing transform:")
    try:
        result = fruit_type.transform(["coding"])
        print(f"  Transform succeeded: {result}")
    except Exception as e:
        print(f"  Transform failed: {e}")


if __name__ == "__main__":
    debug_expression_matching()
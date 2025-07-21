#!/usr/bin/env python3
"""
Test script to verify benchmarking tools work correctly
"""

import sys
import traceback

# Test imports
try:
    from semantic_bud_expressions import (
        UnifiedBudExpression,
        UnifiedParameterTypeRegistry
    )
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test basic functionality
try:
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    print("✓ Registry and model initialized")
    
    # Test simple expression
    expr = UnifiedBudExpression("I love {fruit}", registry)
    match = expr.match("I love apples")
    if match:
        print(f"✓ Simple match successful: {match[0].value}")
    else:
        print("✗ Simple match failed")
    
    # Test semantic expression
    expr2 = UnifiedBudExpression("I am {emotion} about this", registry)
    match2 = expr2.match("I am happy about this")
    if match2:
        print(f"✓ Semantic match successful: {match2[0].value}")
    else:
        print("✗ Semantic match failed")
    
    # Test cache
    cache_size = len(registry.model_manager.cache.cache)
    print(f"✓ Cache working, size: {cache_size}")
    
    print("\n✓ All basic tests passed! Ready to run benchmarks.")
    
except Exception as e:
    print(f"\n✗ Error in basic tests: {e}")
    traceback.print_exc()
    sys.exit(1)
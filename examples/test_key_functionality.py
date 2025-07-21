#!/usr/bin/env python3
"""
Test key functionality to ensure optimizations don't break core features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    SemanticBudExpression,
    SemanticParameterTypeRegistry,
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry,
    OptimizedSemanticParameterType,
    get_global_cache,
    clear_global_cache
)


def test_scenario(name, test_func):
    """Run a test scenario and report results"""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    try:
        test_func()
        print("✅ PASSED")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_basic_semantic():
    """Test basic semantic matching"""
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    expr = SemanticBudExpression("I love {fruit}", registry)
    
    # Should match
    match = expr.match("I love apples")
    assert match is not None, "Should match 'apples'"
    assert match[0].group.value == "apples"
    
    # Should match structurally but fail semantically in transform
    match = expr.match("I love coding")
    assert match is not None, "Should match structurally"
    try:
        match[0].parameter_type.transform([match[0].group.value])
        assert False, "Transform should fail"
    except ValueError:
        pass  # Expected
    
    print("  - Basic semantic matching works correctly")


def test_unified_types():
    """Test unified expression types"""
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Semantic type
    registry.create_semantic_parameter_type(
        'vehicle', ['car', 'truck', 'bike'], 0.6
    )
    expr = UnifiedBudExpression("I drive a {vehicle:semantic}", registry)
    match = expr.match("I drive a automobile")
    assert match is not None, "Should match semantically similar word"
    
    # Regex type
    registry.create_regex_parameter_type('year', r'20\d{2}')
    expr2 = UnifiedBudExpression("Year {year:regex}", registry)
    match = expr2.match("Year 2024")
    assert match is not None, "Should match regex pattern"
    
    # Dynamic type
    registry.enable_dynamic_matching(True)
    expr3 = UnifiedBudExpression("I love {pets:dynamic}", registry)
    match = expr3.match("I love dogs")
    assert match is not None, "Should match dynamically"
    
    print("  - Unified expression types work correctly")


def test_caching():
    """Test caching functionality"""
    clear_global_cache()
    cache = get_global_cache()
    
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    expr = UnifiedBudExpression("Hello {name}", registry)
    
    # First match - cache miss
    match1 = expr.match("Hello World")
    stats1 = cache.get_stats()
    
    # Second match - cache hit
    match2 = expr.match("Hello World")
    stats2 = cache.get_stats()
    
    assert stats2['l1_hits'] > stats1['l1_hits'], "Cache hits should increase"
    assert match1 is not None and match2 is not None, "Both should match"
    
    print("  - Caching works correctly")


def test_optimized_types():
    """Test optimized semantic types"""
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Use optimized type with unique name
    opt_type = OptimizedSemanticParameterType(
        name="mood",
        prototypes=["happy", "sad", "angry", "excited"],
        similarity_threshold=0.6
    )
    registry.define_parameter_type(opt_type)
    
    expr = UnifiedBudExpression("I feel {mood}", registry)
    match = expr.match("I feel joyful")
    assert match is not None, "Should match similar emotion"
    
    print("  - Optimized types work correctly")


def test_multiple_parameters():
    """Test expressions with multiple parameters"""
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    expr = UnifiedBudExpression(
        "{greeting} {name}, the {fruit} costs {price} dollars",
        registry
    )
    
    match = expr.match("Hello John, the apple costs 5 dollars")
    assert match is not None, "Should match all parameters"
    assert len(match) == 4, "Should have 4 matches"
    assert match[0].group.value == "Hello"
    assert match[1].group.value == "John"
    assert match[2].group.value == "apple"
    assert match[3].group.value == "5"
    
    print("  - Multiple parameter matching works correctly")


def test_math_expressions():
    """Test math expression handling"""
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    expr = SemanticBudExpression("Calculate {math}", registry)
    match = expr.match("Calculate 2 + 3 * 4")
    assert match is not None, "Should match math expression"
    assert match[0].group.value == "2 + 3 * 4"
    
    # Transform should work
    result = match[0].parameter_type.transform([match[0].group.value])
    assert result == "2 + 3 * 4", "Math transform should preserve expression"
    
    print("  - Math expressions work correctly")


def main():
    """Run all key functionality tests"""
    print("=== Key Functionality Tests ===")
    print("Testing core features with optimizations enabled...")
    
    tests = [
        ("Basic Semantic Matching", test_basic_semantic),
        ("Unified Expression Types", test_unified_types),
        ("Caching Functionality", test_caching),
        ("Optimized Types", test_optimized_types),
        ("Multiple Parameters", test_multiple_parameters),
        ("Math Expressions", test_math_expressions),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if test_scenario(name, test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Summary: {passed}/{len(tests)} tests passed")
    print(f"{'='*50}")
    
    if failed == 0:
        print("\n✅ All key functionality tests passed!")
        print("\nThe optimizations are working correctly without breaking core features.")
    else:
        print(f"\n❌ {failed} tests failed!")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
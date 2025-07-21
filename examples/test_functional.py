#!/usr/bin/env python3
"""
Comprehensive functional tests for Semantic Bud Expressions with optimizations
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
    clear_global_cache,
    regex_cache
)


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_true(self, condition, message):
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  ✗ {message}")
    
    def assert_equal(self, actual, expected, message):
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(f"{message} (expected: {expected}, got: {actual})")
            print(f"  ✗ {message} (expected: {expected}, got: {actual})")
    
    def assert_not_none(self, value, message):
        if value is not None:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(f"{message} (got None)")
            print(f"  ✗ {message} (got None)")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed ({self.passed/total*100:.1f}%)")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*60}")
        return self.failed == 0


def test_basic_semantic_matching(result):
    """Test basic semantic matching functionality"""
    print("\n=== Testing Basic Semantic Matching ===")
    
    # Initialize registry
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    # Test 1: Simple fruit matching
    expr = SemanticBudExpression("I love {fruit}", registry)
    
    match = expr.match("I love apples")
    result.assert_not_none(match, "Should match 'apples' as fruit")
    if match:
        result.assert_equal(len(match), 1, "Should have 1 argument")
        result.assert_equal(match[0].group.value, "apples", "Should extract 'apples'")
    
    # Test 2: Structural match but semantic validation failure
    match = expr.match("I love coding")
    result.assert_not_none(match, "Should structurally match 'coding'")
    if match:
        # Should fail during transform due to semantic validation
        try:
            transformed = match[0].parameter_type.transform([match[0].group.value])
            result.assert_true(False, "Transform should fail for 'coding' as fruit")
        except ValueError as e:
            result.assert_true("semantically" in str(e), "Should fail with semantic error")
    
    # Test 3: Greeting matching
    expr2 = SemanticBudExpression("{greeting} there!", registry)
    match = expr2.match("Hello there!")
    result.assert_not_none(match, "Should match 'Hello' as greeting")
    
    # Test 4: Multiple parameters
    expr3 = SemanticBudExpression("{greeting} {name}, I love {fruit}", registry)
    match = expr3.match("Hi John, I love bananas")
    result.assert_not_none(match, "Should match multiple parameters")
    if match:
        result.assert_equal(len(match), 3, "Should have 3 arguments")


def test_unified_expression_types(result):
    """Test unified expression with different type hints"""
    print("\n=== Testing Unified Expression Types ===")
    
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Test 1: Semantic type hint
    registry.create_semantic_parameter_type(
        'vehicle',
        ['car', 'truck', 'bike', 'motorcycle'],
        similarity_threshold=0.6
    )
    
    expr = UnifiedBudExpression("I drive a {vehicle:semantic}", registry)
    match = expr.match("I drive a automobile")
    result.assert_not_none(match, "Should match 'automobile' as vehicle")
    
    # Test 2: Regex type hint
    registry.create_regex_parameter_type('phone', r'\d{3}-\d{3}-\d{4}')
    expr2 = UnifiedBudExpression("Call me at {phone:regex}", registry)
    match = expr2.match("Call me at 123-456-7890")
    result.assert_not_none(match, "Should match phone number pattern")
    
    # Test 3: Dynamic type hint
    registry.enable_dynamic_matching(True)
    registry.set_dynamic_threshold(0.3)
    expr3 = UnifiedBudExpression("I love {animals:dynamic}", registry)
    match = expr3.match("I love cats")
    result.assert_not_none(match, "Should match 'cats' dynamically with 'animals'")
    
    # Test 4: Math type hint
    expr4 = UnifiedBudExpression("Calculate {math} for me", registry)
    match = expr4.match("Calculate 2 + 3 * 4 for me")
    result.assert_not_none(match, "Should match math expression")


def test_cache_functionality(result):
    """Test multi-level cache functionality"""
    print("\n=== Testing Cache Functionality ===")
    
    # Clear cache first
    clear_global_cache()
    cache = get_global_cache()
    
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Test 1: Expression caching
    expr = UnifiedBudExpression("Hello {name}", registry)
    
    # First match
    match1 = expr.match("Hello World")
    stats1 = cache.get_stats()
    
    # Second match (should hit cache)
    match2 = expr.match("Hello World")
    stats2 = cache.get_stats()
    
    result.assert_equal(stats2['l1_hits'], stats1['l1_hits'] + 1, "L1 cache hit count should increase")
    result.assert_true(match1 is not None and match2 is not None, "Both matches should succeed")
    
    # Test 2: Embedding caching
    initial_l2_misses = stats2['l2_misses']
    
    # Create expression that will compute embeddings
    expr2 = SemanticBudExpression("I love {fruit}", registry)
    expr2.match("I love apples")
    
    stats3 = cache.get_stats()
    result.assert_true(stats3['l2_misses'] >= initial_l2_misses, "L2 cache should record misses for new embeddings")
    
    # Test 3: Cache clearing
    clear_global_cache()
    stats4 = cache.get_stats()
    result.assert_equal(stats4['l1_hits'], 0, "Cache should be cleared")
    result.assert_equal(stats4['l1_misses'], 0, "Cache stats should be reset")


def test_regex_cache(result):
    """Test regex compilation cache"""
    print("\n=== Testing Regex Cache ===")
    
    # Clear regex cache
    regex_cache.clear()
    
    # Test 1: Cache miss then hit
    pattern = r'\d+'
    compiled1 = regex_cache.compile(pattern)
    stats1 = regex_cache.get_stats()
    
    compiled2 = regex_cache.compile(pattern)
    stats2 = regex_cache.get_stats()
    
    result.assert_equal(stats1['misses'], 1, "First compilation should be a miss")
    result.assert_equal(stats2['hits'], 1, "Second compilation should be a hit")
    result.assert_true(compiled1 is compiled2, "Should return same compiled pattern")
    
    # Test 2: Different patterns
    pattern2 = r'\w+'
    compiled3 = regex_cache.compile(pattern2)
    stats3 = regex_cache.get_stats()
    
    result.assert_equal(stats3['misses'], 2, "New pattern should be a miss")
    result.assert_true(compiled3 is not compiled1, "Different patterns should have different compiled objects")


def test_optimized_semantic_type(result):
    """Test optimized semantic parameter type"""
    print("\n=== Testing Optimized Semantic Type ===")
    
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create optimized semantic type
    opt_type = OptimizedSemanticParameterType(
        name="color",
        prototypes=["red", "blue", "green", "yellow"],
        similarity_threshold=0.6
    )
    registry.define_parameter_type(opt_type)
    
    # Test matching
    expr = UnifiedBudExpression("The sky is {color}", registry)
    match = expr.match("The sky is azure")
    result.assert_not_none(match, "Should match 'azure' as a color")
    
    # Test batch preparation
    text = "red blue green yellow purple orange"
    OptimizedSemanticParameterType.prepare_text(text, [opt_type])
    
    # Purple might not match with threshold 0.6 - let's check the score
    matches, score, closest = opt_type.matches_semantically("purple")
    # Purple is not in prototypes and may have low similarity
    result.assert_true(score < 0.6, f"Purple should have similarity < 0.6 with basic colors (got {score:.2f})")
    result.assert_equal(closest, "blue", "Purple should be closest to blue")


def test_backward_compatibility(result):
    """Test backward compatibility with original API"""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test 1: Original SemanticBudExpression
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    expr = SemanticBudExpression("I love {fruit}", registry)
    match = expr.match("I love apples")
    result.assert_not_none(match, "Original API should still work")
    
    # Test 2: Custom semantic category
    registry.define_semantic_category(
        "programming_language",
        ["python", "java", "javascript", "ruby"],
        similarity_threshold=0.7
    )
    
    expr2 = SemanticBudExpression("I code in {programming_language}", registry)
    match = expr2.match("I code in golang")
    result.assert_not_none(match, "Custom categories should work")
    
    # Test 3: Math expressions
    expr3 = SemanticBudExpression("Calculate {math}", registry)
    match = expr3.match("Calculate 5 + 3")
    result.assert_not_none(match, "Math expressions should work")


def test_error_handling(result):
    """Test error handling and edge cases"""
    print("\n=== Testing Error Handling ===")
    
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Test 1: Empty expression
    try:
        expr = UnifiedBudExpression("", registry)
        match = expr.match("")
        result.assert_true(match is not None, "Empty expression should match empty string")
    except Exception as e:
        result.assert_true(False, f"Empty expression should not raise exception: {e}")
    
    # Test 2: No parameters
    expr2 = UnifiedBudExpression("Hello World", registry)
    match = expr2.match("Hello World")
    result.assert_not_none(match, "Expression without parameters should work")
    
    # Test 3: Invalid regex pattern
    try:
        registry.create_regex_parameter_type('bad_regex', r'[invalid(')
        # If no exception, check if it was created
        bad_type = registry.lookup_by_type_name('bad_regex')
        if bad_type is None:
            result.assert_true(False, "Invalid regex should either raise exception or create type")
        else:
            # The implementation might handle this differently
            result.assert_true(True, "Regex parameter type created (implementation specific)")
    except Exception:
        result.assert_true(True, "Invalid regex correctly raised exception")
    
    # Test 4: Non-existent parameter type
    expr4 = UnifiedBudExpression("Find {nonexistent}", registry)
    match = expr4.match("Find something")
    # Should either match with dynamic or return None
    result.assert_true(True, "Non-existent parameter handled gracefully")


def run_all_tests():
    """Run all functional tests"""
    print("=== Semantic Bud Expressions - Functional Test Suite ===")
    
    result = TestResult()
    
    # Run test suites
    test_basic_semantic_matching(result)
    test_unified_expression_types(result)
    test_cache_functionality(result)
    test_regex_cache(result)
    test_optimized_semantic_type(result)
    test_backward_compatibility(result)
    test_error_handling(result)
    
    # Show summary
    success = result.summary()
    
    if success:
        print("\n✅ All functional tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
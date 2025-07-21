#!/usr/bin/env python3
"""
Debug failing tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    SemanticBudExpression,
    SemanticParameterTypeRegistry,
    UnifiedParameterTypeRegistry,
    OptimizedSemanticParameterType,
    regex_cache
)
import re


def test_coding_fruit():
    """Debug why 'coding' matches as fruit"""
    print("=== Testing 'coding' as fruit ===")
    
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    expr = SemanticBudExpression("I love {fruit}", registry)
    match = expr.match("I love coding")
    
    if match:
        print(f"Unexpected match found!")
        print(f"Matched value: {match[0].group.value}")
        print(f"Parameter type: {match[0].parameter_type.name}")
        
        # Check similarity
        fruit_type = registry.lookup_by_type_name('fruit')
        if hasattr(fruit_type, 'matches_semantically'):
            matches, score, closest = fruit_type.matches_semantically('coding')
            print(f"Semantic match: {matches}, score: {score:.3f}, closest: {closest}")
    else:
        print("No match (as expected)")


def test_purple_color():
    """Debug purple color matching"""
    print("\n=== Testing 'purple' as color ===")
    
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create optimized semantic type
    opt_type = OptimizedSemanticParameterType(
        name="color",
        prototypes=["red", "blue", "green", "yellow"],
        similarity_threshold=0.6
    )
    
    # Test direct matching
    matches, score, closest = opt_type.matches_semantically("purple")
    print(f"Direct match result: {matches}, score: {score:.3f}, closest prototype: {closest}")
    
    # Let's check individual similarities
    print("\nSimilarities with each prototype:")
    purple_emb = opt_type.model_manager.embed_sync(["purple"])[0]
    opt_type._ensure_embeddings()
    
    for i, proto in enumerate(opt_type.prototypes):
        sim = opt_type.model_manager.cosine_similarity(purple_emb, opt_type._prototype_embeddings[i])
        print(f"  purple vs {proto}: {sim:.3f}")
    
    print(f"\nThreshold: {opt_type.similarity_threshold}")


def test_regex_error():
    """Debug regex error handling"""
    print("\n=== Testing regex error handling ===")
    
    registry = UnifiedParameterTypeRegistry()
    
    try:
        # This should raise an error
        pattern = r'[invalid('
        compiled = regex_cache.compile(pattern)
        print("ERROR: Invalid regex was compiled successfully!")
    except re.error as e:
        print(f"Correctly caught regex error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")
    
    # Try with create_regex_parameter_type
    try:
        registry.create_regex_parameter_type('bad_regex', r'[invalid(')
        print("ERROR: create_regex_parameter_type did not raise exception!")
    except Exception as e:
        print(f"Correctly caught error in create_regex_parameter_type: {type(e).__name__}: {e}")


def test_semantic_thresholds():
    """Test semantic similarity thresholds"""
    print("\n=== Testing Semantic Thresholds ===")
    
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    # Get fruit type
    fruit_type = registry.lookup_by_type_name('fruit')
    print(f"Fruit type threshold: {fruit_type.similarity_threshold}")
    
    # Test various words
    test_words = ["apple", "banana", "coding", "programming", "orange", "computer"]
    
    for word in test_words:
        matches, score, closest = fruit_type.matches_semantically(word)
        print(f"{word:12} -> matches: {matches}, score: {score:.3f}, closest: {closest}")


if __name__ == "__main__":
    test_coding_fruit()
    test_purple_color()
    test_regex_error()
    test_semantic_thresholds()
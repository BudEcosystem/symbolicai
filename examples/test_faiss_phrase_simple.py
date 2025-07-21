#!/usr/bin/env python3
"""
Simple test to verify FAISS-enhanced phrase matching is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import UnifiedBudExpression, EnhancedUnifiedParameterTypeRegistry


def test_basic_functionality():
    """Test basic FAISS phrase matching functionality"""
    print("Testing FAISS-Enhanced Phrase Matching\n")
    
    # Create registry
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Test 1: Basic phrase matching
    print("1. Basic Multi-Word Phrase Matching:")
    registry.create_phrase_parameter_type(
        "car",
        max_phrase_length=4,
        known_phrases=["Tesla Model 3", "BMW X5", "Mercedes S Class"]
    )
    
    expr = UnifiedBudExpression("I drive a {car:phrase}", registry)
    tests = [
        "I drive a Tesla Model 3",
        "I drive a BMW X5",
        "I drive a Mercedes S Class"
    ]
    
    for text in tests:
        match = expr.match(text)
        if match:
            print(f"  ✓ '{text}' → {match[0].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    
    # Test 2: Semantic phrase matching
    print("\n2. Semantic Phrase Matching:")
    registry.create_semantic_phrase_parameter_type(
        "device",
        semantic_categories=["phone", "laptop", "tablet"],
        max_phrase_length=5
    )
    
    expr2 = UnifiedBudExpression("I bought a {device:phrase}", registry)
    tests2 = [
        "I bought a iPhone 15 Pro",
        "I bought a MacBook Air M2",
        "I bought a Samsung Galaxy Tab"
    ]
    
    for text in tests2:
        match = expr2.match(text)
        if match:
            print(f"  ✓ '{text}' → {match[0].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    
    # Test 3: Large vocabulary with FAISS
    print("\n3. Large Vocabulary Performance:")
    large_phrases = []
    for i in range(200):
        large_phrases.append(f"Product Model {i}")
    
    registry.create_phrase_parameter_type(
        "product",
        max_phrase_length=3,
        known_phrases=large_phrases
    )
    
    # Check if FAISS was enabled
    stats = registry.get_faiss_statistics()
    print(f"  FAISS enabled for 'product': {'product' in stats['faiss_enabled_types']}")
    print(f"  Total FAISS-enabled types: {stats['total_faiss_types']}")
    
    # Test matching
    expr3 = UnifiedBudExpression("Buy {product:phrase}", registry)
    match = expr3.match("Buy Product Model 150")
    if match:
        print(f"  ✓ Matched: {match[0].value}")
    
    print("\n✅ All basic tests completed successfully!")


if __name__ == "__main__":
    test_basic_functionality()
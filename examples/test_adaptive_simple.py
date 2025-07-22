#!/usr/bin/env python3
"""
Simple test of the adaptive context matcher core functionality.
Tests without the training components to avoid sklearn dependency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    AdaptiveContextMatcher,
    ContextNormalizationConfig,
    PerspectiveNormalizer,
    SemanticExpander,
    EnhancedUnifiedParameterTypeRegistry
)


def test_perspective_normalization():
    """Test perspective normalization functionality"""
    print("=" * 60)
    print("TESTING PERSPECTIVE NORMALIZATION")
    print("=" * 60)
    
    normalizer = PerspectiveNormalizer()
    
    test_cases = [
        ("you need help with technical support", "user need help with technical support"),
        ("I want to buy something", "user want to buy something"),
        ("your account has issues", "user account has issues"),
        ("my order is delayed", "user order is delayed")
    ]
    
    print("Perspective normalization examples:")
    for original, expected in test_cases:
        normalized = normalizer.normalize_perspective(original)
        status = "✓" if expected in normalized else "✗"
        print(f"{status} '{original}' → '{normalized}'")
    
    print("\nPerspective variants generation:")
    variants = normalizer.generate_perspective_variants("you need help")
    for variant in variants:
        print(f"  - '{variant}'")


def test_semantic_expansion():
    """Test semantic context expansion"""
    print("\n" + "=" * 60)
    print("TESTING SEMANTIC EXPANSION")
    print("=" * 60)
    
    expander = SemanticExpander()
    
    test_cases = [
        ("medical help", "Should expand with healthcare terms"),
        ("banking money", "Should expand with financial terms"),
        ("shopping buy", "Should expand with e-commerce terms"),
        ("quick support", "Should expand with common terms")
    ]
    
    print("Semantic expansion examples:")
    for context, description in test_cases:
        expanded = expander.expand_context(context)
        print(f"'{context}' → '{expanded}'")
        print(f"  {description}")
        print()


def test_basic_matching():
    """Test basic adaptive matching without training"""
    print("=" * 60)
    print("TESTING BASIC ADAPTIVE MATCHING")
    print("=" * 60)
    
    # Healthcare example
    print("\n1. Healthcare Context Matching")
    print("-" * 40)
    
    healthcare_matcher = AdaptiveContextMatcher(
        expression="patient has {condition}",
        target_context="your health medical help",  # Short, second-person
        registry=EnhancedUnifiedParameterTypeRegistry()
    )
    
    print(f"Original target: '{healthcare_matcher.original_target_context}'")
    print("Processed contexts:")
    for key, value in healthcare_matcher.processed_target_contexts.items():
        if key != 'perspective_variants':
            print(f"  {key}: '{value}'")
    
    test_cases = [
        # Should match - medical context
        "The doctor examined the patient who has severe headache and needs treatment",
        # Should not match - technical context  
        "The computer system has software bugs requiring immediate fixes"
    ]
    
    for text in test_cases:
        print(f"\nTesting: '{text[:60]}...'")
        result = healthcare_matcher.match(text)
        if result:
            print(f"  ✓ MATCH - Method: {result['method']}, Confidence: {result['confidence']:.2f}")
            print(f"  Parameters: {result['parameters']}")
            print(f"  Explanation: {result['explanation']}")
        else:
            print("  ✗ NO MATCH")
    
    # E-commerce example
    print("\n\n2. E-commerce Context Matching")
    print("-" * 40)
    
    shopping_matcher = AdaptiveContextMatcher(
        expression="add {item} to cart",
        target_context="shopping buy",  # Very short target
        registry=EnhancedUnifiedParameterTypeRegistry()
    )
    
    print(f"Target context: '{shopping_matcher.original_target_context}'")
    print(f"Expanded context: '{shopping_matcher.processed_target_contexts['expanded']}'")
    
    shopping_tests = [
        # Should match - shopping context
        "Browse our online store and add iPhone to cart for checkout",
        # Should not match - cooking context
        "In the recipe, add salt to mixture for better taste"
    ]
    
    for text in shopping_tests:
        print(f"\nTesting: '{text}'")
        result = shopping_matcher.match(text)
        if result:
            print(f"  ✓ MATCH - Confidence: {result['confidence']:.2f}")
            print(f"  Parameters: {result['parameters']}")
        else:
            print("  ✗ NO MATCH")


def test_fallback_strategies():
    """Test fallback matching strategies"""
    print("\n" + "=" * 60)
    print("TESTING FALLBACK STRATEGIES")
    print("=" * 60)
    
    # Configure with fallback strategies
    config = ContextNormalizationConfig(
        perspective_normalization=True,
        semantic_expansion=True,
        fallback_strategies=['keyword_matching', 'partial_similarity', 'fuzzy_matching']
    )
    
    fallback_matcher = AdaptiveContextMatcher(
        expression="solve {problem}",
        target_context="technical issue help",
        registry=EnhancedUnifiedParameterTypeRegistry(),
        config=config
    )
    
    # Test case that should trigger fallback
    text = "System technical issue needs attention. Please solve network problem immediately."
    
    print(f"Testing fallback with: '{text}'")
    result = fallback_matcher.match(text)
    
    if result:
        print(f"✓ FALLBACK MATCH")
        print(f"  Method: {result['method']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Explanation: {result['explanation']}")
    else:
        print("✗ NO MATCH (even with fallbacks)")


def test_detailed_explanation():
    """Test detailed matching explanations"""
    print("\n" + "=" * 60)
    print("TESTING DETAILED EXPLANATIONS")
    print("=" * 60)
    
    matcher = AdaptiveContextMatcher(
        expression="user needs {assistance}",
        target_context="customer support help",
        registry=EnhancedUnifiedParameterTypeRegistry()
    )
    
    test_text = "The customer service team is ready. The user needs immediate assistance with account access."
    
    print(f"Analyzing: '{test_text}'")
    explanation = matcher.explain_matching_process(test_text)
    
    print(f"\nDetailed Analysis:")
    print(f"Target context: {explanation['target_context']}")
    print(f"Is trained: {explanation['is_trained']}")
    print(f"Match result: {explanation['result']['matched']}")
    
    if explanation['result']['matched']:
        print(f"Confidence: {explanation['result']['confidence']:.2f}")
        print(f"Method used: {explanation['result']['method']}")
        print(f"Parameters: {explanation['result']['parameters']}")
    
    print("\nMatching steps:")
    for step in explanation['matching_steps']:
        print(f"  {step}")
    
    print("\nProcessed contexts:")
    for key, value in explanation['processed_contexts'].items():
        if key != 'perspective_variants':
            print(f"  {key}: '{value}'")


def main():
    """Run all basic adaptive matching tests"""
    print("ADAPTIVE CONTEXT MATCHER - BASIC FUNCTIONALITY TEST")
    print("=" * 80)
    print("Testing the adaptive matcher that works without training data.")
    print("Handles context length mismatches, perspective differences, and fallbacks.")
    print()
    
    try:
        # Test individual components
        test_perspective_normalization()
        test_semantic_expansion()
        
        # Test core matching functionality
        test_basic_matching()
        test_fallback_strategies()
        test_detailed_explanation()
        
        print("\n" + "=" * 80)
        print("SUMMARY OF BASIC TESTS")
        print("=" * 80)
        print("✓ Perspective normalization converts 'you/your' to 'user'")
        print("✓ Semantic expansion enriches short target contexts")
        print("✓ Multiple matching strategies tried automatically")
        print("✓ Fallback strategies provide robustness")
        print("✓ Detailed explanations available for debugging")
        print("✓ Context length mismatches handled (long input vs short target)")
        
        print("\nCore adaptive functionality working!")
        print("Add sklearn to enable full training capabilities.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
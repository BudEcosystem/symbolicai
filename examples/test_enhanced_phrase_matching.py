#!/usr/bin/env python3
"""
Test suite demonstrating enhanced multi-word phrase matching with FAISS integration.
Shows how FAISS improves phrase boundary detection and semantic matching.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import UnifiedBudExpression
from semantic_bud_expressions.enhanced_unified_registry import EnhancedUnifiedParameterTypeRegistry
from semantic_bud_expressions.enhanced_unified_parameter_type import EnhancedUnifiedParameterType


def test_basic_phrase_matching():
    """Test basic multi-word phrase matching"""
    print("=== Basic Multi-Word Phrase Matching ===\n")
    
    # Create enhanced registry
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create phrase parameter type with known car models
    car_models = [
        "Tesla Model 3", "Tesla Model S", "Tesla Model X", "Tesla Model Y",
        "BMW 3 Series", "BMW 5 Series", "BMW X5", "BMW M3",
        "Mercedes S Class", "Mercedes E Class", "Mercedes C Class",
        "Audi A4", "Audi Q5", "Audi RS6",
        "Rolls Royce Phantom", "Rolls Royce Ghost", "Rolls Royce Cullinan",
        "Ferrari 488", "Ferrari F8 Tributo", "Lamborghini Huracan",
        "Porsche 911", "Porsche Cayenne", "Porsche Taycan"
    ]
    
    registry.create_phrase_parameter_type(
        "car_model",
        max_phrase_length=5,
        known_phrases=car_models,
        use_faiss=True
    )
    
    # Test expressions
    expr = UnifiedBudExpression("I drive a {car_model:phrase}", registry)
    
    test_cases = [
        "I drive a Tesla Model 3",
        "I drive a BMW X5",
        "I drive a Rolls Royce Phantom",
        "I drive a Mercedes S Class",
        "I drive a Porsche 911 Turbo"  # Should match "Porsche 911"
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"✓ Matched: '{text}' → car_model: '{match[0].value}'")
        else:
            print(f"✗ Failed: '{text}'")
    print()


def test_semantic_phrase_matching():
    """Test semantic phrase matching (phrases with semantic similarity)"""
    print("=== Semantic Phrase Matching ===\n")
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create semantic phrase type for product descriptions
    registry.create_semantic_phrase_parameter_type(
        "product",
        semantic_categories=[
            "smartphone", "laptop", "tablet", "headphones", "smartwatch",
            "gaming console", "camera", "television", "speaker"
        ],
        max_phrase_length=6,
        similarity_threshold=0.4
    )
    
    expr = UnifiedBudExpression("I want to buy {product:phrase}", registry)
    
    test_cases = [
        "I want to buy iPhone 15 Pro Max",          # Semantic: smartphone
        "I want to buy MacBook Pro 16 inch",        # Semantic: laptop
        "I want to buy Sony WH-1000XM5 headphones", # Direct match
        "I want to buy Samsung Galaxy Tab S9",      # Semantic: tablet
        "I want to buy PlayStation 5 Pro"           # Semantic: gaming console
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"✓ Matched: '{text}' → product: '{match[0].value}'")
        else:
            print(f"✗ Failed: '{text}'")
    print()


def test_context_aware_phrase_matching():
    """Test context-aware phrase boundary detection"""
    print("=== Context-Aware Phrase Matching ===\n")
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Create phrase type with context scoring enabled
    registry.create_phrase_parameter_type(
        "location",
        max_phrase_length=6,
        use_context_scoring=True,
        known_phrases=[
            "New York City", "Los Angeles", "San Francisco Bay Area",
            "London Bridge", "Paris France", "Tokyo Japan",
            "Sydney Opera House", "Golden Gate Bridge",
            "Empire State Building", "Central Park"
        ]
    )
    
    expr = UnifiedBudExpression("I visited {location:phrase} last summer", registry)
    
    test_cases = [
        "I visited New York City last summer",
        "I visited the Golden Gate Bridge last summer",
        "I visited Central Park in Manhattan last summer",  # Should extract "Central Park"
        "I visited Sydney Opera House last summer"
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"✓ Matched: '{text}' → location: '{match[0].value}'")
        else:
            print(f"✗ Failed: '{text}'")
    print()


def test_mixed_parameter_types():
    """Test expressions with mixed parameter types including FAISS phrases"""
    print("=== Mixed Parameter Types with FAISS ===\n")
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Clear any existing types to avoid conflicts
    registry.clear_unified_cache()
    
    # Create semantic emotion type
    registry.create_semantic_parameter_type(
        "emotion",
        ["happy", "excited", "thrilled", "delighted", "joyful"],
        similarity_threshold=0.5
    )
    
    # Create phrase type for car descriptions
    registry.create_phrase_parameter_type(
        "car_description",
        max_phrase_length=5,
        known_phrases=[
            "red sports car", "blue sedan", "black SUV",
            "white electric vehicle", "silver luxury car",
            "green hybrid", "yellow convertible"
        ]
    )
    
    expr = UnifiedBudExpression(
        "I am {emotion:semantic} about my new {car_description:phrase}",
        registry
    )
    
    test_cases = [
        "I am thrilled about my new red sports car",
        "I am ecstatic about my new white electric vehicle",  # ecstatic ~ excited
        "I am happy about my new silver luxury car"
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"✓ Matched: '{text}'")
            print(f"  emotion: '{match[0].value}'")
            print(f"  car_description: '{match[1].value}'")
        else:
            print(f"✗ Failed: '{text}'")
    print()


def test_faiss_performance():
    """Test FAISS performance with large phrase vocabulary"""
    print("=== FAISS Performance Test ===\n")
    
    import time
    
    registry = EnhancedUnifiedParameterTypeRegistry(
        faiss_auto_threshold=50  # Lower threshold for demo
    )
    registry.initialize_model()
    
    # Clear any existing types to avoid conflicts
    registry.clear_unified_cache()
    
    # Create large phrase vocabulary
    large_vocabulary = []
    for brand in ["Apple", "Samsung", "Google", "Microsoft", "Sony"]:
        for product in ["Phone", "Tablet", "Laptop", "Watch", "Earbuds"]:
            for model in range(1, 11):
                large_vocabulary.append(f"{brand} {product} {model}")
                large_vocabulary.append(f"{brand} {product} Pro {model}")
                large_vocabulary.append(f"{brand} {product} Max {model}")
    
    print(f"Creating phrase type with {len(large_vocabulary)} known phrases...")
    
    start = time.time()
    registry.create_phrase_parameter_type(
        "tech_product",
        max_phrase_length=4,
        known_phrases=large_vocabulary
    )
    create_time = time.time() - start
    print(f"Creation time: {create_time:.3f}s")
    
    # Test matching speed
    expr = UnifiedBudExpression("I bought a {tech_product:phrase}", registry)
    
    test_phrases = [
        "I bought a Apple Phone Pro 5",
        "I bought a Samsung Tablet Max 8",
        "I bought a Google Watch 3",
        "I bought a Microsoft Laptop Pro 7"
    ]
    
    start = time.time()
    matches = 0
    for text in test_phrases:
        if expr.match(text):
            matches += 1
    match_time = time.time() - start
    
    print(f"Matched {matches}/{len(test_phrases)} phrases in {match_time:.3f}s")
    print(f"Average match time: {match_time/len(test_phrases)*1000:.1f}ms per phrase")
    
    # Show FAISS statistics
    stats = registry.get_faiss_statistics()
    print(f"\nFAISS Statistics:")
    print(f"- FAISS-enabled types: {stats['total_faiss_types']}")
    print(f"- Auto-threshold: {stats['faiss_auto_threshold']}")
    print()


def test_phrase_boundary_detection():
    """Test intelligent phrase boundary detection"""
    print("=== Intelligent Phrase Boundary Detection ===\n")
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    registry.clear_unified_cache()
    
    # Create phrase type with various delimiters
    registry.create_phrase_parameter_type(
        "company",
        max_phrase_length=5,
        phrase_delimiters=['.', ',', '!', '?', '(', ')', ';', ':'],
        known_phrases=[
            "Apple Inc", "Microsoft Corporation", "Google LLC",
            "Tesla Motors", "Amazon Web Services", "Meta Platforms",
            "OpenAI", "Anthropic", "SpaceX"
        ]
    )
    
    expr = UnifiedBudExpression("{company:phrase} announced", registry)
    
    test_cases = [
        "Apple Inc announced new products",
        "Microsoft Corporation announced quarterly results",
        "Tesla Motors (TSLA) announced record sales",  # Delimiter test
        "Amazon Web Services, announced new features",  # Comma delimiter
        "OpenAI announced GPT-5; it's revolutionary"   # Semicolon delimiter
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"✓ Matched: '{text}' → company: '{match[0].value}'")
        else:
            print(f"✗ Failed: '{text}'")
    print()


def test_adaptive_phrase_length():
    """Test adaptive phrase length handling"""
    print("=== Adaptive Phrase Length ===\n")
    
    registry = EnhancedUnifiedParameterTypeRegistry()
    registry.initialize_model()
    registry.clear_unified_cache()
    
    # Create phrase type that adapts to different lengths
    addresses = [
        # Short addresses
        "Main Street", "Park Avenue", "Broadway",
        # Medium addresses
        "123 Main Street", "456 Park Avenue", "789 Broadway",
        # Long addresses
        "123 Main Street Suite 100", "456 Park Avenue Building A",
        "789 Broadway Floor 5 Office 501"
    ]
    
    registry.create_phrase_parameter_type(
        "address",
        max_phrase_length=8,  # Allow longer addresses
        known_phrases=addresses
    )
    
    expr = UnifiedBudExpression("Deliver to {address:phrase}", registry)
    
    test_cases = [
        "Deliver to Main Street",
        "Deliver to 123 Main Street",
        "Deliver to 123 Main Street Suite 100",
        "Deliver to 789 Broadway Floor 5 Office 501"
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            words = len(match[0].value.split())
            print(f"✓ Matched: '{text}' → address: '{match[0].value}' ({words} words)")
        else:
            print(f"✗ Failed: '{text}'")
    print()


def main():
    """Run all tests"""
    print("Enhanced Multi-Word Phrase Matching with FAISS\n")
    print("=" * 50)
    print()
    
    test_basic_phrase_matching()
    test_semantic_phrase_matching()
    test_context_aware_phrase_matching()
    test_mixed_parameter_types()
    test_faiss_performance()
    test_phrase_boundary_detection()
    test_adaptive_phrase_length()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Demo of new features: Context-aware matching and FAISS integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    UnifiedParameterTypeRegistry,
    ContextAwareExpression,
    SemanticParameterType,
    SemanticParameterTypeRegistry
)

print("=== Context-Aware Matching Demo ===\n")

# Initialize registry
registry = UnifiedParameterTypeRegistry()
registry.initialize_model()

# Create context-aware expression
expr = ContextAwareExpression(
    expression="I {emotion} {vehicle}",
    expected_context="cars and automotive technology",
    context_threshold=0.6,
    registry=registry
)

# Test with different contexts
texts = [
    "The automotive industry has revolutionized transportation. Cars are amazing. I love Tesla",
    "The weather is beautiful today. Birds are singing. I love Tesla",
    "Let's discuss electric vehicles and sustainable transport. I adore BMW"
]

for i, text in enumerate(texts, 1):
    print(f"Text {i}: {text}")
    result = expr.match_with_context(text)
    if result:
        print(f"  ✓ Matched: '{result.matched_text}' (context similarity: {result.context_similarity:.2f})")
        print(f"  Context: '{result.context_text[:50]}...'")
        print(f"  Parameters: {result.parameters}")
    else:
        print("  ✗ No match (context doesn't match)")
    print()

print("\n=== FAISS Integration Demo ===\n")

# Create semantic registry
semantic_registry = SemanticParameterTypeRegistry()
semantic_registry.initialize_model()

# Create semantic type with many prototypes (auto-enables FAISS)
car_brands = [
    "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "BMW", "Mercedes",
    "Audi", "Volkswagen", "Tesla", "Volvo", "Mazda", "Subaru", "Hyundai",
    "Kia", "Porsche", "Ferrari", "Lamborghini", "Maserati", "Bentley"
] * 5  # Repeat to create 100+ prototypes

print(f"Creating semantic type with {len(car_brands)} prototypes...")
car_type = SemanticParameterType(
    name="car_brand",
    prototypes=car_brands,
    similarity_threshold=0.7
)

# Register the type
semantic_registry.define_parameter_type(car_type)

# Test matching
test_brands = ["Tesla", "Rolls-Royce", "Bicycle", "Ferrari"]
print(f"\nTesting semantic matching (FAISS enabled: {car_type.use_faiss}):")
for brand in test_brands:
    matches, score, closest = car_type.matches_semantically(brand)
    print(f"  '{brand}': {'✓' if matches else '✗'} (score: {score:.3f}, closest: {closest})")

print("\n=== Combined Demo: Context-Aware with FAISS ===\n")

# Create context-aware expression with semantic types
# First create a semantic parameter type in the unified registry
registry.create_semantic_parameter_type(
    "car_brand", 
    prototypes=["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Tesla", "Ferrari"],
    similarity_threshold=0.7
)
expr2 = ContextAwareExpression(
    expression="I want to buy a {car_brand:semantic}",
    expected_context="car dealership shopping automotive purchase",
    context_threshold=0.5,
    registry=registry
)

texts2 = [
    "Welcome to our car dealership! We have great deals. I want to buy a Mercedes",
    "I'm reading a book about history. I want to buy a Mercedes",
]

for i, text in enumerate(texts2, 1):
    print(f"Text {i}: {text}")
    result = expr2.match_with_context(text)
    if result:
        print(f"  ✓ Matched with context!")
        print(f"  Parameters: {result.parameters}")
    else:
        print("  ✗ No match (wrong context)")
    print()

print("Demo complete!")
#!/usr/bin/env python3
"""
Demonstrates TRUE Cucumber-style exact matching (not semantic).
Parameters match ONLY the predefined list items (case-insensitive).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    budExpression,
    ParameterType,
    ParameterTypeRegistry,
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry
)


def create_cucumber_style_registry():
    """Create a registry with exact Cucumber-style matching"""
    registry = ParameterTypeRegistry()
    
    # Define fruits - exact matching only (case-insensitive)
    fruits = ["apple", "banana", "orange", "grape", "mango", "watermelon"]
    fruits_pattern = "|".join(fruits)  # Creates: apple|banana|orange|...
    
    fruit_type = ParameterType(
        name="fruit",
        regexp=f"({fruits_pattern})",  # Remove (?i) - handle case in transformer
        type=str,
        transformer=lambda value: value.lower()  # Normalize to lowercase
    )
    registry.define_parameter_type(fruit_type)
    
    # Define vehicles - exact matching only
    vehicles = ["car", "truck", "bus", "motorcycle", "bicycle", "scooter"]
    vehicles_pattern = "|".join(vehicles)
    
    vehicle_type = ParameterType(
        name="vehicle",
        regexp=f"(?i)({vehicles_pattern})",
        type=str,
        transformer=lambda value: value.lower()
    )
    registry.define_parameter_type(vehicle_type)
    
    # Define colors - exact matching only
    colors = ["red", "blue", "green", "yellow", "black", "white", "purple"]
    colors_pattern = "|".join(colors)
    
    color_type = ParameterType(
        name="color",
        regexp=f"(?i)({colors_pattern})",
        type=str,
        transformer=lambda value: value.lower()
    )
    registry.define_parameter_type(color_type)
    
    return registry


def demonstrate_exact_matching():
    """Show exact Cucumber-style matching"""
    print("Cucumber-Style Exact Matching (NOT Semantic)")
    print("=" * 60)
    print()
    
    # Create registry with exact matching
    registry = create_cucumber_style_registry()
    
    # Test 1: Fruit matching
    print("1. Fruit Matching (exact only):")
    expr = budExpression("I love {fruit}", registry)
    
    # These will match (exact, case-insensitive)
    test_cases = [
        ("I love apple", True),
        ("I love APPLE", True),  # Case-insensitive
        ("I love Apple", True),  # Case-insensitive
        ("I love banana", True),
        ("I love watermelon", True),
        # These will NOT match (not in list)
        ("I love strawberry", False),  # Not in predefined list
        ("I love pear", False),        # Not in predefined list
        ("I love apples", False),      # Plural not in list
    ]
    
    for text, should_match in test_cases:
        match = expr.match(text)
        status = "✓" if (match is not None) == should_match else "✗"
        result = f"→ {match[0].value}" if match else "→ No match"
        print(f"  {status} '{text}' {result}")
    print()
    
    # Test 2: Vehicle matching
    print("2. Vehicle Matching (exact only):")
    expr2 = budExpression("I drive a {vehicle}", registry)
    
    test_cases2 = [
        ("I drive a car", True),
        ("I drive a CAR", True),        # Case-insensitive
        ("I drive a motorcycle", True),
        # These will NOT match
        ("I drive a Tesla", False),     # Brand name not in list
        ("I drive a sedan", False),     # Type not in list
        ("I drive a vehicle", False),   # Generic term not in list
    ]
    
    for text, should_match in test_cases2:
        match = expr2.match(text)
        status = "✓" if (match is not None) == should_match else "✗"
        result = f"→ {match[0].value}" if match else "→ No match"
        print(f"  {status} '{text}' {result}")
    print()
    
    # Test 3: Multiple parameters
    print("3. Multiple Parameters (all exact):")
    expr3 = budExpression("The {color} {vehicle} carried {fruit}", registry)
    
    test_cases3 = [
        ("The red car carried apple", True),
        ("The BLUE TRUCK carried BANANA", True),  # Case-insensitive
        ("The green bicycle carried orange", True),
        # These will NOT match
        ("The crimson car carried apple", False),  # 'crimson' not in colors
        ("The red sedan carried apple", False),     # 'sedan' not in vehicles
        ("The red car carried strawberry", False),  # 'strawberry' not in fruits
    ]
    
    for text, should_match in test_cases3:
        match = expr3.match(text)
        status = "✓" if (match is not None) == should_match else "✗"
        if match:
            result = f"→ color={match[0].value}, vehicle={match[1].value}, fruit={match[2].value}"
        else:
            result = "→ No match"
        print(f"  {status} '{text}' {result}")
    print()


def create_exact_phrase_matcher():
    """Create exact phrase matching (multi-word Cucumber style)"""
    print("4. Exact Multi-Word Phrase Matching:")
    
    # For multi-word exact matching
    registry = UnifiedParameterTypeRegistry()
    
    # Define exact car models (no semantic matching)
    car_models = [
        "Tesla Model 3",
        "BMW X5", 
        "Mercedes S Class",
        "Toyota Camry",
        "Honda Civic"
    ]
    
    # Create regex pattern for exact matching
    # Escape special characters and join with |
    import re
    # Create case variations for each model
    car_models_ci = []
    for model in car_models:
        car_models_ci.extend([model, model.upper(), model.lower(), model.title()])
    escaped_models = [re.escape(model) for model in car_models_ci]
    pattern = "|".join(escaped_models)
    
    registry.create_regex_parameter_type(
        name="car_model",
        pattern=pattern
    )
    
    expr = UnifiedBudExpression("I drive a {car_model:regex}", registry)
    
    test_cases = [
        ("I drive a Tesla Model 3", True),
        ("I drive a TESLA MODEL 3", True),  # Case-insensitive
        ("I drive a BMW X5", True),
        # These will NOT match
        ("I drive a Tesla Model S", False),  # Model S not in list
        ("I drive a Ferrari", False),        # Not in list
        ("I drive a BMW", False),            # Incomplete match
    ]
    
    for text, should_match in test_cases:
        match = expr.match(text)
        status = "✓" if (match is not None) == should_match else "✗"
        result = f"→ {match[0].value}" if match else "→ No match"
        print(f"  {status} '{text}' {result}")
    print()


def create_large_vocabulary_exact_matcher():
    """Show how to handle large vocabularies efficiently"""
    print("5. Large Vocabulary Exact Matching (1000+ items):")
    
    registry = ParameterTypeRegistry()
    
    # Create a large product catalog
    products = [
        f"Product {i:04d}" for i in range(1000)
    ] + [
        "iPhone 15", "MacBook Pro", "iPad Air", "AirPods Pro",
        "Samsung Galaxy S24", "Google Pixel 8", "Microsoft Surface"
    ]
    
    # For large lists, chunk the pattern to avoid regex size limits
    chunk_size = 100
    chunks = [products[i:i+chunk_size] for i in range(0, len(products), chunk_size)]
    
    # Create multiple patterns and combine
    patterns = []
    for chunk in chunks:
        escaped_chunk = [re.escape(p) for p in chunk]
        patterns.append("|".join(escaped_chunk))
    
    # Combine all patterns
    full_pattern = f"(?i)({"|".join(patterns)})"
    
    product_type = ParameterType(
        name="product",
        regexp=full_pattern,
        type=str,
        transformer=lambda value: value.lower()
    )
    registry.define_parameter_type(product_type)
    
    expr = budExpression("Buy {product} now", registry)
    
    # Test with some products
    test_cases = [
        "Buy Product 0500 now",
        "Buy iPhone 15 now",
        "Buy MACBOOK PRO now",
        "Buy Product 9999 now",  # Not in list
        "Buy iPhone 16 now",     # Not in list
    ]
    
    print(f"  Testing with {len(products)} predefined products...")
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"  ✓ '{text}' → {match[0].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    print()


def main():
    """Run all demonstrations"""
    demonstrate_exact_matching()
    create_exact_phrase_matcher()
    create_large_vocabulary_exact_matcher()
    
    print("=" * 60)
    print("Summary: True Cucumber-Style Matching")
    print("- Exact matching from predefined lists only")
    print("- Case-insensitive but NOT semantic")
    print("- 'strawberry' does NOT match {fruit} if not in list")
    print("- 'Tesla' does NOT match {vehicle} if only 'car' is defined")
    print("- Use regex patterns with alternation (|) for exact matching")


if __name__ == "__main__":
    main()
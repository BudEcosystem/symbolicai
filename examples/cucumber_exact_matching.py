#!/usr/bin/env python3
"""
TRUE Cucumber-style exact matching (NOT semantic).
Parameters match ONLY items from predefined lists.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    budExpression,
    ParameterType,
    ParameterTypeRegistry
)


def main():
    print("True Cucumber-Style Exact Matching")
    print("=" * 60)
    print()
    
    # Create registry
    registry = ParameterTypeRegistry()
    
    # Method 1: Simple exact matching (case-sensitive)
    print("1. Basic Exact Matching (case-sensitive):")
    
    # Define fruits - exact list only
    fruits = ["apple", "banana", "orange", "grape", "mango"]
    fruit_type = ParameterType(
        name="fruit",
        regexp="|".join(fruits),  # Creates: apple|banana|orange|grape|mango
        type=str
    )
    registry.define_parameter_type(fruit_type)
    
    expr = budExpression("I love {fruit}", registry)
    
    test_cases = [
        "I love apple",      # ✓ Match
        "I love banana",     # ✓ Match
        "I love strawberry", # ✗ No match - not in list
        "I love Apple",      # ✗ No match - case sensitive
        "I love apples",     # ✗ No match - plural not in list
    ]
    
    for text in test_cases:
        match = expr.match(text)
        if match:
            print(f"  ✓ '{text}' → {match[0].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    print()
    
    # Method 2: Case-insensitive exact matching
    print("2. Case-Insensitive Exact Matching:")
    
    # Define vehicles with all case variations
    vehicles_base = ["car", "truck", "bus", "motorcycle"]
    vehicles_all = []
    for v in vehicles_base:
        vehicles_all.extend([v, v.upper(), v.capitalize()])
    
    vehicle_type = ParameterType(
        name="vehicle",
        regexp="|".join(vehicles_all),
        type=str,
        transformer=lambda v: v.lower()  # Normalize output
    )
    registry.define_parameter_type(vehicle_type)
    
    expr2 = budExpression("I drive a {vehicle}", registry)
    
    test_cases2 = [
        "I drive a car",        # ✓ Match
        "I drive a CAR",        # ✓ Match
        "I drive a Car",        # ✓ Match
        "I drive a Tesla",      # ✗ No match - not in list
        "I drive a sedan",      # ✗ No match - not in list
    ]
    
    for text in test_cases2:
        match = expr2.match(text)
        if match:
            print(f"  ✓ '{text}' → {match[0].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    print()
    
    # Method 3: Multi-word exact matching
    print("3. Multi-Word Exact Matching:")
    
    # Define exact car models
    car_models = [
        "Tesla Model 3",
        "BMW X5",
        "Mercedes S Class",
        "Toyota Camry"
    ]
    
    # Create pattern with escaped special characters
    import re
    escaped_models = [re.escape(model) for model in car_models]
    
    car_type = ParameterType(
        name="car",
        regexp="|".join(escaped_models),
        type=str
    )
    registry.define_parameter_type(car_type)
    
    expr3 = budExpression("I drive a {car}", registry)
    
    test_cases3 = [
        "I drive a Tesla Model 3",    # ✓ Match
        "I drive a BMW X5",           # ✓ Match
        "I drive a Tesla Model S",    # ✗ No match - different model
        "I drive a Ferrari",          # ✗ No match - not in list
    ]
    
    for text in test_cases3:
        match = expr3.match(text)
        if match:
            print(f"  ✓ '{text}' → {match[0].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    print()
    
    # Method 4: Multiple parameters with exact matching
    print("4. Multiple Parameters (all exact):")
    
    # Define colors
    colors = ["red", "blue", "green", "yellow"]
    color_type = ParameterType(
        name="color",
        regexp="|".join(colors),
        type=str
    )
    registry.define_parameter_type(color_type)
    
    expr4 = budExpression("The {color} {vehicle} has {fruit}", registry)
    
    test_cases4 = [
        "The red car has apple",       # ✓ All match
        "The blue truck has banana",   # ✓ All match
        "The crimson car has apple",   # ✗ 'crimson' not in colors
        "The red Tesla has apple",     # ✗ 'Tesla' not in vehicles
        "The red car has strawberry",  # ✗ 'strawberry' not in fruits
    ]
    
    for text in test_cases4:
        match = expr4.match(text)
        if match:
            print(f"  ✓ '{text}' → color={match[0].value}, vehicle={match[1].value}, fruit={match[2].value}")
        else:
            print(f"  ✗ '{text}' → No match")
    print()
    
    print("=" * 60)
    print("Key Points:")
    print("✓ ONLY exact matches from predefined lists")
    print("✓ NO semantic similarity ('strawberry' ≠ 'fruit')")
    print("✓ Case handling through explicit variations")
    print("✓ Multi-word support with escaped patterns")
    print("✓ True Cucumber-style behavior")


if __name__ == "__main__":
    main()
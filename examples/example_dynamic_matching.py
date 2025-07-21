#!/usr/bin/env python3
"""
Example demonstrating dynamic semantic matching
"""

from semantic_bud_expressions import (
    SemanticBudExpression,
    SemanticParameterTypeRegistry
)

def demo_dynamic_matching():
    """Demonstrate dynamic semantic matching"""
    print("=== Dynamic Semantic Matching Demo ===\n")
    
    # Initialize registry
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    # Example 1: Basic dynamic matching
    print("1. Basic Dynamic Matching:")
    print("   Pattern: 'I love {cars}'")
    expr1 = SemanticBudExpression("I love {cars}", registry)
    
    test_phrases = [
        "I love Ferrari",
        "I love Tesla", 
        "I love Porsche",
        "I love Rolls Royce"
    ]
    
    for phrase in test_phrases:
        match = expr1.match(phrase)
        if match:
            print(f"   ✓ '{phrase}' matched! Extracted: {match[0].value}")
        else:
            print(f"   ✗ '{phrase}' did not match")
    
    # Example 2: Multiple dynamic parameters
    print("\n2. Multiple Dynamic Parameters:")
    print("   Pattern: 'The {animal} ate the {food}'")
    expr2 = SemanticBudExpression("The {animal} ate the {food}", registry)
    
    match = expr2.match("The dog ate the pizza")
    if match:
        print(f"   ✓ Matched! Animal: {match[0].value}, Food: {match[1].value}")
    
    # Example 3: Mixing predefined and dynamic
    print("\n3. Mixing Predefined and Dynamic:")
    print("   Pattern: 'I ate {fruit} in my {vehicle}'")
    print("   Note: {fruit} is predefined, {vehicle} is dynamic")
    expr3 = SemanticBudExpression("I ate {fruit} in my {vehicle}", registry)
    
    match = expr3.match("I ate apple in my BMW")
    if match:
        print(f"   ✓ Matched! Fruit: {match[0].value}, Vehicle: {match[1].value}")
    
    # Example 4: Adjusting similarity threshold
    print("\n4. Adjusting Similarity Threshold:")
    print("   Default threshold (0.3):")
    expr4a = SemanticBudExpression("I see a {bird}", registry)
    match4a = expr4a.match("I see a eagle")
    print(f"   'I see a eagle' -> {'Matched' if match4a else 'No match'}")
    
    print("\n   Higher threshold (0.5):")
    registry.set_dynamic_threshold(0.5)
    registry._dynamic_types_cache.clear()  # Clear cache to use new threshold
    expr4b = SemanticBudExpression("I see a {bird}", registry)
    try:
        match4b = expr4b.match("I see a eagle")
        if match4b:
            _ = match4b[0].value  # This might raise ValueError
            print("   'I see a eagle' -> Matched")
    except ValueError as e:
        print(f"   'I see a eagle' -> No match (similarity too low)")
    
    # Example 5: Disabling dynamic matching
    print("\n5. Disabling Dynamic Matching:")
    registry.enable_dynamic_matching(False)
    try:
        expr5 = SemanticBudExpression("I drive a {automobile}", registry)
        print("   Expression created (shouldn't happen)")
    except Exception as e:
        print(f"   ✓ Expected error: {type(e).__name__}: Undefined parameter type 'automobile'")
    
    # Re-enable for next examples
    registry.enable_dynamic_matching(True)
    registry.set_dynamic_threshold(0.3)  # Reset to default
    
    # Example 6: Real-world use case
    print("\n6. Real-world Use Case - Shopping Intent:")
    print("   Pattern: 'I need a new {product}'")
    # Lower threshold for this example
    registry.set_dynamic_threshold(0.2)
    registry._dynamic_types_cache.clear()
    expr6 = SemanticBudExpression("I need a new {product}", registry)
    
    real_world_tests = [
        "I need a new laptop",
        "I need a new phone",
        "I need a new car",
        "I need a new television"
    ]
    
    for phrase in real_world_tests:
        match = expr6.match(phrase)
        if match:
            try:
                print(f"   ✓ '{phrase}' -> Product: {match[0].value}")
            except ValueError:
                print(f"   ✗ '{phrase}' -> Matched pattern but similarity too low")

if __name__ == "__main__":
    demo_dynamic_matching()
#!/usr/bin/env python3
"""
Quick Start Example - Semantic Bud Expressions

This example shows the basic usage of all parameter types.
"""

from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry
)

def main():
    print("=== Semantic Bud Expressions - Quick Start ===\n")
    
    # Initialize registry
    registry = UnifiedParameterTypeRegistry()
    print("1. Initializing Model2Vec embeddings...")
    registry.initialize_model()
    print("   ✓ Model loaded\n")
    
    # Example 1: Simple semantic matching
    print("2. Simple Semantic Matching:")
    expr1 = UnifiedBudExpression("I love {fruit}", registry)
    
    test_inputs = ["I love apples", "I love oranges", "I love mango"]
    for input_text in test_inputs:
        match = expr1.match(input_text)
        if match:
            print(f"   ✓ '{input_text}' matched: fruit = {match[0].value}")
    
    # Example 2: Multi-word phrases
    print("\n3. Multi-word Phrase Matching:")
    registry.create_phrase_parameter_type('product', max_phrase_length=5)
    expr2 = UnifiedBudExpression("Buy {product:phrase} online", registry)
    
    match = expr2.match("Buy MacBook Pro 16 inch online")
    if match:
        print(f"   ✓ Matched phrase: product = '{match[0].value}'")
    
    # Example 3: Dynamic semantic matching
    print("\n4. Dynamic Semantic Matching:")
    registry.set_dynamic_threshold(0.3)
    expr3 = UnifiedBudExpression("{vehicle:dynamic} is fast", registry)
    
    test_vehicles = ["Ferrari is fast", "Porsche is fast", "Tesla is fast"]
    for input_text in test_vehicles:
        match = expr3.match(input_text)
        if match:
            print(f"   ✓ '{input_text}' matched: vehicle = {match[0].value}")
    
    # Example 4: Complex expression
    print("\n5. Complex Expression with Multiple Types:")
    registry.create_semantic_parameter_type(
        'action', 
        ['buy', 'purchase', 'get', 'obtain'], 
        similarity_threshold=0.3
    )
    
    expr4 = UnifiedBudExpression(
        "{customer:dynamic} wants to {action:semantic} {count} {items:phrase}",
        registry
    )
    
    match = expr4.match("shopper wants to acquire 3 luxury watches")
    if match:
        print(f"   ✓ Complex match:")
        print(f"     - customer: {match[0].value}")
        print(f"     - action: {match[1].value} (semantically matched with 'acquire')")
        print(f"     - count: {match[2].value}")
        print(f"     - items: {match[3].value}")
    
    print("\n✅ All examples completed successfully!")
    print("\nPerformance Note: After warm-up, matching takes ~0.02-0.05ms per expression")

if __name__ == "__main__":
    main()
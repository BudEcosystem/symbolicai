#!/usr/bin/env python3
"""
Demo of the unified expression system with multi-word phrase matching
"""

from semantic_bud_expressions import (
    UnifiedBudExpression, 
    UnifiedParameterTypeRegistry,
    ParameterTypeHint
)

def demo_unified_system():
    """Demonstrate the unified expression system"""
    print("=== Unified Expression System Demo ===\n")
    
    # Initialize registry
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    print("1. Multi-word Phrase Matching:")
    print("   Pattern: 'I drive a {car:phrase}'")
    
    # Create phrase parameter type
    registry.create_phrase_parameter_type('car', max_phrase_length=4)
    
    expr = UnifiedBudExpression('I drive a {car:phrase}', registry)
    
    # Test various inputs
    test_cases = [
        "I drive a Tesla",
        "I drive a red Ferrari", 
        "I drive a Rolls Royce",
        "I drive a Mercedes Benz"
    ]
    
    for test_text in test_cases:
        try:
            match = expr.match(test_text)
            if match:
                print(f"   ✓ '{test_text}' -> Car: {match[0].group.value}")
            else:
                print(f"   ✗ '{test_text}' -> No match")
        except Exception as e:
            print(f"   ✗ '{test_text}' -> Error: {e}")
    
    print("\n2. Semantic Matching with Type Hints:")
    print("   Pattern: 'I love {my_fruit:semantic}'")
    
    # Create semantic parameter type with different name
    registry.create_semantic_parameter_type(
        'my_fruit',
        ['apple', 'banana', 'orange', 'grape'],
        similarity_threshold=0.5
    )
    
    expr2 = UnifiedBudExpression('I love {my_fruit:semantic}', registry)
    
    semantic_tests = [
        "I love apples",
        "I love oranges", 
        "I love mango",
        "I love computer"  # Should fail
    ]
    
    for test_text in semantic_tests:
        try:
            match = expr2.match(test_text)
            if match:
                print(f"   ✓ '{test_text}' -> Fruit: {match[0].group.value}")
            else:
                print(f"   ✗ '{test_text}' -> No match")
        except Exception as e:
            print(f"   ✗ '{test_text}' -> Error: {e}")
    
    print("\n3. Dynamic Semantic Matching:")
    print("   Pattern: 'I need {furniture:dynamic}'")
    
    expr3 = UnifiedBudExpression('I need {furniture:dynamic}', registry)
    
    dynamic_tests = [
        "I need chair",
        "I need table",
        "I need sofa"
    ]
    
    for test_text in dynamic_tests:
        try:
            match = expr3.match(test_text)
            if match:
                print(f"   ✓ '{test_text}' -> Furniture: {match[0].group.value}")
            else:
                print(f"   ✗ '{test_text}' -> No match")
        except Exception as e:
            print(f"   ✗ '{test_text}' -> Error: {e}")
    
    print("\n4. Mixed Parameter Types:")
    print("   Pattern: 'I {action:semantic} {count} {items:phrase}'")
    
    registry.create_semantic_parameter_type(
        'action',
        ['buy', 'sell', 'want', 'need'],
        similarity_threshold=0.3
    )
    
    registry.create_phrase_parameter_type('items', max_phrase_length=3)
    
    expr4 = UnifiedBudExpression('I {action:semantic} {count} {items:phrase}', registry)
    
    mixed_tests = [
        "I purchased 5 red cars",
        "I want 2 sports cars",
        "I need 1 luxury vehicle"
    ]
    
    for test_text in mixed_tests:
        try:
            match = expr4.match(test_text)
            if match:
                print(f"   ✓ '{test_text}' -> Action: {match[0].group.value}, Count: {match[1].group.value}, Items: {match[2].group.value}")
            else:
                print(f"   ✗ '{test_text}' -> No match")
        except Exception as e:
            print(f"   ✗ '{test_text}' -> Error: {e}")
    
    print("\n5. Backward Compatibility:")
    print("   Using old SemanticBudExpression:")
    
    from semantic_bud_expressions import SemanticBudExpression, SemanticParameterTypeRegistry
    
    old_registry = SemanticParameterTypeRegistry()
    old_registry.initialize_model()
    
    old_expr = SemanticBudExpression('I love {fruit}', old_registry)
    match = old_expr.match('I love banana')
    if match:
        print(f"   ✓ Old system still works: {match[0].value}")
    
    print("\n=== Summary ===")
    print("The unified system supports:")
    print("• Multi-word phrase matching with {param:phrase}")
    print("• Semantic similarity matching with {param:semantic}")
    print("• Dynamic semantic matching with {param:dynamic}")
    print("• Mixed parameter types in single expressions")
    print("• Full backward compatibility with existing API")
    print("• Configurable similarity thresholds and phrase boundaries")

if __name__ == "__main__":
    demo_unified_system()
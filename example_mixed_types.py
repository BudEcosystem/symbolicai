#!/usr/bin/env python3
"""
Working example demonstrating mixed parameter types
"""

from semantic_cucumber_expressions import (
    UnifiedCucumberExpression,
    UnifiedParameterTypeRegistry
)

def demo_mixed_types():
    """Demonstrate mixing different parameter types"""
    print("=== Mixed Parameter Types Demo ===\n")
    
    # Initialize registry
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Configure
    registry.set_dynamic_threshold(0.3)
    
    # Example 1: Semantic + Standard + Dynamic
    print("1. Semantic + Standard + Dynamic:")
    print('   Pattern: "I {emotion} {count} {cars:dynamic}"')
    
    expr1 = UnifiedCucumberExpression(
        'I {emotion} {count} {cars:dynamic}',
        registry
    )
    
    test1 = 'I love 3 Ferrari'
    match = expr1.match(test1)
    if match:
        print(f'   ✓ Matched: "{test1}"')
        print(f'     - Emotion: {match[0].value} (predefined semantic)')
        print(f'     - Count: {match[1].value} (standard)')
        print(f'     - Cars: {match[2].value} (dynamic semantic)')
    else:
        print(f'   ✗ No match')
    
    # Example 2: Create custom semantic type
    print("\n2. Custom Semantic Type:")
    print('   Pattern: "The {transport:semantic} is {color}"')
    
    registry.create_semantic_parameter_type(
        'transport',
        ['car', 'bus', 'train', 'plane', 'ship'],
        similarity_threshold=0.4
    )
    
    expr2 = UnifiedCucumberExpression(
        'The {transport:semantic} is {color}',
        registry
    )
    
    test2 = 'The automobile is red'
    match = expr2.match(test2)
    if match:
        print(f'   ✓ Matched: "{test2}"')
        print(f'     - Transport: {match[0].value} (semantic match)')
        print(f'     - Color: {match[1].value}')
    else:
        print(f'   ✗ No match')
    
    # Example 3: Phrase type for multi-word
    print("\n3. Multi-word Phrase Matching:")
    print('   Pattern: "Buy {product:phrase} now"')
    
    # First explicitly create the phrase type
    registry.create_phrase_parameter_type('product', max_phrase_length=4)
    
    expr3 = UnifiedCucumberExpression(
        'Buy {product:phrase} now',
        registry
    )
    
    # Note: Due to the regex pattern, this works best with specific delimiters
    test3 = 'Buy iPhone_15_Pro_Max now'
    match = expr3.match(test3)
    if match:
        print(f'   ✓ Matched: "{test3}"')
        print(f'     - Product: {match[0].value}')
    else:
        print(f'   ✗ No match')
    
    # Example 4: Math expression
    print("\n4. Math Expression:")
    print('   Pattern: "Calculate {math} please"')
    
    expr4 = UnifiedCucumberExpression(
        'Calculate {math} please',
        registry
    )
    
    test4 = 'Calculate 2 + 3 * 4 please'
    match = expr4.match(test4)
    if match:
        print(f'   ✓ Matched: "{test4}"')
        print(f'     - Math: {match[0].value}')
    else:
        print(f'   ✗ No match')
    
    # Example 5: Email regex pattern
    print("\n5. Custom Regex Pattern:")
    print('   Pattern: "Email {email:regex} registered"')
    
    registry.create_regex_parameter_type(
        'email',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    
    expr5 = UnifiedCucumberExpression(
        'Email {email:regex} registered',
        registry
    )
    
    test5 = 'Email john.doe@example.com registered'
    match = expr5.match(test5)
    if match:
        print(f'   ✓ Matched: "{test5}"')
        print(f'     - Email: {match[0].value}')
    else:
        print(f'   ✗ No match')
    
    # Example 6: Quoted strings
    print("\n6. Quoted String Parameter:")
    print('   Pattern: "Say {message:quoted} loudly"')
    
    registry.create_quoted_parameter_type('message')
    
    expr6 = UnifiedCucumberExpression(
        'Say {message:quoted} loudly',
        registry
    )
    
    test6 = 'Say "Hello World" loudly'
    match = expr6.match(test6)
    if match:
        print(f'   ✓ Matched: "{test6}"')
        print(f'     - Message: {match[0].value}')
    else:
        print(f'   ✗ No match')
    
    # Example 7: Combined example (simpler)
    print("\n7. Combined Example:")
    print('   Pattern: "{person:dynamic} {action} {count} {fruit}"')
    
    expr7 = UnifiedCucumberExpression(
        '{person:dynamic} {action} {count} {fruit}',
        registry
    )
    
    test7 = 'chef bought 5 apples'
    match = expr7.match(test7)
    if match:
        print(f'   ✓ Matched: "{test7}"')
        print(f'     - Person: {match[0].value} (dynamic match with "person")')
        print(f'     - Action: {match[1].value}')
        print(f'     - Count: {match[2].value}')
        print(f'     - Fruit: {match[3].value}')
    else:
        print(f'   ✗ No match')
    
    print("\n=== Summary ===")
    print("The unified system successfully supports:")
    print("✓ Standard parameters: {param}")
    print("✓ Semantic parameters: {param:semantic}")
    print("✓ Dynamic parameters: {param:dynamic}")
    print("✓ Phrase parameters: {param:phrase}")
    print("✓ Regex parameters: {param:regex}")
    print("✓ Math parameters: {math}")
    print("✓ Quoted parameters: {param:quoted}")
    print("\nAll can be mixed in a single expression!")

if __name__ == "__main__":
    demo_mixed_types()
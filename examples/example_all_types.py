#!/usr/bin/env python3
"""
Example demonstrating ALL parameter types combined in single expressions
"""

from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry,
    ParameterTypeHint
)

def demo_all_types():
    """Demonstrate combining all parameter types"""
    print("=== Combining All Parameter Types Demo ===\n")
    
    # Initialize registry
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Configure registry
    registry.set_dynamic_threshold(0.3)
    
    # Create various parameter types
    
    # 1. Semantic parameter type (use different name to avoid conflict)
    registry.create_semantic_parameter_type(
        'feeling',
        ['happy', 'sad', 'excited', 'angry', 'frustrated'],
        similarity_threshold=0.4
    )
    
    # 2. Phrase parameter type
    registry.create_phrase_parameter_type(
        'product_name',
        max_phrase_length=5
    )
    
    # 3. Regex parameter type
    registry.create_regex_parameter_type(
        'email',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    
    # 4. Quoted parameter type
    registry.create_quoted_parameter_type('message')
    
    # Example 1: Complex e-commerce expression
    print("1. E-commerce Order Expression:")
    print('   Pattern: I am {feeling:semantic} to order {quantity} {product_name:phrase} for ${price} to {email:regex}')
    
    expr1 = UnifiedBudExpression(
        'I am {feeling:semantic} to order {quantity} {product_name:phrase} for ${price} to {email:regex}',
        registry
    )
    
    test_order = 'I am thrilled to order 2 MacBook Pro 16 inch for $2999 to john.doe@example.com'
    
    try:
        match = expr1.match(test_order)
        if match:
            print(f'\n   ✓ Matched: "{test_order}"')
            print(f'     - Emotion: {match[0].group.value} (semantic match for "thrilled")')
            print(f'     - Quantity: {match[1].group.value}')
            print(f'     - Product: {match[2].group.value} (multi-word phrase)')
            print(f'     - Price: {match[3].group.value}')
            print(f'     - Email: {match[4].group.value} (regex validated)')
        else:
            print(f'   ✗ No match for: "{test_order}"')
    except Exception as e:
        print(f'   Error: {e}')
    
    # Example 2: Complex communication expression
    print("\n\n2. Communication Expression:")
    print('   Pattern: {greeting:semantic}, please {action:dynamic} the {item:phrase} with message {msg:quoted}')
    
    expr2 = UnifiedBudExpression(
        '{greeting:semantic}, please {action:dynamic} the {item:phrase} with message {msg:quoted}',
        registry
    )
    
    test_comm = 'Greetings, please send the quarterly sales report with message "Please review ASAP"'
    
    try:
        match = expr2.match(test_comm)
        if match:
            print(f'\n   ✓ Matched: "{test_comm}"')
            print(f'     - Greeting: {match[0].group.value} (semantic match)')
            print(f'     - Action: {match[1].group.value} (dynamic semantic)')
            print(f'     - Item: {match[2].group.value} (multi-word phrase)')
            print(f'     - Message: {match[3].group.value} (quoted string)')
        else:
            print(f'   ✗ No match for: "{test_comm}"')
    except Exception as e:
        print(f'   Error: {e}')
    
    # Example 3: Mathematical expression with semantic context
    print("\n\n3. Math Expression with Context:")
    print('   Pattern: I {feeling:semantic} about solving {equation:math} for {variable}')
    
    expr3 = UnifiedBudExpression(
        'I {feeling:semantic} about solving {equation:math} for {variable}',
        registry
    )
    
    test_math = 'I anxious about solving x^2 + 5x + 6 = 0 for x'
    
    try:
        match = expr3.match(test_math)
        if match:
            print(f'\n   ✓ Matched: "{test_math}"')
            print(f'     - Feeling: {match[0].group.value} (semantic match for "worried")')
            print(f'     - Equation: {match[1].group.value} (math expression)')
            print(f'     - Variable: {match[2].group.value}')
        else:
            print(f'   ✗ No match for: "{test_math}"')
    except Exception as e:
        print(f'   Error: {e}')
    
    # Example 4: Ultimate combination - ALL types
    print("\n\n4. Ultimate Combination - Using ALL Parameter Types:")
    print('   Pattern: The {customer_type:dynamic} customer {name} is {mood:semantic} to buy')
    print('            {count} {product:phrase} for ${amount} with code {promo:regex}')
    print('            and wants delivery to {address:quoted} by {date}')
    
    # Create regex for promo codes (e.g., SAVE20, DEAL50)
    registry.create_regex_parameter_type(
        'promo',
        r'[A-Z]{3,6}\d{2}'
    )
    
    expr4 = UnifiedBudExpression(
        'The {customer_type:dynamic} customer {name} is {mood:semantic} to buy '
        '{count} {product:phrase} for ${amount} with code {promo:regex} '
        'and wants delivery to {address:quoted} by {date}',
        registry
    )
    
    test_ultimate = ('The premium customer John is eager to buy '
                    '3 Samsung Galaxy S24 Ultra for $3600 with code SAVE20 '
                    'and wants delivery to "123 Main St, Suite 456" by tomorrow')
    
    try:
        match = expr4.match(test_ultimate)
        if match:
            print(f'\n   ✓ Matched: "{test_ultimate}"')
            print(f'     - Customer Type: {match[0].group.value} (dynamic semantic)')
            print(f'     - Name: {match[1].group.value} (standard)')
            print(f'     - Mood: {match[2].group.value} (semantic match for "eager")')
            print(f'     - Count: {match[3].group.value} (standard)')
            print(f'     - Product: {match[4].group.value} (multi-word phrase)')
            print(f'     - Amount: {match[5].group.value} (standard)')
            print(f'     - Promo Code: {match[6].group.value} (regex validated)')
            print(f'     - Address: {match[7].group.value} (quoted string)')
            print(f'     - Date: {match[8].group.value} (standard)')
        else:
            print(f'   ✗ No match for: "{test_ultimate}"')
    except Exception as e:
        print(f'   Error: {e}')
    
    # Example 5: Real-world use case
    print("\n\n5. Real-World Use Case - Natural Language Commands:")
    print('   Pattern: {assistant:dynamic}, {action:semantic} me a {vehicle:phrase} from')
    print('            {location:quoted} for {duration:phrase} at {price:regex} per day')
    
    # Create regex for price formats
    registry.create_regex_parameter_type(
        'price',
        r'\$?\d+(?:\.\d{2})?'
    )
    
    registry.create_phrase_parameter_type('duration', max_phrase_length=3)
    
    expr5 = UnifiedBudExpression(
        '{assistant:dynamic}, {action:semantic} me a {vehicle:phrase} from '
        '{location:quoted} for {duration:phrase} at {price:regex} per day',
        registry
    )
    
    test_real = 'Alexa, book me a luxury SUV from "Los Angeles Airport" for 5 business days at $89.99 per day'
    
    try:
        match = expr5.match(test_real)
        if match:
            print(f'\n   ✓ Matched: "{test_real}"')
            print(f'     - Assistant: {match[0].group.value} (dynamic - matched with "assistant")')
            print(f'     - Action: {match[1].group.value} (semantic - matched with "book")')
            print(f'     - Vehicle: {match[2].group.value} (phrase)')
            print(f'     - Location: {match[3].group.value} (quoted)')
            print(f'     - Duration: {match[4].group.value} (phrase)')
            print(f'     - Price: {match[5].group.value} (regex)')
        else:
            print(f'   ✗ No match for: "{test_real}"')
    except Exception as e:
        print(f'   Error: {e}')
    
    print("\n\n=== Summary ===")
    print("This demonstration shows how the unified system can:")
    print("• Combine ALL parameter types in a single expression")
    print("• Handle complex real-world patterns")
    print("• Mix semantic understanding with structural patterns")
    print("• Process multi-word phrases alongside other types")
    print("• Validate with regex while understanding meaning")

if __name__ == "__main__":
    demo_all_types()
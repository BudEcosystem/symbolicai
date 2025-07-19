#!/usr/bin/env python3
"""
Example showing how to handle flexible patterns in semantic cucumber expressions
"""

from semantic_cucumber_expressions import (
    SemanticCucumberExpression,
    SemanticParameterTypeRegistry,
    SemanticParameterType,
    ParameterType
)

def demo_flexible_patterns():
    """Demonstrate flexible pattern matching"""
    
    # Initialize registry
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    # Option 1: Create a custom parameter type that matches multiple words
    registry.define_parameter_type(ParameterType(
        name="phrase",
        regexp=r".+?",  # Matches any characters (non-greedy)
        type=str,
        transformer=lambda s: s
    ))
    
    # Option 2: Create a semantic parameter type that matches multiple words
    registry.define_parameter_type(SemanticParameterType(
        name="car_phrase",
        prototypes=["Ferrari", "Tesla", "Mercedes", "BMW", "Rolls Royce"],
        similarity_threshold=0.3,
        fallback_regexp=r"[\w\s]+"  # Matches words and spaces
    ))
    
    # Option 3: Use alternation to handle specific multi-word cases
    registry.define_parameter_type(ParameterType(
        name="luxury_car",
        regexp=r"(?:Rolls Royce|Mercedes Benz|Ferrari|Lamborghini|[A-Za-z]+)",
        type=str,
        transformer=lambda s: s
    ))
    
    print("=== Flexible Pattern Examples ===\n")
    
    # Example 1: Using the phrase parameter
    print("1. Using flexible 'phrase' parameter:")
    expr1 = SemanticCucumberExpression("I {emotion} {phrase}", registry)
    match1 = expr1.match("I happy driving my red Ferrari")
    if match1:
        print(f"   Emotion: {match1[0].value}")
        print(f"   Phrase: {match1[1].value}")
    
    # Example 2: Using semantic car_phrase
    print("\n2. Using semantic 'car_phrase' parameter:")
    expr2 = SemanticCucumberExpression("I drive a {car_phrase}", registry)
    
    test_phrases = [
        "I drive a Rolls Royce",
        "I drive a Tesla Model S",
        "I drive a red Ferrari"
    ]
    
    for phrase in test_phrases:
        match = expr2.match(phrase)
        if match:
            print(f"   ✓ '{phrase}' -> Car: {match[0].value}")
        else:
            print(f"   ✗ '{phrase}' -> No match")
    
    # Example 3: Using luxury_car with specific patterns
    print("\n3. Using 'luxury_car' with specific patterns:")
    expr3 = SemanticCucumberExpression("I want a {luxury_car}", registry)
    
    test_cars = [
        "I want a Rolls Royce",
        "I want a Mercedes Benz",
        "I want a Ferrari",
        "I want a Toyota"  # Single word will also work
    ]
    
    for phrase in test_cars:
        match = expr3.match(phrase)
        if match:
            print(f"   ✓ '{phrase}' -> Car: {match[0].value}")

    # Example 4: Working around the limitation
    print("\n4. Alternative approach - structured patterns:")
    
    # Instead of "I {emotion} .+ {car}", use more structured patterns
    print("   Instead of: 'I {emotion} .+ {car}'")
    print("   Use: 'I am {emotion} about/with/for {object}'")
    
    registry.define_parameter_type(ParameterType(
        name="preposition",
        regexp=r"(?:about|with|for|at|on|in)",
        type=str,
        transformer=lambda s: s
    ))
    
    expr4 = SemanticCucumberExpression("I am {emotion} {preposition} the {cars}", registry)
    
    test_sentences = [
        "I am happy about the Ferrari",
        "I am excited with the Tesla",
        "I am frustrated at the BMW"
    ]
    
    for sentence in test_sentences:
        match = expr4.match(sentence)
        if match:
            print(f"   ✓ '{sentence}'")
            print(f"     Emotion: {match[0].value}, Prep: {match[1].value}, Car: {match[2].value}")

if __name__ == "__main__":
    demo_flexible_patterns()
# Example usage file: example_usage.py
"""
Example usage of semantic cucumber expressions
"""

from semantic_cucumber_expressions import (
    SemanticCucumberExpression,
    SemanticParameterTypeRegistry,
    SemanticParameterType
)

def example_basic_usage():
    """Basic usage example"""
    # Create registry and initialize model
    registry = SemanticParameterTypeRegistry()
    registry.initialize_model()
    
    # Create expressions
    expr1 = SemanticCucumberExpression("I like {fruit}", registry)
    expr2 = SemanticCucumberExpression("{greeting}, how are you?", registry)
    expr3 = SemanticCucumberExpression("solve {math}", registry)
    exp4 = SemanticCucumberExpression("I am {emotion} about the {cars}", registry)

    match0 = exp4.match("I am happy .+ Mercedes")
    if match0:
        print(f"Emotion: {match0[0].value}, Car: {match0[1].value}")
    else:
        print("No match for exp4")
    
    # Test matches
    match1 = expr1.match("I love watermelon")
    if match1:
        print(f"Matched fruit: {match1[0].value}")
        
    match2 = expr2.match("Howdy, how are you?")
    if match2:
        print(f"Matched greeting: {match2[0].value}")
        
    match3 = expr3.match("solve 0 =x + 2 ")
    if match3:
        print(f"Matched math: {match3[0].value}")

if __name__ == "__main__":
    example_basic_usage()
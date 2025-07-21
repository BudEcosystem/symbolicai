#!/usr/bin/env python3
"""
Debug context matching functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_bud_expressions import (
    UnifiedParameterTypeRegistry,
    ContextAwareExpression
)

# Initialize registry
registry = UnifiedParameterTypeRegistry()
registry.initialize_model()

# Create expression
expr = ContextAwareExpression(
    expression="I {emotion} {vehicle}",
    expected_context="cars and automotive technology",
    context_threshold=0.6,
    registry=registry
)

# Test text with matching context
text1 = "The automotive industry has revolutionized transportation. Cars are amazing. I love Tesla"
print(f"Testing text: {text1}")
print(f"Expression pattern: {expr.expression.regexp}")

# Check if regex matches
import re
matches = list(re.finditer(expr.expression.regexp, text1))
print(f"Regex matches found: {len(matches)}")
for i, match in enumerate(matches):
    print(f"  Match {i}: '{match.group()}' at position {match.start()}-{match.end()}")
    
# Try matching with context
result = expr.match_with_context(text1)
print(f"Context match result: {result}")

if result is None:
    # Debug individual steps
    for match in matches:
        match_text = match.group()
        match_start = match.start()
        
        # Try to parse the match
        parsed = expr.expression.match(match_text)
        print(f"\nDebug match '{match_text}':")
        print(f"  Parsed result: {parsed}")
        
        if parsed:
            # Extract context
            context = expr.extract_context(text1, match_start)
            print(f"  Extracted context: '{context}'")
            
            # Check similarity
            similarity = expr._compare_context(context)
            print(f"  Context similarity: {similarity:.3f} (threshold: {expr.context_threshold})")
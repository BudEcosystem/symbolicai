# Enhanced Multi-Word Phrase Matching with FAISS

This document describes the enhanced multi-word phrase matching capabilities added to the SymbolicAI library using FAISS (Facebook AI Similarity Search) for improved performance and accuracy.

## Overview

The enhanced phrase matching system solves several key challenges:

1. **Multi-word phrase boundary detection** - Intelligently identifies where phrases begin and end
2. **Semantic phrase matching** - Matches phrases based on meaning, not just exact text
3. **Large vocabulary support** - Efficiently handles thousands of known phrases using FAISS
4. **Context-aware matching** - Uses surrounding context to improve phrase extraction
5. **Adaptive phrase length** - Handles phrases of varying lengths intelligently

## Key Components

### 1. FAISSPhraseMatcher

The core component that provides intelligent phrase matching capabilities:

```python
from semantic_bud_expressions import FAISSPhraseMatcher

# Initialize phrase matcher
matcher = FAISSPhraseMatcher(
    model_manager=model_manager,
    max_phrase_length=10,
    similarity_threshold=0.4,
    use_context_scoring=True
)

# Add known phrases
matcher.add_known_phrases([
    "Tesla Model 3", "BMW X5", "Mercedes S Class"
])

# Match phrases in text
result = matcher.match_phrase_in_text(
    "I drive a Tesla Model 3",
    start_pos=10
)
# Returns: ("Tesla Model 3", 10, 23)
```

### 2. EnhancedUnifiedParameterType

An enhanced parameter type that integrates FAISS for better performance:

```python
from semantic_bud_expressions import EnhancedUnifiedParameterType

# Create enhanced phrase parameter
param = EnhancedUnifiedParameterType(
    name="car_model",
    type_hint=ParameterTypeHint.PHRASE,
    prototypes=["Tesla Model 3", "BMW X5"],
    use_faiss=True,
    phrase_similarity_threshold=0.4
)
```

### 3. EnhancedUnifiedParameterTypeRegistry

A registry that automatically enables FAISS for large vocabularies:

```python
from semantic_bud_expressions import EnhancedUnifiedParameterTypeRegistry

# Create enhanced registry
registry = EnhancedUnifiedParameterTypeRegistry(
    use_faiss=True,
    faiss_auto_threshold=100  # Auto-enable FAISS for 100+ prototypes
)

# Create phrase parameter type with FAISS
registry.create_phrase_parameter_type(
    "product",
    max_phrase_length=6,
    known_phrases=large_product_list  # Automatically uses FAISS if large
)
```

## Usage Examples

### Basic Multi-Word Phrase Matching

```python
from semantic_bud_expressions import UnifiedBudExpression, EnhancedUnifiedParameterTypeRegistry

# Initialize registry
registry = EnhancedUnifiedParameterTypeRegistry()
registry.initialize_model()

# Create phrase type for car models
registry.create_phrase_parameter_type(
    "car_model",
    max_phrase_length=5,
    known_phrases=[
        "Tesla Model 3", "Tesla Model S", "BMW 3 Series",
        "Mercedes S Class", "Rolls Royce Phantom"
    ]
)

# Create expression
expr = UnifiedBudExpression("I drive a {car_model:phrase}", registry)

# Match multi-word phrases
match = expr.match("I drive a Rolls Royce Phantom")
print(match[0].value)  # "Rolls Royce Phantom"
```

### Semantic Phrase Matching

Match phrases based on semantic similarity:

```python
# Create semantic phrase type
registry.create_semantic_phrase_parameter_type(
    "product",
    semantic_categories=["smartphone", "laptop", "tablet"],
    max_phrase_length=6,
    similarity_threshold=0.4
)

expr = UnifiedBudExpression("I want to buy {product:phrase}", registry)

# Matches "iPhone 15 Pro Max" because iPhone ~ smartphone
match = expr.match("I want to buy iPhone 15 Pro Max")
```

### Context-Aware Phrase Matching

Use context to improve phrase boundary detection:

```python
# Create location phrase type with context scoring
registry.create_phrase_parameter_type(
    "location",
    max_phrase_length=6,
    use_context_scoring=True,
    known_phrases=[
        "New York City", "San Francisco Bay Area",
        "Central Park", "Golden Gate Bridge"
    ]
)

# Context helps identify "Central Park" even with extra text
match = expr.match("I visited Central Park in Manhattan last summer")
# Extracts: "Central Park" (not "Central Park in Manhattan")
```

### Large Vocabulary Support

FAISS automatically enables for large phrase sets:

```python
# Create large vocabulary (1000+ phrases)
large_vocabulary = generate_product_names()  # Returns 1000+ product names

# FAISS automatically enabled due to size
registry.create_phrase_parameter_type(
    "product_name",
    known_phrases=large_vocabulary  # FAISS handles this efficiently
)

# Fast matching even with thousands of phrases
expr = UnifiedBudExpression("Buy {product_name:phrase}", registry)
match = expr.match("Buy Apple iPhone 15 Pro Max")  # Still fast!
```

## Performance Benefits

### Without FAISS (Naive Approach)
- Linear search through all phrases: O(n)
- Slow for large vocabularies (1000+ phrases)
- Memory inefficient for similarity computations

### With FAISS
- Approximate nearest neighbor search: O(log n) or better
- 5-10x speedup for large vocabularies
- Memory-efficient index structures
- GPU acceleration available

## Configuration Options

### Registry Configuration

```python
registry = EnhancedUnifiedParameterTypeRegistry(
    use_faiss=True,                    # Enable FAISS globally
    faiss_auto_threshold=100,          # Auto-enable for 100+ prototypes
    phrase_similarity_threshold=0.4     # Default similarity threshold
)
```

### Parameter Type Configuration

```python
registry.create_phrase_parameter_type(
    "car_model",
    # Phrase-specific options
    max_phrase_length=5,               # Maximum words in phrase
    phrase_delimiters=['.', ',', '!'], # Custom delimiters
    
    # FAISS-specific options
    use_faiss=True,                    # Force FAISS usage
    phrase_similarity_threshold=0.5,    # Similarity threshold
    use_context_scoring=True,          # Enable context scoring
    
    # Known phrases for FAISS index
    known_phrases=car_models_list
)
```

## Advanced Features

### 1. Phrase Boundary Detection

The system intelligently detects phrase boundaries using:
- Delimiter detection (punctuation, brackets, etc.)
- Word count limits
- Semantic coherence scoring
- Context analysis

### 2. Adaptive Phrase Length

Handles phrases of different lengths intelligently:

```python
addresses = [
    "Main Street",                      # 2 words
    "123 Main Street",                  # 3 words
    "123 Main Street Suite 100",        # 5 words
    "789 Broadway Floor 5 Office 501"   # 6 words
]

registry.create_phrase_parameter_type(
    "address",
    max_phrase_length=8,
    known_phrases=addresses
)
```

### 3. FAISS Index Types

The system automatically selects appropriate FAISS index types:
- **Flat**: For small datasets (<100 phrases) - exact search
- **IVF**: For medium datasets (100-10k phrases) - clustered search
- **HNSW**: For large datasets (10k+ phrases) - graph-based search

### 4. Statistics and Monitoring

Get insights into FAISS usage:

```python
stats = registry.get_faiss_statistics()
print(stats)
# {
#     'faiss_enabled': True,
#     'faiss_auto_threshold': 100,
#     'faiss_enabled_types': ['car_model', 'product'],
#     'type_car_model': {
#         'num_known_phrases': 150,
#         'faiss_index': {'type': 'IVF', 'num_vectors': 150}
#     }
# }
```

## Migration Guide

### From Basic UnifiedParameterType

```python
# Old approach
registry = UnifiedParameterTypeRegistry()
registry.create_phrase_parameter_type("car_model", max_phrase_length=5)

# New enhanced approach
registry = EnhancedUnifiedParameterTypeRegistry()
registry.create_phrase_parameter_type(
    "car_model",
    max_phrase_length=5,
    known_phrases=car_models,  # Add known phrases for FAISS
    use_faiss=True
)
```

### Adding FAISS to Existing Types

```python
# Add phrases to existing type
registry.add_phrases_to_type(
    "car_model",
    ["Tesla Cybertruck", "Ford F-150 Lightning"],
    categories=["electric_truck", "electric_truck"]
)
```

## Best Practices

1. **Provide Known Phrases**: Always provide known phrases when possible for better accuracy
2. **Set Appropriate Thresholds**: Adjust similarity thresholds based on your use case
3. **Use Context Scoring**: Enable for better phrase boundary detection
4. **Monitor Performance**: Use statistics to optimize configurations
5. **Batch Operations**: Add multiple phrases at once for better performance

## Troubleshooting

### FAISS Not Available

If FAISS is not installed, the system automatically falls back to numpy-based matching:

```python
# System will use numpy fallback if FAISS not available
# Install FAISS: pip install faiss-cpu or faiss-gpu
```

### Memory Issues with Large Vocabularies

For very large vocabularies (100k+ phrases):
1. Use GPU acceleration if available
2. Consider using approximate indices (IVF, HNSW)
3. Adjust index parameters for memory/accuracy tradeoff

### Phrase Not Matching

If phrases aren't matching as expected:
1. Check similarity threshold (lower = more permissive)
2. Verify phrase delimiters are configured correctly
3. Ensure known phrases cover your use cases
4. Enable debug logging to see similarity scores

## Future Enhancements

Planned improvements include:
- Multilingual phrase matching
- Custom similarity metrics
- Phrase clustering and categorization
- Dynamic phrase learning from usage
- Integration with spaCy/NLTK for better NLP
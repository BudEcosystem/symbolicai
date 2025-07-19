# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library that extends `cucumber-expressions` with semantic matching capabilities using AI embeddings. It allows matching text based on meaning similarity rather than exact patterns, making it useful for more flexible BDD testing scenarios.

## Dependencies

The project requires:
- Python 3.12
- cucumber-expressions
- numpy
- sympy
- model2vec (uses 'minishlab/potion-base-8M' model)

Install dependencies:
```bash
pip install cucumber-expressions numpy sympy model2vec
```

## Running the Example

To test the library functionality:
```bash
python example.py
```

## Architecture

### Core Components

1. **SemanticCucumberExpression** (`semantic_expression.py`): Main class that extends CucumberExpression with semantic matching
2. **SemanticParameterTypeRegistry** (`semantic_registry.py`): Registry for semantic parameter types, manages the embedding model
3. **ModelManager** (`model_manager.py`): Handles the Model2Vec embeddings model initialization and caching
4. **SemanticParameterType** (`semantic_parameter_type.py`): Defines semantic categories with prototypes and similarity thresholds
5. **MathParameterType** (`math_parameter_type.py`): Special handling for mathematical expressions using SymPy

### Key Features

- **Semantic Categories**: Built-in categories include `fruit`, `greeting`, `math`, `float`, `int`, `word`
- **Custom Categories**: Define new categories with prototypes and similarity thresholds
- **Transformers**: Custom functions to process matched values
- **Caching**: Embeddings are cached for performance
- **Math Support**: Symbolic math expression parsing and normalization

### Usage Pattern

```python
from semantic_cucumber_expressions import SemanticCucumberExpression, SemanticParameterTypeRegistry

# Initialize registry and model
registry = SemanticParameterTypeRegistry()
registry.initialize_model()

# Create expression and match
expr = SemanticCucumberExpression("I love {fruit}", registry)
match = expr.match("I love grapes")
```

## Development Notes

- The library inherits from the original cucumber-expressions library classes
- Model initialization happens once per registry instance
- Semantic matching uses cosine similarity with configurable thresholds
- Math expressions are normalized using SymPy for consistent matching

## Recent Fixes Applied

1. **Model2Vec Import**: Changed from `Model2Vec` to `StaticModel` and from `Model2Vec.load()` to `StaticModel.from_pretrained()`
2. **Async/Sync Issues**: Removed unnecessary async handling as model2vec operations are synchronous
3. **Cache Deadlock**: Fixed threading lock issue in `SemanticCache.put_batch()` method
4. **Method Names**: Changed from `model.embed()` to `model.encode()` for model2vec compatibility

## Dynamic Semantic Matching

The library now supports dynamic semantic matching, where undefined parameter types are automatically matched based on semantic similarity between the parameter name and the value:

```python
# If {cars} is not predefined, it dynamically matches based on similarity
expr = SemanticCucumberExpression("I love {cars}", registry)
match = expr.match("I love Ferrari")  # Works! Compares "cars" ↔ "Ferrari"
```

### Configuration
- **Enable/Disable**: `registry.enable_dynamic_matching(True/False)`
- **Set Threshold**: `registry.set_dynamic_threshold(0.3)`  # Default is 0.3
- **Clear Cache**: `registry._dynamic_types_cache.clear()`  # When changing thresholds

### Testing
Run tests: `python test_dynamic_semantic.py`
Run example: `python example_dynamic_matching.py`

## Unified Expression System

The library now includes a unified expression system that supports multiple parameter types and matching strategies through type hints:

### Syntax
```python
{param}              # Standard parameter
{param:semantic}     # Semantic parameter with similarity matching
{param:phrase}       # Multi-word phrase parameter
{param:dynamic}      # Dynamic semantic parameter
{param:regex}        # Custom regex parameter
{param:math}         # Mathematical expression
{param:quoted}       # Quoted string parameter
```

### Usage
```python
from semantic_cucumber_expressions import UnifiedCucumberExpression, UnifiedParameterTypeRegistry

# Initialize registry
registry = UnifiedParameterTypeRegistry()
registry.initialize_model()

# Create phrase parameter type for multi-word matching
registry.create_phrase_parameter_type('car_model', max_phrase_length=4)

# Use with type hints
expr = UnifiedCucumberExpression("I drive a {car_model:phrase}", registry)
match = expr.match("I drive a Rolls Royce")  # ✓ Matches multi-word phrases!
```

### Multi-Word Phrase Matching
The unified system solves the multi-word phrase matching problem:

```python
# Traditional: Limited to single words
expr1 = SemanticCucumberExpression("I drive {car}", registry)
match1 = expr1.match("I drive Ferrari")  # ✓ Works
match2 = expr1.match("I drive Rolls Royce")  # ✗ Fails

# Unified: Supports multi-word phrases
registry.create_phrase_parameter_type('vehicle', max_phrase_length=5)
expr2 = UnifiedCucumberExpression("I drive {vehicle:phrase}", registry)
match3 = expr2.match("I drive Rolls Royce")  # ✓ Works!
match4 = expr2.match("I drive Mercedes Benz S Class")  # ✓ Works!
```

### Mixed Parameter Types
```python
# Mix different parameter types in one expression
expr = UnifiedCucumberExpression(
    "I {action:semantic} {count} {items:phrase}",
    registry
)
match = expr.match("I purchased 5 red sports cars")
# action: "purchased" (semantic match)
# count: "5" (standard)
# items: "red sports cars" (phrase)
```

### Configuration
- **Phrase boundaries**: `registry.set_phrase_config(max_phrase_length=10, phrase_delimiters=['.', '!', '?'])`
- **Semantic thresholds**: Different per parameter type
- **Backward compatibility**: Existing API continues to work

### Testing
Run tests: `python test_unified_system.py`
Run example: `python example_unified_demo.py`
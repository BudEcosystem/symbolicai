# Bud Symbolic AI

An effort to build a holistic matching engine that supports regex, cucumber expressions, semantic phrases, and combinations of that. Could be useful for Guardrails, Caching systems etc.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Semantic Matching**: Match text based on meaning, not just exact patterns
- **Multi-word Phrase Support**: Capture phrases like "Rolls Royce" or "MacBook Pro 16 inch"
- **Unified Expression System**: Mix different parameter types in a single expression
- **Dynamic Parameter Types**: Automatically create parameter types based on context
- **Embedding support**: embeddings (Model2Vec) for semantic understanding, Model2Vec could be replaced with any other Embedding models, we chose this for performance reasons.
- **Backward Compatible**: Works with existing Cucumber Expression syntax
- **Extensible**: Easy to add custom parameter types and matching strategies

## 🚀 Quick Start

```python
from semantic_cucumber_expressions import UnifiedCucumberExpression, UnifiedParameterTypeRegistry

# Initialize
registry = UnifiedParameterTypeRegistry()
registry.initialize_model()

# Create expression with multiple parameter types
expr = UnifiedCucumberExpression(
    "{greeting} {name}, I want to {action:semantic} a {product:phrase}",
    registry
)

# Match natural language
match = expr.match("Hello John, I want to purchase a MacBook Pro")
# Results: greeting="Hello", name="John", action="purchase", product="MacBook Pro"
```

## 📦 Installation

### Requirements
- Python 3.12+
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-cucumber-expressions.git
cd semantic-cucumber-expressions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
cucumber-expressions>=17.0.0
numpy>=1.24.0
sympy>=1.12
model2vec>=0.2.0
```

## 🎯 Supported Expression Types

### 1. Standard Parameters
```python
"{name}"  # Matches any single word
"{count}" # Matches numbers
"{word}"  # Matches words
```

### 2. Semantic Parameters
```python
"{fruit:semantic}"     # Matches based on semantic similarity
"{emotion}"           # Uses predefined semantic categories
"{vehicle:semantic}"  # Custom semantic matching
```

### 3. Dynamic Parameters
```python
"{cars:dynamic}"      # Dynamically matches based on parameter name
"{furniture:dynamic}" # No predefined list needed
```

### 4. Phrase Parameters
```python
"{product:phrase}"    # Matches multi-word phrases
"{car_model:phrase}"  # "Rolls Royce", "Mercedes Benz S Class"
```

### 5. Regex Parameters
```python
"{email:regex}"       # Custom regex patterns
"{phone:regex}"       # Validate specific formats
```

### 6. Special Parameters
```python
"{math}"              # Mathematical expressions
"{quoted:quoted}"     # Quoted strings
```

## 📖 Detailed Usage

### Basic Semantic Matching

```python
from semantic_cucumber_expressions import SemanticCucumberExpression, SemanticParameterTypeRegistry

# Initialize registry
registry = SemanticParameterTypeRegistry()
registry.initialize_model()

# Create expression
expr = SemanticCucumberExpression("I love {fruit}", registry)

# Matches semantically similar words
match = expr.match("I love apples")    # ✓ Matches
match = expr.match("I love oranges")   # ✓ Matches
match = expr.match("I love mango")     # ✓ Matches
```

### Multi-word Phrase Matching

```python
# Create phrase parameter type
registry.create_phrase_parameter_type('car_model', max_phrase_length=5)

# Use in expression
expr = UnifiedCucumberExpression("I drive a {car_model:phrase}", registry)

# Matches multi-word phrases
match = expr.match("I drive a Rolls Royce")           # ✓ Matches: "Rolls Royce"
match = expr.match("I drive a Mercedes Benz S Class") # ✓ Matches: "Mercedes Benz S Class"
```

### Custom Semantic Categories

```python
# Define custom semantic category
registry.define_semantic_category(
    name="vehicle",
    prototypes=["car", "truck", "bus", "motorcycle", "bicycle"],
    similarity_threshold=0.6
)

expr = SemanticCucumberExpression("I drive a {vehicle}", registry)
match = expr.match("I drive a Tesla")  # ✓ Matches
```

### Dynamic Semantic Matching

```python
# Enable dynamic matching
registry.enable_dynamic_matching(True)
registry.set_dynamic_threshold(0.3)

# Undefined parameters are matched dynamically
expr = SemanticCucumberExpression("I love {cars}", registry)
match = expr.match("I love Ferrari")  # ✓ Matches (compares "cars" ↔ "Ferrari")
```

### Mixed Parameter Types

```python
# Combine all parameter types in one expression
expr = UnifiedCucumberExpression(
    "{person:dynamic} wants to {action:semantic} {count} {items:phrase} for {price:regex}",
    registry
)

match = expr.match("customer wants to purchase 5 red sports cars for $50,000")
# Extracts all parameters with their specific matching strategies
```

## 🏗️ Architecture

### Core Components

```
semantic_cucumber_expressions/
├── Core Expression System
│   ├── expression.py              # Base CucumberExpression
│   ├── semantic_expression.py     # Semantic-enhanced expressions
│   └── unified_expression.py      # Unified expression system
│
├── Parameter Types
│   ├── parameter_type.py          # Base parameter type
│   ├── semantic_parameter_type.py # Semantic matching
│   ├── dynamic_semantic_parameter_type.py # Dynamic types
│   └── unified_parameter_type.py  # Unified type system
│
├── Registries
│   ├── parameter_type_registry.py # Base registry
│   ├── semantic_registry.py       # Semantic-aware registry
│   └── unified_registry.py        # Unified registry
│
├── AI/ML Components
│   ├── model_manager.py           # Model2Vec integration
│   └── semantic_cache.py          # Embedding cache
│
└── Utilities
    ├── math_parameter_type.py     # Math expression support
    └── unified_expression_parser.py # Type hint parser
```

### Processing Pipeline

1. **Expression Parsing**
   - Parse expression string
   - Extract parameter type hints
   - Create AST representation

2. **Parameter Type Resolution**
   - Look up existing parameter types
   - Create dynamic types if needed
   - Apply type-specific regex patterns

3. **Matching Process**
   - Apply regex matching
   - Perform semantic validation
   - Handle phrase boundaries
   - Transform matched values

4. **Result Processing**
   - Extract matched groups
   - Apply transformations
   - Return Argument objects

## 🔧 Configuration

### Registry Configuration

```python
# Set dynamic matching threshold
registry.set_dynamic_threshold(0.3)

# Configure phrase matching
registry.set_phrase_config(
    max_phrase_length=10,
    phrase_delimiters=['.', '!', '?', ',', ';', ':']
)

# Enable/disable features
registry.enable_dynamic_matching(True)
registry.enable_advanced_phrase_matching(True)
```

### Model Configuration

```python
# Use different embedding model
registry.initialize_model(model_name='minishlab/potion-base-8M')

# Model is cached and shared across instances
```

## 🔌 Extending the Library

### Custom Parameter Types

```python
from semantic_cucumber_expressions import UnifiedParameterType, ParameterTypeHint

class MyCustomParameterType(UnifiedParameterType):
    def __init__(self, name: str, **kwargs):
        super().__init__(
            name=name,
            type_hint=ParameterTypeHint.CUSTOM,
            **kwargs
        )
    
    def transform(self, group_values: List[str]) -> Any:
        # Custom transformation logic
        return custom_transform(group_values[0])

# Register custom type
registry.define_parameter_type(MyCustomParameterType('custom'))
```

### Custom Transformers

```python
def color_transformer(value: str, **kwargs) -> dict:
    """Transform color names to hex codes"""
    similarity = kwargs.get('similarity', 0)
    return {
        'name': value,
        'hex': get_hex_color(value),
        'similarity': similarity
    }

registry.define_parameter_type(SemanticParameterType(
    name="color",
    prototypes=["red", "blue", "green"],
    transformer=color_transformer
))
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python test_unified_system.py
python test_dynamic_semantic.py

# Run examples
python example.py
python example_unified_demo.py
python test_all_types.py
```

## 🗺️ Roadmap

### Version 1.1 (Next Release)
- [ ] Improved phrase boundary detection using NLP
- [ ] Support for contextual parameters
- [ ] Performance optimizations for large-scale matching
- [ ] Better error messages and debugging tools

### Version 1.2
- [ ] Multiple language support
- [ ] Custom embedding models
- [ ] Async/await support throughout
- [ ] Integration with popular testing frameworks

### Version 2.0
- [ ] GraphQL-style nested parameter matching
- [ ] Machine learning-based parameter type inference
- [ ] Real-time pattern learning from examples
- [ ] Cloud-based model serving option

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.




## 📚 References

- [Cucumber Expressions Documentation](https://cucumber.io/docs/cucumber/cucumber-expressions/)
- [Model2Vec Paper](https://arxiv.org/abs/2310.00656)
- [Semantic Similarity in NLP](https://en.wikipedia.org/wiki/Semantic_similarity)


---

Made with ❤️ by the Bud Team
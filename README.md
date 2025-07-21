# Bud Symbolic AI

An effort to build a holistic matching engine that supports regex, budExpressions, semantic phrases, and combinations of that. Could be useful for Guardrails, Caching systems etc.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Semantic Matching**: Match text based on meaning, not just exact patterns
- **Multi-word Phrase Support**: Capture phrases like "Rolls Royce" or "MacBook Pro 16 inch"
- **Unified Expression System**: Mix different parameter types in a single expression
- **Dynamic Parameter Types**: Automatically create parameter types based on context
- **High Performance**: Sub-millisecond latency, 50,000+ ops/sec throughput
- **Embedding Support**: Model2Vec for semantic understanding (replaceable with other models)
- **Backward Compatible**: Works with existing bud Expression syntax
- **Extensible**: Easy to add custom parameter types and matching strategies

### 🚀 Performance Optimizations (New!)
- **Regex Compilation Cache**: 99%+ hit rate, eliminates recompilation overhead
- **Prototype Embedding Pre-computation**: Instant similarity checks
- **Batch Embedding Computation**: 60-80% reduction in model calls
- **Multi-level Caching**: L1 (expressions), L2 (embeddings), L3 (prototypes)
- **Optimized Semantic Types**: Reuse embeddings across matches
- **Thread-safe Design**: All caches use proper locking for concurrent access

## 🚀 Quick Start

```python
from semantic_bud_expressions import UnifiedBudExpression, UnifiedParameterTypeRegistry

# Initialize
registry = UnifiedParameterTypeRegistry()
registry.initialize_model()

# Create expression with multiple parameter types
expr = UnifiedBudExpression(
    "{greeting} {name}, I want to {action:semantic} a {product:phrase}",
    registry
)

# Match natural language
match = expr.match("Hello John, I want to purchase a MacBook Pro")
# Results: greeting="Hello", name="John", action="purchase", product="MacBook Pro"
```

## ⚡ Performance

Benchmarked on Apple M1 MacBook Pro:

| Expression Type | Avg Latency | Max Throughput | At 1000 RPS |
|----------------|-------------|----------------|-------------|
| Simple         | 0.020 ms    | 50,227 ops/sec | ✓ 100% success |
| Semantic       | 0.018 ms    | 55,735 ops/sec | ✓ 100% success |
| Complex        | 0.031 ms    | 32,513 ops/sec | ✓ 100% success |
| Mixed          | 0.027 ms    | 36,557 ops/sec | ✓ 100% success |

**With Optimizations Enabled:**
- **First match**: ~0.029 ms (cold cache)
- **Cached match**: ~0.002 ms (warm cache) - **12x speedup**
- **Throughput**: 17,000+ matches/second
- **Cache hit rate**: 50%+ after warm-up

**Real-world use cases:**
- API Guardrails: 580,000+ RPS capability
- Semantic Caching: 168,000+ RPS capability  
- Natural Language Commands: 55,000+ RPS capability
- Log Analysis: 397,000+ RPS capability

See [benchmarks/](benchmarks/) for detailed performance analysis.

## 📦 Installation

### Requirements
- Python 3.12+
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-bud-expressions.git
cd semantic-bud-expressions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
# Core dependencies
numpy>=1.24.0
sympy>=1.12
model2vec>=0.2.0

# Optional for advanced features
faiss-cpu>=1.7.4  # For large-scale similarity search (future)
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
from semantic_bud_expressions import SemanticBudExpression, SemanticParameterTypeRegistry

# Initialize registry
registry = SemanticParameterTypeRegistry()
registry.initialize_model()

# Create expression
expr = SemanticBudExpression("I love {fruit}", registry)

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
expr = UnifiedBudExpression("I drive a {car_model:phrase}", registry)

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

expr = SemanticBudExpression("I drive a {vehicle}", registry)
match = expr.match("I drive a Tesla")  # ✓ Matches
```

### Dynamic Semantic Matching

```python
# Enable dynamic matching
registry.enable_dynamic_matching(True)
registry.set_dynamic_threshold(0.3)

# Undefined parameters are matched dynamically
expr = SemanticBudExpression("I love {cars}", registry)
match = expr.match("I love Ferrari")  # ✓ Matches (compares "cars" ↔ "Ferrari")
```

### Mixed Parameter Types

```python
# Combine all parameter types in one expression
expr = UnifiedBudExpression(
    "{person:dynamic} wants to {action:semantic} {count} {items:phrase} for {price:regex}",
    registry
)

match = expr.match("customer wants to purchase 5 red sports cars for $50,000")
# Extracts all parameters with their specific matching strategies
```

## 📁 Project Structure

```
symbolicai/
├── semantic_bud_expressions/        # Core library
│   ├── expression.py                # Base expressions
│   ├── semantic_expression.py       # Semantic matching
│   ├── unified_expression.py        # Unified system
│   ├── parameter_types/             # All parameter types
│   ├── registries/                  # Type registries
│   ├── optimization/                # Performance optimizations
│   │   ├── regex_cache.py          # Regex compilation cache
│   │   ├── multi_level_cache.py    # Multi-level caching system
│   │   ├── batch_matcher.py        # Batch embedding computation
│   │   └── optimized_semantic_type.py  # Optimized parameter types
│   └── utils/                       # Utilities
├── examples/                        # Usage examples
│   ├── example.py                   # Basic example
│   ├── example_all_types.py         # All parameter types
│   ├── test_functional.py           # Functional tests
│   ├── benchmark_optimizations.py   # Performance benchmarks
│   └── README.md                    # Examples guide
├── benchmarks/                      # Performance testing
│   ├── scripts/                     # Benchmark tools
│   ├── results/                     # Results & visualizations
│   └── README.md                    # Benchmark guide
└── requirements.txt                 # Dependencies
```

## 🏗️ Architecture

### Core Components

- **Expression System**: Base, semantic, and unified expression classes
- **Parameter Types**: Standard, semantic, dynamic, phrase, regex, math, quoted
- **Registries**: Type management and resolution
- **AI/ML Components**: Model2Vec integration and embedding cache
- **Performance Optimizations**: 
  - Regex compilation cache
  - Multi-level caching (L1: expressions, L2: embeddings, L3: prototypes)
  - Batch embedding computation
  - Optimized semantic parameter types
- **Utilities**: Expression parsing and type hint processing

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

### Performance Configuration

```python
from semantic_bud_expressions import get_global_cache, clear_global_cache

# Configure cache sizes
cache = get_global_cache()
# Default: L1=1000, L2=10000, L3=5000

# Clear caches when needed
clear_global_cache()

# Use optimized semantic types for better performance
from semantic_bud_expressions import OptimizedSemanticParameterType

opt_type = OptimizedSemanticParameterType(
    name="product",
    prototypes=["laptop", "phone", "tablet"],
    similarity_threshold=0.6
)
registry.define_parameter_type(opt_type)
```

## 🔌 Extending the Library

### Custom Parameter Types

```python
from semantic_bud_expressions import UnifiedParameterType, ParameterTypeHint

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

## 🧪 Examples & Testing

### Running Examples

```bash
# Basic examples
cd examples
python example.py                    # Basic semantic matching
python example_all_types.py          # All parameter types demo
python example_unified_final.py      # Complete unified system

# Performance optimization examples
python benchmark_optimizations.py    # Test all optimizations
python test_cache_performance.py     # Cache performance demo
python optimized_example.py          # Optimized usage patterns

# Run tests
python test_functional.py            # Comprehensive functional tests
python test_unified_system.py        # Unified system tests
python test_dynamic_semantic.py      # Dynamic matching tests
```

### Performance Benchmarking

```bash
cd benchmarks
pip install -r requirements.txt

# Quick performance test
python scripts/benchmark_quick.py

# Full benchmark suite
python scripts/benchmark_tool.py

# Generate visualizations
python scripts/benchmark_visualizer.py
```

## 🎯 Performance Best Practices

### 1. Pre-initialize the Model
```python
# Initialize once at startup
registry = UnifiedParameterTypeRegistry()
registry.initialize_model()  # Pre-computes prototype embeddings
```

### 2. Reuse Expressions
```python
# Good - create once, use many times
expr = UnifiedBudExpression("Hello {name}", registry)
for text in texts:
    match = expr.match(text)  # Uses cached regex

# Avoid - creating expression in loop
for text in texts:
    expr = UnifiedBudExpression("Hello {name}", registry)  # Recompiles regex
    match = expr.match(text)
```

### 3. Use Optimized Types for High-Volume Matching
```python
# For high-throughput scenarios
from semantic_bud_expressions import OptimizedSemanticParameterType

opt_type = OptimizedSemanticParameterType(
    name="intent",
    prototypes=["buy", "sell", "trade"],
    similarity_threshold=0.7
)
```

### 4. Batch Process When Possible
```python
# Process multiple texts efficiently
from semantic_bud_expressions import BatchMatcher

batch_matcher = BatchMatcher(registry.model_manager)
phrases = batch_matcher.extract_all_phrases(text)
embeddings = batch_matcher.batch_compute_embeddings(phrases)
```

## 🗺️ Roadmap

### Version 1.1 (Current Release)
- [x] Performance optimizations for large-scale matching
  - [x] Regex compilation cache (99%+ hit rate)
  - [x] Prototype embedding pre-computation
  - [x] Batch embedding computation
  - [x] Multi-level caching system
- [x] Optimized semantic parameter types
- [x] Thread-safe caching design
- [ ] Improved phrase boundary detection using NLP
- [ ] Support for contextual parameters
- [ ] Better error messages and debugging tools

### Version 1.2 (Upcoming)
- [ ] FAISS integration for 100K+ prototype similarity search
- [ ] Sentence-level embedding strategy
- [ ] Async/await support throughout
- [ ] Multiple language support
- [ ] Custom embedding models
- [ ] Integration with popular testing frameworks

### Version 2.0 (Future)
- [ ] GraphQL-style nested parameter matching
- [ ] Machine learning-based parameter type inference
- [ ] Real-time pattern learning from examples
- [ ] Cloud-based model serving option
- [ ] Distributed caching support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.




## 📚 References

# Resources
- [Model2Vec Paper](https://arxiv.org/abs/2310.00656)
- [Semantic Similarity in NLP](https://en.wikipedia.org/wiki/Semantic_similarity)


---

Made with ❤️ by the Bud Team
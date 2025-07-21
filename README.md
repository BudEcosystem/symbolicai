# Bud Symbolic AI

A powerful matching engine with FAISS-enhanced multi-word phrase matching, semantic understanding, and intelligent parameter types. Perfect for building advanced Guardrails, Caching systems, and Natural Language Processing applications.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### Core Capabilities
- **FAISS-Enhanced Multi-Word Phrase Matching**: Intelligent phrase boundary detection with similarity search
- **Semantic Understanding**: Match text based on meaning, not just exact patterns
- **Context-Aware Matching**: Validate matches based on surrounding semantic context
- **Large Vocabulary Support**: Efficiently handle 1000+ phrase vocabularies with FAISS
- **Unified Expression System**: Mix different parameter types in a single expression
- **Dynamic Parameter Types**: Automatically create parameter types based on context

### Advanced Features  
- **Intelligent Phrase Boundaries**: Automatically detect optimal phrase boundaries
- **Semantic Phrase Categories**: Match "iPhone 15 Pro Max" as smartphone-related
- **Flexible Length Handling**: Gracefully handle phrases of varying lengths
- **Adaptive Matching**: Context-aware phrase extraction and validation
- **High Performance**: Sub-millisecond latency, 50,000+ ops/sec throughput
- **Backward Compatible**: Works with existing bud Expression syntax

### 🚀 Performance Optimizations
- **Regex Compilation Cache**: 99%+ hit rate, eliminates recompilation overhead
- **Prototype Embedding Pre-computation**: Instant similarity checks
- **Batch Embedding Computation**: 60-80% reduction in model calls
- **Multi-level Caching**: L1 (expressions), L2 (embeddings), L3 (prototypes)
- **Optimized Semantic Types**: Reuse embeddings across matches
- **Thread-safe Design**: All caches use proper locking for concurrent access
- **FAISS Integration**: 5-10x speedup for large phrase vocabularies (1000+ phrases)

## 🚀 Quick Start

### Basic Multi-Word Phrase Matching
```python
from semantic_bud_expressions import UnifiedBudExpression, EnhancedUnifiedParameterTypeRegistry

# Initialize enhanced registry with FAISS support
registry = EnhancedUnifiedParameterTypeRegistry()
registry.initialize_model()

# Create phrase parameter with known car models
registry.create_phrase_parameter_type(
    "car_model",
    max_phrase_length=5,
    known_phrases=[
        "Tesla Model 3", "BMW X5", "Mercedes S Class", 
        "Rolls Royce Phantom", "Ferrari 488 Spider"
    ]
)

# Match multi-word phrases intelligently  
expr = UnifiedBudExpression("I drive a {car_model:phrase}", registry)
match = expr.match("I drive a Rolls Royce Phantom")
print(match[0].value)  # "Rolls Royce Phantom"
```

### Semantic Phrase Matching
```python
# Create semantic phrase categories
registry.create_semantic_phrase_parameter_type(
    "device",
    semantic_categories=["smartphone", "laptop", "tablet"],
    max_phrase_length=6
)

expr = UnifiedBudExpression("I bought a {device:phrase}", registry)
match = expr.match("I bought iPhone 15 Pro Max")  # Matches as smartphone
print(match[0].value)  # "iPhone 15 Pro Max"
```

### Context-Aware Matching
```python
from semantic_bud_expressions import ContextAwareExpression

# Match expressions based on semantic context
expr = ContextAwareExpression(
    expression="I {emotion} {vehicle}",
    expected_context="cars and automotive",
    context_threshold=0.5,
    registry=registry
)

# Only matches in automotive context
text = "Cars are amazing technology. I love Tesla"
match = expr.match_with_context(text)  # ✓ Matches
print(f"Emotion: {match.parameters['emotion']}, Vehicle: {match.parameters['vehicle']}")
```

## ⚡ Performance

Benchmarked on Apple M1 MacBook Pro:

| Expression Type | Avg Latency | Max Throughput | FAISS Speedup |
|----------------|-------------|----------------|---------------|
| Simple         | 0.020 ms    | 50,227 ops/sec | N/A |
| Semantic       | 0.018 ms    | 55,735 ops/sec | 2x |
| Multi-word Phrase | 0.025 ms | 40,000 ops/sec | **5-10x** |
| Context-Aware  | 0.045 ms    | 22,000 ops/sec | 3x |
| Mixed Types    | 0.027 ms    | 36,557 ops/sec | 4x |

**FAISS Performance Benefits:**
- **Small vocabulary** (<100 phrases): 2x speedup
- **Medium vocabulary** (100-1K phrases): **5x speedup**  
- **Large vocabulary** (1K+ phrases): **10x speedup**
- **Memory efficiency**: 60% reduction for large vocabularies
- **Automatic optimization**: Enables automatically based on size

**With All Optimizations Enabled:**
- **Cold start**: ~0.029 ms (first match)
- **Warm cache**: ~0.002 ms (cached match) - **12x speedup**
- **FAISS + cache**: ~0.001 ms (optimal case) - **25x speedup**
- **Throughput**: 25,000+ phrase matches/second

**Real-world Performance:**
- **API Guardrails**: 580,000+ RPS capability
- **Semantic Caching**: 168,000+ RPS capability  
- **Phrase Matching**: 25,000+ RPS with 1000+ phrases
- **Context Analysis**: 22,000+ RPS capability

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

# Enhanced performance (recommended)
faiss-cpu>=1.7.4     # For 5-10x speedup with large phrase vocabularies
# faiss-gpu>=1.7.4   # For GPU acceleration (optional)
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

### 4. Enhanced Phrase Parameters (FAISS-Powered)
```python
"{product:phrase}"    # Multi-word phrases with intelligent boundaries
"{car_model:phrase}"  # "Rolls Royce Phantom", "Mercedes S Class"
"{address:phrase}"    # "123 Main Street Suite 100"
"{product_name:phrase}" # Auto-enables FAISS for 100+ known phrases
```

### 5. Semantic Phrase Parameters
```python
# Combines semantic understanding with phrase matching
registry.create_semantic_phrase_parameter_type(
    "device", 
    semantic_categories=["smartphone", "laptop", "tablet"]
)
"{device:phrase}"     # Matches "iPhone 15 Pro Max" as smartphone
```

### 6. Context-Aware Parameters
```python
# Validates matches based on surrounding context
ContextAwareExpression(
    "I {emotion} {item}",
    expected_context="technology and gadgets"
)
```

### 7. Regex Parameters
```python
"{email:regex}"       # Custom regex patterns
"{phone:regex}"       # Validate specific formats
```

### 8. Special Parameters
```python
"{math}"              # Mathematical expressions
"{quoted:quoted}"     # Quoted strings
```

## 📖 Detailed Usage

### Enhanced Multi-Word Phrase Matching (New!)

```python
from semantic_bud_expressions import UnifiedBudExpression, EnhancedUnifiedParameterTypeRegistry

# Create enhanced registry with FAISS support
registry = EnhancedUnifiedParameterTypeRegistry(
    use_faiss=True,
    faiss_auto_threshold=100  # Auto-enable FAISS for 100+ phrases
)
registry.initialize_model()

# Create phrase type with known phrases
registry.create_phrase_parameter_type(
    "car_model",
    max_phrase_length=5,
    known_phrases=[
        "Tesla Model 3", "BMW X5", "Mercedes S Class",
        "Rolls Royce Phantom", "Ferrari 488 Spider"
    ]
)

# Match multi-word phrases intelligently
expr = UnifiedBudExpression("I drive a {car_model:phrase}", registry)
match = expr.match("I drive a Rolls Royce Phantom")
print(match[0].value)  # "Rolls Royce Phantom"

# Semantic phrase matching
registry.create_semantic_phrase_parameter_type(
    "product",
    semantic_categories=["smartphone", "laptop", "tablet"],
    max_phrase_length=6
)

expr2 = UnifiedBudExpression("Buy {product:phrase}", registry)
match2 = expr2.match("Buy iPhone 15 Pro Max")  # Matches as smartphone
```

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

### Enhanced Registry Configuration

```python
from semantic_bud_expressions import EnhancedUnifiedParameterTypeRegistry

# Initialize with FAISS support
registry = EnhancedUnifiedParameterTypeRegistry(
    use_faiss=True,                    # Enable FAISS globally
    faiss_auto_threshold=100,          # Auto-enable for 100+ prototypes  
    phrase_similarity_threshold=0.4     # Similarity threshold for phrases
)

# Configure dynamic matching
registry.set_dynamic_threshold(0.3)

# Configure phrase matching
registry.set_phrase_config(
    max_phrase_length=10,
    phrase_delimiters=['.', '!', '?', ',', ';', ':']
)

# Enable/disable features
registry.enable_dynamic_matching(True)
```

### FAISS Configuration

```python
# Check FAISS statistics
stats = registry.get_faiss_statistics()
print(f"FAISS enabled types: {stats['faiss_enabled_types']}")
print(f"Total FAISS types: {stats['total_faiss_types']}")

# Optimize registry for performance
registry.optimize_for_performance()

# Add phrases to existing types
registry.add_phrases_to_type(
    "car_model",
    ["Tesla Cybertruck", "Ford F-150 Lightning"],
    categories=["electric_truck", "electric_truck"]
)
```

### Context-Aware Configuration

```python
from semantic_bud_expressions import ContextAwareExpression

# Configure context matching
expr = ContextAwareExpression(
    expression="I {emotion} {item}",
    expected_context="technology and gadgets",
    context_threshold=0.5,              # Similarity threshold
    context_window='sentence',          # Context extraction method
    context_comparison='direct'         # Comparison strategy
)
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

### Enhanced Examples

```bash
# Basic examples
cd examples
python example.py                    # Basic semantic matching
python example_all_types.py          # All parameter types demo
python example_unified_final.py      # Complete unified system

# NEW: Enhanced phrase matching examples
python test_faiss_phrase_simple.py   # FAISS phrase matching demo
python test_enhanced_phrase_matching.py  # Comprehensive phrase tests
python demo_new_features.py          # Context-aware + FAISS demo

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

## 🌍 Real-World Use Cases

### 1. **E-Commerce Product Matching**
```python
# Match product queries with multi-word product names
registry.create_phrase_parameter_type(
    "product", 
    known_phrases=[
        "iPhone 15 Pro Max", "MacBook Pro 16-inch", 
        "Sony WH-1000XM5", "Samsung Galaxy S24 Ultra"
    ]
)

expr = UnifiedBudExpression("I want to buy {product:phrase}", registry)
match = expr.match("I want to buy iPhone 15 Pro Max")  # ✓ Perfect match
```

### 2. **Smart Customer Support**
```python
# Context-aware intent classification
support_expr = ContextAwareExpression(
    expression="I need help with {issue}",
    expected_context="technical support customer service",
    registry=registry
)

# Only matches in support context
result = support_expr.match_with_context(
    "I'm having technical difficulties. I need help with login"
)
```

### 3. **Content Moderation & Guardrails**
```python
# Detect harmful content with context awareness
moderation_expr = ContextAwareExpression(
    expression="I want to {action} {target}",
    expected_context="violence harmful dangerous",
    context_threshold=0.6,
    registry=registry
)

# High-performance screening (580k+ RPS capability)
def screen_content(text):
    result = moderation_expr.match_with_context(text)
    return result is not None  # True if potentially harmful
```

### 4. **Location & Address Parsing**
```python
# Intelligent address parsing with flexible length
registry.create_phrase_parameter_type(
    "address",
    max_phrase_length=8,
    known_phrases=[
        "123 Main Street", "456 Oak Avenue Suite 100",
        "789 Broadway Floor 5 Office 501"
    ]
)

addr_expr = UnifiedBudExpression("Ship to {address:phrase}", registry)
```

### 5. **Financial Transaction Processing**
```python
# Semantic transaction categorization with large vocabularies
registry.create_semantic_phrase_parameter_type(
    "merchant",
    semantic_categories=[
        "restaurant", "grocery", "gas_station", "pharmacy",
        "electronics", "clothing", "entertainment"
    ],
    max_phrase_length=5
)

# Automatically categorizes "McDonald's Downtown" as restaurant
txn_expr = UnifiedBudExpression(
    "Transaction at {merchant:phrase} for {amount}", 
    registry
)
```

## 🎯 Performance Best Practices

### 1. Use Enhanced Registry for FAISS Acceleration
```python
# Use enhanced registry for automatic FAISS optimization
from semantic_bud_expressions import EnhancedUnifiedParameterTypeRegistry

registry = EnhancedUnifiedParameterTypeRegistry(
    use_faiss=True,                    # Enable FAISS globally
    faiss_auto_threshold=100           # Auto-enable for 100+ phrases
)
registry.initialize_model()
registry.optimize_for_performance()   # Pre-build all indices
```

### 2. Reuse Expressions & Registries
```python
# Good - create once, use many times
expr = UnifiedBudExpression("Hello {name}", registry)
for text in texts:
    match = expr.match(text)  # Uses cached regex + FAISS indices

# Avoid - creating expression in loop
for text in texts:
    expr = UnifiedBudExpression("Hello {name}", registry)  # Recompiles everything
    match = expr.match(text)
```

### 3. Optimize Large Vocabularies with FAISS
```python
# For large phrase vocabularies (1000+ phrases)
large_phrases = load_product_catalog()  # 10,000+ product names

registry.create_phrase_parameter_type(
    "product",
    known_phrases=large_phrases,        # FAISS auto-enables
    max_phrase_length=6
)

# Get 10x performance improvement
expr = UnifiedBudExpression("Buy {product:phrase}", registry)
```

### 4. Context-Aware Performance Tuning
```python
# Optimize context matching for your use case
context_expr = ContextAwareExpression(
    expression="I {action} {item}",
    expected_context="shopping e-commerce",
    context_threshold=0.4,              # Lower = more matches, faster
    context_window=10,                  # Fewer words = faster  
    context_comparison='direct'         # Faster than 'chunked_mean'
)
```

### 5. Batch Process When Possible
```python
# Process multiple texts efficiently
from semantic_bud_expressions import BatchMatcher

batch_matcher = BatchMatcher(registry.model_manager)
phrases = batch_matcher.extract_all_phrases(text)
embeddings = batch_matcher.batch_compute_embeddings(phrases)
```

### 6. Monitor Performance
```python
# Check FAISS usage and performance
stats = registry.get_faiss_statistics()
print(f"FAISS-enabled types: {stats['total_faiss_types']}")

# Verify phrase matching performance
import time
start = time.time()
for text in test_texts:
    match = expr.match(text)
print(f"Average: {(time.time() - start) / len(test_texts) * 1000:.2f}ms per match")
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
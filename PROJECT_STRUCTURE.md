# Project Structure

```
symbolicai/
│
├── semantic_bud_expressions/      # Core Library
│   ├── __init__.py                    # Package exports
│   ├── expression.py                  # Base expression classes
│   ├── semantic_expression.py         # Semantic matching implementation
│   ├── unified_expression.py          # Unified parameter system
│   ├── parameter_type.py              # Base parameter types
│   ├── semantic_parameter_type.py     # Semantic parameter implementation
│   ├── dynamic_semantic_parameter_type.py  # Dynamic type creation
│   ├── unified_parameter_type.py      # Unified type system
│   ├── parameter_type_registry.py     # Type registry base
│   ├── semantic_registry.py           # Semantic-aware registry
│   ├── unified_registry.py            # Unified registry implementation
│   ├── model_manager.py               # Model2Vec integration
│   ├── semantic_cache.py              # Embedding cache
│   ├── math_parameter_type.py         # Math expression support
│   └── ...                            # Other supporting files
│
├── examples/                          # Usage Examples
│   ├── README.md                      # Examples guide
│   ├── quickstart.py                  # Quick start example
│   ├── example.py                     # Basic semantic matching
│   ├── example_all_types.py           # All parameter types demo
│   ├── example_unified_final.py       # Complete unified system
│   ├── example_mixed_types.py         # Mixing parameter types
│   ├── test_unified_system.py         # Unified system tests
│   ├── test_dynamic_semantic.py       # Dynamic semantic tests
│   └── test_all_types.py              # Integration test
│
├── benchmarks/                        # Performance Testing
│   ├── README.md                      # Benchmark guide
│   ├── DETAILED_README.md             # Detailed benchmark documentation
│   ├── requirements.txt               # Benchmark dependencies
│   │
│   ├── scripts/                       # Benchmarking Tools
│   │   ├── benchmark_tool.py          # Comprehensive benchmark
│   │   ├── benchmark_quick.py         # Quick performance test
│   │   ├── benchmark_analysis.py      # Real-world use case analysis
│   │   ├── benchmark_visualizer.py    # Result visualization
│   │   └── ...                        # Analysis scripts
│   │
│   └── results/                       # Benchmark Results
│       ├── benchmark_results.json     # Raw benchmark data
│       ├── performance_summary.txt    # Performance summary
│       ├── latency_vs_rps.png        # Latency charts
│       ├── success_rate_vs_rps.png   # Success rate charts
│       ├── throughput_analysis.png    # Throughput analysis
│       └── ...                        # Other visualizations
│
├── README.md                          # Main documentation
├── CLAUDE.md                          # Claude.ai integration guide
├── requirements.txt                   # Core dependencies
└── .git/                             # Git repository
```

## Key Directories

### `/semantic_bud_expressions`
The core library implementation with all expression types, parameter matching, and AI integration.

### `/examples`
Ready-to-run examples demonstrating various features:
- Basic semantic matching
- All parameter types (semantic, dynamic, phrase, regex, math, quoted)
- Complex expressions combining multiple types
- Test suites for validation

### `/benchmarks`
Comprehensive performance testing suite:
- **scripts/**: Benchmarking tools for different scenarios
- **results/**: Performance data and visualizations
- Shows sub-millisecond latency and 50,000+ ops/sec throughput

## Quick Navigation

- **Get Started**: See `examples/quickstart.py`
- **Run Tests**: Check `examples/test_*.py` files
- **Benchmark**: Use `benchmarks/scripts/benchmark_quick.py`
- **Full Docs**: Read `README.md`
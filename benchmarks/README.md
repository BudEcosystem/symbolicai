# Performance Benchmarks

This directory contains performance benchmarking tools and results for Semantic Bud Expressions.

## Structure

```
benchmarks/
├── scripts/          # Benchmarking tools and scripts
├── results/          # Benchmark results and visualizations  
└── requirements.txt  # Dependencies for benchmarking
```

## Benchmarking Tools

### Main Tools
- `benchmark_tool.py` - Comprehensive performance benchmark
- `benchmark_quick.py` - Quick performance test
- `benchmark_analysis.py` - Real-world use case analysis

### Visualization Tools
- `benchmark_visualizer.py` - Generate charts from results
- `visualize_latency_pattern.py` - Latency pattern analysis
- `generate_visualizations.py` - Additional visualization scripts

## Running Benchmarks

### Install Dependencies
```bash
cd benchmarks
pip install -r requirements.txt
```

### Run Full Benchmark
```bash
python scripts/benchmark_tool.py
# Creates results/benchmark_results.json
```

### Generate Visualizations
```bash
python scripts/benchmark_visualizer.py
# Creates charts in results/
```

### Quick Performance Test
```bash
python scripts/benchmark_quick.py
```

## Performance Results Summary

### Overall Performance
- **Average latency**: 0.046 ms across all tests
- **Maximum throughput**: 55,735 ops/sec
- **100% success rate** at all RPS levels (1-1000)
- **CPU usage**: <1% even at 1000 RPS

### By Expression Type
| Expression Type | Avg Latency | Max Throughput |
|----------------|-------------|----------------|
| Simple         | 0.020 ms    | 50,227 ops/sec |
| Semantic       | 0.018 ms    | 55,735 ops/sec |
| Complex        | 0.031 ms    | 32,513 ops/sec |
| Mixed          | 0.027 ms    | 36,557 ops/sec |

### Real-World Use Cases
All use cases **PASSED** performance requirements:
- **API Guardrails**: 580,000+ RPS (needs 1,000)
- **Semantic Caching**: 168,000+ RPS (needs 500)
- **Natural Language Commands**: 55,000+ RPS (needs 100)
- **Log Analysis**: 397,000+ RPS (needs 5,000)

See `results/` directory for detailed visualizations and reports.
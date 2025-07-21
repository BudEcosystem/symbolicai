#!/usr/bin/env python3
"""
Quick Performance Benchmark for Semantic Bud Expressions

A lightweight version for rapid performance testing focusing on:
- Expression matching speed
- Memory footprint
- Cache effectiveness
"""

import time
import psutil
import gc
from typing import List, Dict, Tuple
import statistics

from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry
)


class QuickBenchmark:
    """Quick benchmarking tool"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def benchmark_expression_types(self):
        """Benchmark different expression types"""
        print("QUICK PERFORMANCE BENCHMARK")
        print("=" * 50)
        
        # Initialize registry
        registry = UnifiedParameterTypeRegistry()
        registry.initialize_model()
        registry.set_dynamic_threshold(0.3)
        
        # Setup parameter types (using unique names to avoid conflicts)
        registry.create_semantic_parameter_type(
            'buy_action', 
            ['buy', 'purchase', 'get', 'obtain'], 
            0.3
        )
        registry.create_phrase_parameter_type('product', 5)
        registry.create_regex_parameter_type('email', r'[^\s]+@[^\s]+\.[^\s]+')
        registry.create_quoted_parameter_type('location')
        
        # Test cases
        test_cases = [
            # Simple
            ("Simple", "{name} loves {fruit}", "John loves apples", 1000),
            
            # Semantic
            ("Semantic", "I want to {buy_action:semantic} {item}", 
             "I want to acquire laptop", 500),
            
            # Dynamic
            ("Dynamic", "{person:dynamic} needs {items:dynamic}", 
             "customer needs assistance", 500),
            
            # Phrase
            ("Phrase", "Buy {product:phrase} now", 
             "Buy MacBook Pro 16 inch now", 300),
            
            # Complex
            ("Complex", "{user:dynamic} wants to {buy_action:semantic} {product:phrase} from {location:quoted}",
             'client wants to purchase luxury sports car from "New York"', 200),
            
            # Math
            ("Math", "Calculate {math} equals", 
             "Calculate 10 + 20 * 3 equals", 500),
        ]
        
        results = []
        
        for name, pattern, test_input, iterations in test_cases:
            print(f"\n{name} Expression:")
            print(f"  Pattern: {pattern}")
            print(f"  Input: {test_input}")
            
            # Create expression
            expr = UnifiedBudExpression(pattern, registry)
            
            # Warm up
            for _ in range(10):
                expr.match(test_input)
            
            # Benchmark
            gc.collect()
            start_mem = self.process.memory_info().rss / 1024 / 1024
            
            latencies = []
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                iter_start = time.perf_counter()
                match = expr.match(test_input)
                iter_end = time.perf_counter()
                latencies.append((iter_end - iter_start) * 1000)
            
            end_time = time.perf_counter()
            end_mem = self.process.memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            throughput = iterations / total_time
            memory_used = end_mem - start_mem
            
            print(f"  Results:")
            print(f"    - Iterations: {iterations}")
            print(f"    - Avg latency: {avg_latency:.3f} ms")
            print(f"    - P95 latency: {p95_latency:.3f} ms")
            print(f"    - Throughput: {throughput:.0f} ops/sec")
            print(f"    - Memory delta: {memory_used:.2f} MB")
            print(f"    - Match success: {match is not None}")
            
            results.append({
                'name': name,
                'avg_latency': avg_latency,
                'p95_latency': p95_latency,
                'throughput': throughput,
                'memory': memory_used
            })
        
        return results
    
    def benchmark_cache_effectiveness(self):
        """Test cache effectiveness through expression matching"""
        print("\n\nCACHE EFFECTIVENESS TEST")
        print("=" * 50)
        
        registry = UnifiedParameterTypeRegistry()
        registry.initialize_model()
        
        # Create expression
        expr = UnifiedBudExpression("I love {fruit}", registry)
        
        # Test with repeated vs unique inputs
        repeated_inputs = ["I love apple", "I love banana", "I love orange"] * 100
        unique_inputs = [f"I love fruit{i}" for i in range(300)]
        
        # Benchmark repeated (cache hits expected)
        start = time.perf_counter()
        for input_text in repeated_inputs:
            expr.match(input_text)
        repeated_time = time.perf_counter() - start
        
        # Create new registry to simulate cache miss scenario
        registry2 = UnifiedParameterTypeRegistry()
        registry2.initialize_model()
        expr2 = UnifiedBudExpression("I love {fruit}", registry2)
        
        # Benchmark unique (cache misses)
        start = time.perf_counter()
        for input_text in unique_inputs:
            expr2.match(input_text)
        unique_time = time.perf_counter() - start
        
        print(f"Repeated inputs (300 calls, 3 unique):")
        print(f"  Total time: {repeated_time:.3f}s")
        print(f"  Avg per match: {repeated_time/300*1000:.3f} ms")
        
        print(f"\nUnique inputs (300 calls, 300 unique):")
        print(f"  Total time: {unique_time:.3f}s")
        print(f"  Avg per match: {unique_time/300*1000:.3f} ms")
        
        print(f"\nCache effectiveness: {unique_time/repeated_time:.1f}x faster with cache")
        print(f"Cache size after test: {len(registry.model_manager.cache.cache)} items")
    
    def benchmark_scaling(self):
        """Test performance at different scales"""
        print("\n\nSCALING TEST")
        print("=" * 50)
        
        registry = UnifiedParameterTypeRegistry()
        registry.initialize_model()
        registry.set_dynamic_threshold(0.3)
        
        # Simple expression for scaling test
        expr = UnifiedBudExpression(
            "User {user:dynamic} performed {action} on {item}",
            registry
        )
        
        scales = [10, 100, 1000, 5000]
        
        for scale in scales:
            gc.collect()
            
            # Generate test data
            test_inputs = [
                f"User user{i%10} performed action{i%5} on item{i%20}"
                for i in range(scale)
            ]
            
            # Benchmark
            start_time = time.perf_counter()
            start_mem = self.process.memory_info().rss / 1024 / 1024
            
            matches = 0
            for test_input in test_inputs:
                if expr.match(test_input):
                    matches += 1
            
            end_time = time.perf_counter()
            end_mem = self.process.memory_info().rss / 1024 / 1024
            
            total_time = end_time - start_time
            throughput = scale / total_time
            memory_used = end_mem - start_mem
            
            print(f"\nScale: {scale} requests")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {throughput:.0f} ops/sec")
            print(f"  Avg latency: {total_time/scale*1000:.3f} ms")
            print(f"  Memory used: {memory_used:.2f} MB")
            print(f"  Success rate: {matches/scale*100:.1f}%")
    
    def run(self):
        """Run all quick benchmarks"""
        # Expression types benchmark
        self.benchmark_expression_types()
        
        # Cache effectiveness
        self.benchmark_cache_effectiveness()
        
        # Scaling test
        self.benchmark_scaling()
        
        print("\n" + "=" * 50)
        print("Quick benchmark complete!")


def main():
    """Main entry point"""
    benchmark = QuickBenchmark()
    benchmark.run()


if __name__ == "__main__":
    main()
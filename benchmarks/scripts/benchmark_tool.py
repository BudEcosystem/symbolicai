#!/usr/bin/env python3
"""
Performance Benchmarking Tool for Semantic Bud Expressions

Measures:
- Compute time (expression creation, matching, parameter extraction)
- Memory usage (baseline, peak, cache growth)
- Scalability at different RPS levels
- Cache effectiveness
- Model loading times
"""

import time
import psutil
import gc
import asyncio
import statistics
import json
import sys
import tracemalloc
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from threading import Lock
import signal
import os

from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry,
    SemanticBudExpression,
    SemanticParameterTypeRegistry
)


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single test"""
    test_name: str
    rps: int
    total_requests: int
    successful_matches: int
    failed_matches: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    memory_used_mb: float
    peak_memory_mb: float
    cache_size: int
    cache_hit_rate: float
    cpu_percent: float
    timestamp: str


class PerformanceBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, output_file: str = "benchmark_results.json"):
        self.output_file = output_file
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()
        self.lock = Lock()
        
    def measure_memory(self) -> Tuple[float, float]:
        """Measure current and peak memory usage in MB"""
        current = self.process.memory_info().rss / 1024 / 1024
        peak = self.process.memory_info().rss / 1024 / 1024  # psutil doesn't track peak easily
        return current, peak
    
    def measure_cpu(self) -> float:
        """Measure CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    def create_test_expressions(self) -> List[Tuple[str, str]]:
        """Create various test expressions and inputs"""
        return [
            # Simple expressions
            ("I love {fruit}", "I love apples"),
            ("I have {count} {items}", "I have 5 cars"),
            
            # Semantic expressions
            ("I am {emotion} about {vehicle:dynamic}", "I am excited about Ferrari"),
            ("The {transport:semantic} is {color}", "The automobile is red"),
            
            # Complex expressions with phrases
            ("Buy {product:phrase} for ${price}", "Buy MacBook Pro 16 inch for $2999"),
            ("I want to {buy_action:semantic} a {item:phrase} from {location:quoted}", 
             'I want to purchase a luxury sports car from "Los Angeles"'),
            
            # Mixed types
            ("{person:dynamic} needs {count} {items:phrase} with ID {id:regex}",
             "customer needs 3 Samsung Galaxy phones with ID A123"),
            
            # Math expressions
            ("Calculate {math} equals {result}", "Calculate 2 + 3 * 4 equals 14"),
            
            # Long expressions
            ("The {customer_type:dynamic} customer {name} wants to {buy_action:semantic} "
             "{count} units of {product:phrase} for ${price} with code {code:regex} "
             "and delivery to {address:quoted} by {date}",
             'The premium customer John wants to buy 10 units of iPhone 15 Pro Max '
             'for $12000 with code SAVE20 and delivery to "123 Main St" by tomorrow'),
        ]
    
    def benchmark_single_expression(
        self, 
        expression: str, 
        test_input: str,
        registry: UnifiedParameterTypeRegistry,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark a single expression"""
        latencies = []
        
        # Create expression once
        expr = UnifiedBudExpression(expression, registry)
        
        # Warm up
        for _ in range(10):
            expr.match(test_input)
        
        # Actual benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            match = expr.match(test_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            "latencies": latencies,
            "success": all(expr.match(test_input) for _ in range(5))
        }
    
    def benchmark_rps(
        self,
        target_rps: int,
        duration_seconds: int = 10,
        expression_type: str = "mixed"
    ) -> BenchmarkResult:
        """Benchmark at specific RPS for given duration"""
        print(f"\nBenchmarking at {target_rps} RPS for {duration_seconds}s...")
        
        # Initialize registry
        registry = UnifiedParameterTypeRegistry()
        registry.initialize_model()
        registry.set_dynamic_threshold(0.3)
        
        # Create parameter types
        registry.create_semantic_parameter_type(
            'transport', 
            ['car', 'bus', 'train', 'plane'], 
            0.4
        )
        registry.create_semantic_parameter_type(
            'buy_action',
            ['buy', 'purchase', 'get', 'obtain'],
            0.3
        )
        registry.create_phrase_parameter_type('product', 5)
        registry.create_phrase_parameter_type('item', 4)
        registry.create_phrase_parameter_type('items', 3)
        registry.create_regex_parameter_type('id', r'[A-Z]\d{3}')
        registry.create_regex_parameter_type('code', r'[A-Z]{4}\d{2}')
        registry.create_quoted_parameter_type('location')
        registry.create_quoted_parameter_type('address')
        
        # Get test expressions
        test_cases = self.create_test_expressions()
        if expression_type == "simple":
            test_cases = test_cases[:2]
        elif expression_type == "semantic":
            test_cases = test_cases[2:4]
        elif expression_type == "complex":
            test_cases = test_cases[4:6]
        
        # Tracking variables
        all_latencies = []
        successful_matches = 0
        failed_matches = 0
        
        # Start memory tracking
        tracemalloc.start()
        start_memory, _ = self.measure_memory()
        
        # Calculate delay between requests
        delay = 1.0 / target_rps if target_rps > 0 else 0
        
        # Run benchmark
        start_time = time.time()
        request_count = 0
        
        with ThreadPoolExecutor(max_workers=min(target_rps, 100)) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                # Select a random test case
                expr, test_input = test_cases[request_count % len(test_cases)]
                
                # Submit request
                future = executor.submit(
                    self.benchmark_single_expression,
                    expr, test_input, registry, 1
                )
                futures.append(future)
                
                request_count += 1
                
                # Control RPS
                if delay > 0:
                    time.sleep(delay)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=5)
                    all_latencies.extend(result["latencies"])
                    if result["success"]:
                        successful_matches += 1
                    else:
                        failed_matches += 1
                except Exception as e:
                    failed_matches += 1
                    print(f"Error in request: {e}")
        
        # Calculate metrics
        end_memory, peak_memory = self.measure_memory()
        memory_used = end_memory - start_memory
        
        # Get cache stats
        cache_size = len(registry.model_manager.cache.cache)
        cache_hit_rate = 0.0  # Would need to implement cache hit tracking
        
        # CPU usage
        cpu_percent = self.measure_cpu()
        
        # Latency percentiles
        if all_latencies:
            latencies_sorted = sorted(all_latencies)
            result = BenchmarkResult(
                test_name=f"{expression_type}_expressions",
                rps=target_rps,
                total_requests=request_count,
                successful_matches=successful_matches,
                failed_matches=failed_matches,
                avg_latency_ms=statistics.mean(all_latencies),
                p50_latency_ms=np.percentile(latencies_sorted, 50),
                p95_latency_ms=np.percentile(latencies_sorted, 95),
                p99_latency_ms=np.percentile(latencies_sorted, 99),
                max_latency_ms=max(all_latencies),
                min_latency_ms=min(all_latencies),
                memory_used_mb=memory_used,
                peak_memory_mb=peak_memory,
                cache_size=cache_size,
                cache_hit_rate=cache_hit_rate,
                cpu_percent=cpu_percent,
                timestamp=datetime.now().isoformat()
            )
        else:
            # No successful requests
            result = BenchmarkResult(
                test_name=f"{expression_type}_expressions",
                rps=target_rps,
                total_requests=request_count,
                successful_matches=0,
                failed_matches=request_count,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                max_latency_ms=0,
                min_latency_ms=0,
                memory_used_mb=memory_used,
                peak_memory_mb=peak_memory,
                cache_size=cache_size,
                cache_hit_rate=0,
                cpu_percent=cpu_percent,
                timestamp=datetime.now().isoformat()
            )
        
        # Stop memory tracking
        tracemalloc.stop()
        
        return result
    
    def benchmark_model_loading(self) -> Dict[str, float]:
        """Benchmark semantic bud expressions initialization and first use"""
        print("\nBenchmarking system initialization...")
        
        times = {}
        
        # Time registry initialization
        start = time.perf_counter()
        registry = UnifiedParameterTypeRegistry()
        end = time.perf_counter()
        times["registry_init_ms"] = (end - start) * 1000
        
        # Time model loading
        start = time.perf_counter()
        registry.initialize_model()
        end = time.perf_counter()
        times["model_load_ms"] = (end - start) * 1000
        
        # Time creating first expression
        start = time.perf_counter()
        expr = UnifiedBudExpression("I love {fruit}", registry)
        end = time.perf_counter()
        times["first_expression_create_ms"] = (end - start) * 1000
        
        # Time first match (includes first embedding)
        start = time.perf_counter()
        match = expr.match("I love apples")
        end = time.perf_counter()
        times["first_match_ms"] = (end - start) * 1000
        
        # Time subsequent match (should use cache)
        start = time.perf_counter()
        match = expr.match("I love apples")
        end = time.perf_counter()
        times["cached_match_ms"] = (end - start) * 1000
        
        # Time creating semantic parameter type
        start = time.perf_counter()
        registry.create_semantic_parameter_type(
            'vehicle', 
            ['car', 'truck', 'bus'], 
            0.5
        )
        end = time.perf_counter()
        times["create_semantic_type_ms"] = (end - start) * 1000
        
        # Time creating complex expression
        start = time.perf_counter()
        complex_expr = UnifiedBudExpression(
            "{person:dynamic} wants to {action:semantic} {product:phrase}",
            registry
        )
        end = time.perf_counter()
        times["complex_expression_create_ms"] = (end - start) * 1000
        
        return times
    
    def benchmark_cache_performance(self, num_items: int = 1000) -> Dict[str, Any]:
        """Benchmark cache performance through expression matching"""
        print(f"\nBenchmarking cache performance with {num_items} unique expressions...")
        
        registry = UnifiedParameterTypeRegistry()
        registry.initialize_model()
        
        # Create expressions with different fruits to test cache
        fruits = [f"fruit{i}" for i in range(num_items)]
        expr = UnifiedBudExpression("I love {fruit}", registry)
        
        # First pass - populate cache
        start = time.perf_counter()
        for fruit in fruits:
            expr.match(f"I love {fruit}")
        first_pass_time = time.perf_counter() - start
        
        # Second pass - should hit cache
        start = time.perf_counter()
        for fruit in fruits:
            expr.match(f"I love {fruit}")
        second_pass_time = time.perf_counter() - start
        
        # Test with semantic matching
        semantic_expr = UnifiedBudExpression("I am {emotion} about this", registry)
        emotions = ["happy", "sad", "excited", "angry", "frustrated"] * 20  # 100 calls, 5 unique
        
        start = time.perf_counter()
        for emotion in emotions:
            semantic_expr.match(f"I am {emotion} about this")
        semantic_time = time.perf_counter() - start
        
        # Estimate cache size
        cache_size = len(registry.model_manager.cache.cache)
        cache_memory = sys.getsizeof(registry.model_manager.cache.cache) / 1024 / 1024
        
        return {
            "num_items": num_items,
            "first_pass_ms": first_pass_time * 1000,
            "second_pass_ms": second_pass_time * 1000,
            "cache_speedup": first_pass_time / second_pass_time if second_pass_time > 0 else 0,
            "semantic_100_calls_ms": semantic_time * 1000,
            "cache_size": cache_size,
            "cache_memory_mb": cache_memory
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting Semantic Bud Expressions Performance Benchmark")
        print("=" * 60)
        
        # 1. System initialization benchmark
        model_times = self.benchmark_model_loading()
        print(f"\nSystem Initialization Times:")
        for key, value in model_times.items():
            print(f"  {key}: {value:.2f} ms")
        
        # 2. Cache performance
        cache_perf = self.benchmark_cache_performance(1000)
        print(f"\nCache Performance:")
        print(f"  Items tested: {cache_perf['num_items']}")
        print(f"  First pass: {cache_perf['first_pass_ms']:.1f} ms")
        print(f"  Second pass (cached): {cache_perf['second_pass_ms']:.1f} ms")
        print(f"  Cache speedup: {cache_perf['cache_speedup']:.1f}x")
        print(f"  Cache size: {cache_perf['cache_size']} items")
        print(f"  Memory: {cache_perf['cache_memory_mb']:.2f} MB")
        
        # 3. RPS benchmarks
        rps_levels = [1, 10, 50, 100, 500, 1000]
        expression_types = ["simple", "semantic", "complex", "mixed"]
        
        all_results = {
            "model_loading": model_times,
            "cache_performance": cache_perf,
            "rps_benchmarks": []
        }
        
        for expr_type in expression_types:
            print(f"\n\nBenchmarking {expr_type} expressions:")
            print("-" * 40)
            
            for rps in rps_levels:
                try:
                    # Force garbage collection before each test
                    gc.collect()
                    
                    result = self.benchmark_rps(
                        target_rps=rps,
                        duration_seconds=10,
                        expression_type=expr_type
                    )
                    
                    self.results.append(result)
                    all_results["rps_benchmarks"].append(asdict(result))
                    
                    # Print summary
                    print(f"\nRPS: {rps}")
                    print(f"  Success Rate: {result.successful_matches/result.total_requests*100:.1f}%")
                    print(f"  Avg Latency: {result.avg_latency_ms:.2f} ms")
                    print(f"  P95 Latency: {result.p95_latency_ms:.2f} ms")
                    print(f"  P99 Latency: {result.p99_latency_ms:.2f} ms")
                    print(f"  Memory Used: {result.memory_used_mb:.2f} MB")
                    print(f"  CPU Usage: {result.cpu_percent:.1f}%")
                    
                except Exception as e:
                    print(f"Error benchmarking at {rps} RPS: {e}")
                    traceback.print_exc()
        
        # Save results
        with open(self.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\nBenchmark complete! Results saved to {self.output_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            return
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Group by expression type
        by_type = {}
        for result in self.results:
            if result.test_name not in by_type:
                by_type[result.test_name] = []
            by_type[result.test_name].append(result)
        
        for expr_type, results in by_type.items():
            print(f"\n{expr_type.upper()}:")
            print(f"{'RPS':<8} {'Success%':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Memory(MB)':<12} {'CPU%':<8}")
            print("-" * 80)
            
            for r in sorted(results, key=lambda x: x.rps):
                success_rate = (r.successful_matches / r.total_requests * 100) if r.total_requests > 0 else 0
                print(f"{r.rps:<8} {success_rate:<10.1f} {r.avg_latency_ms:<10.2f} "
                      f"{r.p95_latency_ms:<10.2f} {r.p99_latency_ms:<10.2f} "
                      f"{r.memory_used_mb:<12.2f} {r.cpu_percent:<8.1f}")


def main():
    """Main entry point"""
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nBenchmark interrupted. Saving partial results...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
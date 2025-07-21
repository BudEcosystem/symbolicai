#!/usr/bin/env python3
"""
Simple test to demonstrate cache performance improvements
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry,
    get_global_cache,
    clear_global_cache
)


def main():
    print("=== Cache Performance Test ===\n")
    
    # Initialize registry
    print("Initializing model...")
    registry = UnifiedParameterTypeRegistry()
    registry.initialize_model()
    
    # Clear cache to start fresh
    clear_global_cache()
    
    # Test expression
    expr = UnifiedBudExpression("Hello {name}, I love {fruit}", registry)
    test_text = "Hello John, I love apples"
    
    # First match - no cache
    print("\nFirst match (cache miss):")
    start = time.perf_counter()
    result1 = expr.match(test_text)
    time1 = (time.perf_counter() - start) * 1000
    print(f"  Time: {time1:.4f} ms")
    print(f"  Result: {result1 is not None}")
    
    # Second match - should hit cache
    print("\nSecond match (cache hit):")
    start = time.perf_counter()
    result2 = expr.match(test_text)
    time2 = (time.perf_counter() - start) * 1000
    print(f"  Time: {time2:.4f} ms")
    print(f"  Result: {result2 is not None}")
    
    # Show speedup
    if time2 > 0:
        speedup = time1 / time2
        print(f"\nSpeedup: {speedup:.2f}x")
    
    # Show cache stats
    cache = get_global_cache()
    stats = cache.get_stats()
    print(f"\nCache statistics:")
    print(f"  L1 hits: {stats['l1_hits']}, misses: {stats['l1_misses']}")
    print(f"  L2 hits: {stats['l2_hits']}, misses: {stats['l2_misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    # Test with multiple expressions
    print("\n\n=== Multiple Expression Test ===")
    clear_global_cache()
    
    expressions = [
        ("I love {fruit}", "I love apples"),
        ("Hello {greeting}", "Hello there"),
        ("The {animal} is {color}", "The cat is black"),
    ]
    
    # First pass
    print("\nFirst pass (building cache):")
    start = time.perf_counter()
    for pattern, text in expressions:
        expr = UnifiedBudExpression(pattern, registry)
        expr.match(text)
    first_pass_time = (time.perf_counter() - start) * 1000
    print(f"  Total time: {first_pass_time:.4f} ms")
    
    # Second pass
    print("\nSecond pass (using cache):")
    start = time.perf_counter()
    for pattern, text in expressions:
        expr = UnifiedBudExpression(pattern, registry)
        expr.match(text)
    second_pass_time = (time.perf_counter() - start) * 1000
    print(f"  Total time: {second_pass_time:.4f} ms")
    
    print(f"\nSpeedup: {first_pass_time / second_pass_time:.2f}x")
    
    # Final cache stats
    stats = cache.get_stats()
    print(f"\nFinal cache statistics:")
    print(f"  Total hits: {stats['total_hits']}")
    print(f"  Total misses: {stats['total_misses']}")
    print(f"  Overall hit rate: {stats['hit_rate']:.2%}")


if __name__ == "__main__":
    main()
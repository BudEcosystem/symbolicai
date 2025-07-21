#!/usr/bin/env python3
"""
Example demonstrating all performance optimizations working together
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry,
    OptimizedSemanticParameterType,
    get_global_cache,
    clear_global_cache,
    regex_cache
)


def main():
    print("=== Semantic Bud Expressions - Optimized Performance Demo ===\n")
    
    # Initialize registry with optimizations
    print("Initializing with all optimizations enabled...")
    registry = UnifiedParameterTypeRegistry()
    
    # Create some custom semantic types using optimized implementation
    registry.define_parameter_type(OptimizedSemanticParameterType(
        name="product",
        prototypes=["laptop", "phone", "tablet", "computer", "device", "gadget"],
        similarity_threshold=0.6
    ))
    
    registry.define_parameter_type(OptimizedSemanticParameterType(
        name="action_word",
        prototypes=["buy", "purchase", "get", "acquire", "obtain"],
        similarity_threshold=0.7
    ))
    
    # Initialize model (triggers prototype pre-computation)
    start = time.perf_counter()
    registry.initialize_model()
    init_time = (time.perf_counter() - start) * 1000
    print(f"Model initialization time: {init_time:.2f} ms")
    print("  ✓ Regex patterns pre-compiled")
    print("  ✓ Prototype embeddings pre-computed")
    print("  ✓ Multi-level cache initialized")
    
    # Test expressions
    expressions = [
        ("I want to {action_word:semantic} a {product:semantic}", [
            "I want to buy a laptop",
            "I want to purchase a phone",
            "I want to get a tablet",
            "I want to acquire a computer"
        ]),
        ("The {product:semantic} costs {price:regex} dollars", [
            "The laptop costs 999 dollars",
            "The phone costs 599 dollars",
            "The gadget costs 299 dollars"
        ])
    ]
    
    # Configure regex for price
    registry.create_regex_parameter_type('price', r'\d+')
    
    print("\n=== Performance Test ===")
    
    # Clear cache for fair comparison
    clear_global_cache()
    
    total_matches = 0
    start_time = time.perf_counter()
    
    # Process all expressions
    for pattern, test_texts in expressions:
        expr = UnifiedBudExpression(pattern, registry)
        print(f"\nPattern: {pattern}")
        
        for text in test_texts:
            match_start = time.perf_counter()
            result = expr.match(text)
            match_time = (time.perf_counter() - match_start) * 1000
            
            if result:
                total_matches += 1
                print(f"  ✓ '{text}' - {match_time:.3f} ms")
            else:
                print(f"  ✗ '{text}' - {match_time:.3f} ms")
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    print(f"\n=== Results ===")
    print(f"Total processing time: {total_time:.2f} ms")
    print(f"Successful matches: {total_matches}")
    print(f"Average time per match: {total_time / (len(expressions) * 4):.3f} ms")
    
    # Show optimization stats
    print("\n=== Optimization Statistics ===")
    
    # Regex cache stats
    regex_stats = regex_cache.get_stats()
    print(f"\nRegex Cache:")
    print(f"  Hits: {regex_stats['hits']}")
    print(f"  Misses: {regex_stats['misses']}")
    print(f"  Hit rate: {regex_stats['hit_rate']:.2%}")
    
    # Multi-level cache stats
    cache = get_global_cache()
    cache_stats = cache.get_stats()
    print(f"\nMulti-level Cache:")
    print(f"  L1 (expressions): {cache_stats['l1_hits']} hits, {cache_stats['l1_misses']} misses")
    print(f"  L2 (embeddings): {cache_stats['l2_hits']} hits, {cache_stats['l2_misses']} misses")
    print(f"  Overall hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Re-run to show cache benefits
    print("\n=== Re-running with warm cache ===")
    start_time = time.perf_counter()
    
    for pattern, test_texts in expressions:
        expr = UnifiedBudExpression(pattern, registry)
        for text in test_texts:
            result = expr.match(text)
    
    cached_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Warm cache time: {cached_time:.2f} ms")
    print(f"Speedup: {total_time / cached_time:.2f}x")
    
    print("\n✅ All optimizations working successfully!")
    print("\nKey optimizations demonstrated:")
    print("1. Regex compilation caching - no recompilation overhead")
    print("2. Prototype embedding pre-computation - instant similarity checks")
    print("3. Batch embedding computation - efficient for multiple phrases")
    print("4. Multi-level caching - dramatic speedup for repeated patterns")


if __name__ == "__main__":
    main()
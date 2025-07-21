#!/usr/bin/env python3
"""
Performance Analysis Tool for Semantic Bud Expressions

Provides detailed analysis of performance characteristics for real-world use cases
"""

import time
import statistics
import json
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from semantic_bud_expressions import (
    UnifiedBudExpression,
    UnifiedParameterTypeRegistry
)


@dataclass
class UseCase:
    """Represents a real-world use case for benchmarking"""
    name: str
    description: str
    expressions: List[Tuple[str, List[str]]]  # (pattern, test_inputs)
    expected_rps: int  # Expected requests per second


class PerformanceAnalyzer:
    """Analyzes performance for real-world use cases"""
    
    def __init__(self):
        self.use_cases = self._define_use_cases()
        
    def _define_use_cases(self) -> List[UseCase]:
        """Define real-world use cases for benchmarking"""
        return [
            UseCase(
                name="API Guardrails",
                description="Content filtering and validation for API responses",
                expressions=[
                    (
                        "The response contains {forbidden_word:semantic} content",
                        [
                            "The response contains inappropriate content",
                            "The response contains offensive content",
                            "The response contains harmful content",
                            "The response contains dangerous content"
                        ]
                    ),
                    (
                        "User is trying to {malicious_action:semantic} the system",
                        [
                            "User is trying to hack the system",
                            "User is trying to exploit the system",
                            "User is trying to breach the system",
                            "User is trying to compromise the system"
                        ]
                    )
                ],
                expected_rps=1000
            ),
            UseCase(
                name="Semantic Caching",
                description="Intelligent caching based on semantic similarity",
                expressions=[
                    (
                        "Get {data_type:semantic} for {entity:phrase}",
                        [
                            "Get statistics for user John Smith",
                            "Get metrics for user Jane Doe",
                            "Get analytics for company Acme Corp",
                            "Get information for product iPhone 15"
                        ]
                    ),
                    (
                        "Calculate {metric:semantic} for {time_period:phrase}",
                        [
                            "Calculate revenue for last quarter",
                            "Calculate profit for Q3 2024",
                            "Calculate income for past month",
                            "Calculate earnings for this year"
                        ]
                    )
                ],
                expected_rps=500
            ),
            UseCase(
                name="Natural Language Commands",
                description="Processing natural language commands for automation",
                expressions=[
                    (
                        "{assistant:dynamic}, {action:semantic} the {device:phrase} in {location:quoted}",
                        [
                            'Alexa, turn on the living room lights in "master bedroom"',
                            'Hey Google, switch off the smart TV in "kitchen"',
                            'Siri, activate the air conditioner in "office room"',
                            'Assistant, enable the security system in "garage"'
                        ]
                    ),
                    (
                        "Schedule {task:phrase} for {time:phrase} with {participants:quoted}",
                        [
                            'Schedule team meeting for tomorrow at 3 PM with "John, Jane, Bob"',
                            'Schedule code review for next Monday with "dev team"',
                            'Schedule project update for Friday afternoon with "stakeholders"',
                            'Schedule training session for next week with "new hires"'
                        ]
                    )
                ],
                expected_rps=100
            ),
            UseCase(
                name="Log Analysis",
                description="Pattern matching in system logs",
                expressions=[
                    (
                        "{severity:regex} - {component:phrase} {error_type:semantic} at {timestamp}",
                        [
                            "ERROR - Database Connection failed at 2024-01-01T10:30:00",
                            "WARN - API Gateway timeout at 2024-01-01T10:31:00",
                            "CRITICAL - Authentication Service crashed at 2024-01-01T10:32:00",
                            "INFO - Cache Manager initialized at 2024-01-01T10:33:00"
                        ]
                    )
                ],
                expected_rps=5000
            )
        ]
    
    def analyze_use_case(self, use_case: UseCase) -> Dict[str, Any]:
        """Analyze performance for a specific use case"""
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {use_case.name}")
        print(f"Description: {use_case.description}")
        print(f"Expected RPS: {use_case.expected_rps}")
        print(f"{'=' * 60}")
        
        # Initialize registry
        registry = UnifiedParameterTypeRegistry()
        registry.initialize_model()
        registry.set_dynamic_threshold(0.3)
        
        # Setup parameter types for this use case
        self._setup_parameter_types(registry, use_case)
        
        results = {
            "use_case": use_case.name,
            "expected_rps": use_case.expected_rps,
            "expressions": []
        }
        
        for pattern, test_inputs in use_case.expressions:
            print(f"\nPattern: {pattern}")
            
            # Create expression
            expr = UnifiedBudExpression(pattern, registry)
            
            # Warm up
            for _ in range(50):
                for test_input in test_inputs:
                    expr.match(test_input)
            
            # Benchmark
            latencies = []
            success_count = 0
            
            iterations = 100
            for _ in range(iterations):
                for test_input in test_inputs:
                    start = time.perf_counter()
                    match = expr.match(test_input)
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)
                    if match:
                        success_count += 1
            
            total_requests = iterations * len(test_inputs)
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            # Calculate achievable RPS
            achievable_rps = 1000 / avg_latency if avg_latency > 0 else 0
            
            expr_result = {
                "pattern": pattern,
                "test_inputs": len(test_inputs),
                "total_requests": total_requests,
                "success_rate": success_count / total_requests * 100,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "achievable_rps": achievable_rps,
                "meets_requirement": achievable_rps >= use_case.expected_rps
            }
            
            results["expressions"].append(expr_result)
            
            # Print results
            print(f"  Success Rate: {expr_result['success_rate']:.1f}%")
            print(f"  Avg Latency: {avg_latency:.3f} ms")
            print(f"  P95 Latency: {p95_latency:.3f} ms")
            print(f"  P99 Latency: {p99_latency:.3f} ms")
            print(f"  Achievable RPS: {achievable_rps:.0f}")
            print(f"  Meets Requirement: {'✓' if expr_result['meets_requirement'] else '✗'}")
        
        return results
    
    def _setup_parameter_types(self, registry: UnifiedParameterTypeRegistry, use_case: UseCase):
        """Setup parameter types for specific use case"""
        if use_case.name == "API Guardrails":
            registry.create_semantic_parameter_type(
                'forbidden_word',
                ['inappropriate', 'offensive', 'harmful', 'dangerous', 'explicit'],
                0.4
            )
            registry.create_semantic_parameter_type(
                'malicious_action',
                ['hack', 'exploit', 'breach', 'attack', 'compromise'],
                0.4
            )
        
        elif use_case.name == "Semantic Caching":
            registry.create_semantic_parameter_type(
                'data_type',
                ['statistics', 'metrics', 'analytics', 'data', 'information'],
                0.5
            )
            registry.create_semantic_parameter_type(
                'metric',
                ['revenue', 'profit', 'income', 'sales', 'earnings'],
                0.5
            )
            registry.create_phrase_parameter_type('entity', 5)
            registry.create_phrase_parameter_type('time_period', 4)
        
        elif use_case.name == "Natural Language Commands":
            registry.create_semantic_parameter_type(
                'nlc_action',  # Renamed to avoid conflict
                ['turn on', 'switch off', 'activate', 'enable', 'disable'],
                0.4
            )
            registry.create_phrase_parameter_type('device', 4)
            registry.create_phrase_parameter_type('task', 3)
            registry.create_phrase_parameter_type('time', 5)
            registry.create_quoted_parameter_type('participants')
        
        elif use_case.name == "Log Analysis":
            registry.create_regex_parameter_type(
                'severity',
                r'(DEBUG|INFO|WARN|ERROR|CRITICAL)'
            )
            registry.create_phrase_parameter_type('component', 3)
            registry.create_semantic_parameter_type(
                'error_type',
                ['failed', 'timeout', 'crashed', 'error', 'initialized'],
                0.5
            )
    
    def run_full_analysis(self) -> List[Dict[str, Any]]:
        """Run analysis for all use cases"""
        all_results = []
        
        print("SEMANTIC BUD EXPRESSIONS - PERFORMANCE ANALYSIS")
        print("Real-World Use Case Benchmarks")
        
        for use_case in self.use_cases:
            result = self.analyze_use_case(use_case)
            all_results.append(result)
        
        # Generate summary
        self._print_summary(all_results)
        
        # Save results
        with open('performance_analysis.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['use_case']}:")
            print(f"  Expected RPS: {result['expected_rps']}")
            
            all_meet = all(expr['meets_requirement'] for expr in result['expressions'])
            print(f"  Overall Status: {'✓ PASS' if all_meet else '✗ FAIL'}")
            
            for expr in result['expressions']:
                print(f"  - Pattern: {expr['pattern'][:50]}...")
                print(f"    Achievable RPS: {expr['achievable_rps']:.0f}")
                print(f"    Avg Latency: {expr['avg_latency_ms']:.3f} ms")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS:")
        print("=" * 60)
        
        for result in results:
            if not all(expr['meets_requirement'] for expr in result['expressions']):
                print(f"\n{result['use_case']}:")
                for expr in result['expressions']:
                    if not expr['meets_requirement']:
                        deficit = result['expected_rps'] - expr['achievable_rps']
                        print(f"  - Pattern needs {deficit:.0f} more RPS")
                        print(f"    Consider: Caching, parallel processing, or optimization")


def main():
    """Main entry point"""
    analyzer = PerformanceAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
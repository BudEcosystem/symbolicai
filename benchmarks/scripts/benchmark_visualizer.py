#!/usr/bin/env python3
"""
Visualization tool for benchmark results

Creates charts and graphs from benchmark data to visualize:
- Latency vs RPS
- Memory usage trends
- CPU utilization
- Success rates
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import pandas as pd
import seaborn as sns
from datetime import datetime


class BenchmarkVisualizer:
    """Visualize benchmark results"""
    
    def __init__(self, results_file: str = "benchmark_results.json"):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.data['rps_benchmarks'])
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_latency_vs_rps(self):
        """Plot latency metrics vs RPS for each expression type"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Latency vs RPS by Expression Type', fontsize=16)
        
        expr_types = self.df['test_name'].unique()
        
        for idx, expr_type in enumerate(expr_types):
            ax = axes[idx // 2, idx % 2]
            
            # Filter data for this expression type
            expr_data = self.df[self.df['test_name'] == expr_type]
            
            # Sort by RPS
            expr_data = expr_data.sort_values('rps')
            
            # Plot different percentiles
            ax.plot(expr_data['rps'], expr_data['avg_latency_ms'], 
                   'o-', label='Average', linewidth=2)
            ax.plot(expr_data['rps'], expr_data['p50_latency_ms'], 
                   's-', label='P50', linewidth=1.5)
            ax.plot(expr_data['rps'], expr_data['p95_latency_ms'], 
                   '^-', label='P95', linewidth=1.5)
            ax.plot(expr_data['rps'], expr_data['p99_latency_ms'], 
                   'd-', label='P99', linewidth=1.5)
            
            ax.set_xlabel('Requests Per Second (RPS)')
            ax.set_ylabel('Latency (ms)')
            ax.set_title(expr_type.replace('_', ' ').title())
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('latency_vs_rps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_success_rate(self):
        """Plot success rate vs RPS"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for expr_type in self.df['test_name'].unique():
            expr_data = self.df[self.df['test_name'] == expr_type]
            expr_data = expr_data.sort_values('rps')
            
            # Calculate success rate
            success_rate = (expr_data['successful_matches'] / 
                          expr_data['total_requests'] * 100)
            
            ax.plot(expr_data['rps'], success_rate, 'o-', 
                   label=expr_type.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Requests Per Second (RPS)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate vs RPS by Expression Type')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.savefig('success_rate_vs_rps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_resource_usage(self):
        """Plot memory and CPU usage vs RPS"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory usage
        for expr_type in self.df['test_name'].unique():
            expr_data = self.df[self.df['test_name'] == expr_type]
            expr_data = expr_data.sort_values('rps')
            
            ax1.plot(expr_data['rps'], expr_data['memory_used_mb'], 'o-',
                    label=expr_type.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Requests Per Second (RPS)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage vs RPS')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CPU usage
        for expr_type in self.df['test_name'].unique():
            expr_data = self.df[self.df['test_name'] == expr_type]
            expr_data = expr_data.sort_values('rps')
            
            ax2.plot(expr_data['rps'], expr_data['cpu_percent'], 'o-',
                    label=expr_type.replace('_', ' ').title(), linewidth=2)
        
        ax2.set_xlabel('Requests Per Second (RPS)')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_title('CPU Usage vs RPS')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resource_usage_vs_rps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_latency_distribution(self):
        """Plot latency distribution heatmap"""
        # Create a pivot table for the heatmap
        pivot_data = self.df.pivot_table(
            values='p95_latency_ms',
            index='test_name',
            columns='rps'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'P95 Latency (ms)'})
        
        plt.title('P95 Latency Heatmap: Expression Type vs RPS')
        plt.xlabel('Requests Per Second (RPS)')
        plt.ylabel('Expression Type')
        
        # Fix y-axis labels
        plt.gca().set_yticklabels([label.get_text().replace('_', ' ').title() 
                                   for label in plt.gca().get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig('latency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_loading_times(self):
        """Plot model loading times"""
        model_times = self.data['model_loading']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        operations = list(model_times.keys())
        times = list(model_times.values())
        
        # Clean up operation names
        operations = [op.replace('_', ' ').title().replace(' Ms', '') for op in operations]
        
        bars = ax.bar(operations, times)
        
        # Color code bars
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.1f} ms', ha='center', va='bottom')
        
        ax.set_ylabel('Time (ms)')
        ax.set_title('Model Loading and Initialization Times')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('model_loading_times.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        report = []
        report.append("SEMANTIC BUD EXPRESSIONS - PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System initialization summary
        report.append("SYSTEM INITIALIZATION PERFORMANCE:")
        report.append("-" * 30)
        for key, value in self.data['model_loading'].items():
            key_formatted = key.replace('_', ' ').title().replace(' Ms', '')
            report.append(f"  {key_formatted}: {value:.2f} ms")
        report.append("")
        
        # Cache performance summary
        cache = self.data['cache_performance']
        report.append("CACHE PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"  Items tested: {cache['num_items']}")
        report.append(f"  First pass time: {cache['first_pass_ms']:.1f} ms")
        report.append(f"  Cached pass time: {cache['second_pass_ms']:.1f} ms")
        report.append(f"  Cache speedup: {cache['cache_speedup']:.1f}x")
        report.append(f"  Cache size: {cache['cache_size']} items")
        report.append(f"  Memory usage: {cache['cache_memory_mb']:.2f} MB")
        report.append("")
        
        # RPS performance by expression type
        for expr_type in self.df['test_name'].unique():
            report.append(f"{expr_type.upper()}:")
            report.append("-" * 30)
            
            expr_data = self.df[self.df['test_name'] == expr_type]
            
            # Find max sustainable RPS (>95% success rate)
            sustainable = expr_data[
                (expr_data['successful_matches'] / expr_data['total_requests']) > 0.95
            ]
            max_sustainable_rps = sustainable['rps'].max() if not sustainable.empty else 0
            
            report.append(f"  Max sustainable RPS (>95% success): {max_sustainable_rps}")
            
            # Get metrics at different RPS levels
            for _, row in expr_data.iterrows():
                success_rate = row['successful_matches'] / row['total_requests'] * 100
                report.append(f"  At {row['rps']} RPS:")
                report.append(f"    - Success rate: {success_rate:.1f}%")
                report.append(f"    - Avg latency: {row['avg_latency_ms']:.2f} ms")
                report.append(f"    - P95 latency: {row['p95_latency_ms']:.2f} ms")
                report.append(f"    - Memory: {row['memory_used_mb']:.2f} MB")
                report.append(f"    - CPU: {row['cpu_percent']:.1f}%")
            report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        report.append("-" * 30)
        
        # Find best and worst performing expression types
        avg_latencies = self.df.groupby('test_name')['avg_latency_ms'].mean()
        best_expr = avg_latencies.idxmin()
        worst_expr = avg_latencies.idxmax()
        
        report.append(f"  Best performing: {best_expr} (avg {avg_latencies[best_expr]:.2f} ms)")
        report.append(f"  Most complex: {worst_expr} (avg {avg_latencies[worst_expr]:.2f} ms)")
        
        # Memory efficiency
        max_memory = self.df['memory_used_mb'].max()
        report.append(f"  Max memory usage: {max_memory:.2f} MB")
        
        # Save report
        with open('benchmark_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report
    
    def create_all_visualizations(self):
        """Create all visualization charts"""
        print("Generating visualizations...")
        
        try:
            self.plot_latency_vs_rps()
            print("✓ Latency vs RPS chart created")
        except Exception as e:
            print(f"✗ Error creating latency chart: {e}")
        
        try:
            self.plot_success_rate()
            print("✓ Success rate chart created")
        except Exception as e:
            print(f"✗ Error creating success rate chart: {e}")
        
        try:
            self.plot_resource_usage()
            print("✓ Resource usage charts created")
        except Exception as e:
            print(f"✗ Error creating resource charts: {e}")
        
        try:
            self.plot_latency_distribution()
            print("✓ Latency heatmap created")
        except Exception as e:
            print(f"✗ Error creating heatmap: {e}")
        
        try:
            self.plot_model_loading_times()
            print("✓ Model loading chart created")
        except Exception as e:
            print(f"✗ Error creating model loading chart: {e}")
        
        try:
            self.generate_report()
            print("✓ Benchmark report generated")
        except Exception as e:
            print(f"✗ Error generating report: {e}")
        
        print("\nAll visualizations complete!")


def main():
    """Main entry point"""
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    
    visualizer = BenchmarkVisualizer(results_file)
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()
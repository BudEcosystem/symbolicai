#!/usr/bin/env python3
"""
Generate visualizations from benchmark results with error handling
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load results
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['rps_benchmarks'])

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

print("Generating visualizations...")

# 1. Success rate vs RPS
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for expr_type in df['test_name'].unique():
        expr_data = df[df['test_name'] == expr_type]
        expr_data = expr_data.sort_values('rps')
        
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
    ax.set_ylim(95, 101)
    
    plt.tight_layout()
    plt.savefig('success_rate_vs_rps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Success rate chart created")
except Exception as e:
    print(f"✗ Error creating success rate chart: {e}")

# 2. Average latency comparison
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by expression type and calculate mean latency
    latency_by_type = df.groupby('test_name')['avg_latency_ms'].mean().sort_values()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(range(len(latency_by_type)), latency_by_type.values, color=colors)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, latency_by_type.values)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
               f'{value:.3f}', ha='center', va='bottom')
    
    ax.set_xticks(range(len(latency_by_type)))
    ax.set_xticklabels([name.replace('_', ' ').title() for name in latency_by_type.index], 
                      rotation=45, ha='right')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Average Latency by Expression Type')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Latency comparison chart created")
except Exception as e:
    print(f"✗ Error creating latency comparison: {e}")

# 3. Throughput analysis
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for expr_type in df['test_name'].unique():
        expr_data = df[df['test_name'] == expr_type]
        expr_data = expr_data.sort_values('rps')
        
        # Calculate actual throughput (requests/second)
        throughput = 1000 / expr_data['avg_latency_ms']
        
        ax.plot(expr_data['rps'], throughput, 'o-', 
               label=expr_type.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Target RPS')
    ax.set_ylabel('Achievable Throughput (ops/sec)')
    ax.set_title('Achievable Throughput vs Target RPS')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('throughput_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Throughput analysis chart created")
except Exception as e:
    print(f"✗ Error creating throughput analysis: {e}")

# 4. P95 Latency heatmap
try:
    pivot_data = df.pivot_table(
        values='p95_latency_ms',
        index='test_name',
        columns='rps'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in pivot_data.index])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P95 Latency (ms)')
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax.text(j, i, f'{pivot_data.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('P95 Latency Heatmap: Expression Type vs RPS')
    ax.set_xlabel('Requests Per Second (RPS)')
    ax.set_ylabel('Expression Type')
    
    plt.tight_layout()
    plt.savefig('p95_latency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ P95 latency heatmap created")
except Exception as e:
    print(f"✗ Error creating heatmap: {e}")

# 5. Performance summary
try:
    # Create summary statistics
    summary_stats = df.groupby('test_name').agg({
        'avg_latency_ms': ['mean', 'min', 'max'],
        'p95_latency_ms': 'mean',
        'p99_latency_ms': 'mean',
        'successful_matches': 'sum',
        'total_requests': 'sum'
    })
    
    # Calculate success rate
    summary_stats['success_rate'] = (summary_stats[('successful_matches', 'sum')] / 
                                     summary_stats[('total_requests', 'sum')] * 100)
    
    # Save summary
    summary_text = "PERFORMANCE SUMMARY\n"
    summary_text += "=" * 60 + "\n\n"
    
    for expr_type in summary_stats.index:
        summary_text += f"{expr_type.upper()}:\n"
        summary_text += f"  Average latency: {summary_stats.loc[expr_type, ('avg_latency_ms', 'mean')]:.3f} ms\n"
        summary_text += f"  Min latency: {summary_stats.loc[expr_type, ('avg_latency_ms', 'min')]:.3f} ms\n"
        summary_text += f"  Max latency: {summary_stats.loc[expr_type, ('avg_latency_ms', 'max')]:.3f} ms\n"
        summary_text += f"  Avg P95 latency: {summary_stats.loc[expr_type, ('p95_latency_ms', 'mean')]:.3f} ms\n"
        summary_text += f"  Success rate: {summary_stats.loc[expr_type, 'success_rate']:.1f}%\n"
        summary_text += f"  Max throughput: {1000/summary_stats.loc[expr_type, ('avg_latency_ms', 'min')]:.0f} ops/sec\n"
        summary_text += "\n"
    
    # System initialization stats
    summary_text += "\nSYSTEM INITIALIZATION:\n"
    for key, value in data['model_loading'].items():
        summary_text += f"  {key}: {value:.2f} ms\n"
    
    # Cache performance
    cache = data['cache_performance']
    summary_text += f"\nCACHE PERFORMANCE:\n"
    summary_text += f"  Cache speedup: {cache['cache_speedup']:.1f}x\n"
    
    with open('performance_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print("✓ Performance summary created")
    print("\n" + summary_text)
    
except Exception as e:
    print(f"✗ Error creating summary: {e}")

print("\nVisualization complete! Generated files:")
print("- latency_vs_rps.png")
print("- success_rate_vs_rps.png") 
print("- latency_comparison.png")
print("- throughput_analysis.png")
print("- p95_latency_heatmap.png")
print("- performance_summary.txt")
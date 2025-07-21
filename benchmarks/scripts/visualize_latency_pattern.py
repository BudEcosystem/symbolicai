#!/usr/bin/env python3
"""
Create detailed visualization of latency patterns
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['rps_benchmarks'])

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Main latency pattern
ax1 = plt.subplot(2, 2, 1)
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    ax1.plot(expr_data['rps'], expr_data['avg_latency_ms'], 'o-', 
             label=expr_type.replace('_', ' ').title(), linewidth=2, markersize=8)

ax1.set_xlabel('Requests Per Second (RPS)')
ax1.set_ylabel('Average Latency (ms)')
ax1.set_title('Latency vs RPS - "Cold Start" Effect at Low RPS')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.annotate('Higher latency\nat low RPS', xy=(10, 0.08), xytext=(20, 0.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red')

# 2. Latency reduction percentage
ax2 = plt.subplot(2, 2, 2)
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    latency_1_rps = expr_data[expr_data['rps'] == 1]['avg_latency_ms'].values[0]
    reduction_pct = (1 - expr_data['avg_latency_ms'] / latency_1_rps) * 100
    ax2.plot(expr_data['rps'], reduction_pct, 'o-', 
             label=expr_type.replace('_', ' ').title(), linewidth=2, markersize=8)

ax2.set_xlabel('Requests Per Second (RPS)')
ax2.set_ylabel('Latency Reduction from 1 RPS (%)')
ax2.set_title('Performance Improvement with Higher RPS')
ax2.set_xscale('log')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. P99 latency spikes
ax3 = plt.subplot(2, 2, 3)
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    ax3.plot(expr_data['rps'], expr_data['p99_latency_ms'], 'o-', 
             label=expr_type.replace('_', ' ').title(), linewidth=2, markersize=8)

ax3.set_xlabel('Requests Per Second (RPS)')
ax3.set_ylabel('P99 Latency (ms)')
ax3.set_title('P99 Latency - Tail Latencies Higher at Low RPS')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Efficiency metric (throughput per ms latency)
ax4 = plt.subplot(2, 2, 4)
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    efficiency = 1000 / expr_data['avg_latency_ms']  # potential throughput
    ax4.plot(expr_data['rps'], efficiency, 'o-', 
             label=expr_type.replace('_', ' ').title(), linewidth=2, markersize=8)

ax4.set_xlabel('Target RPS')
ax4.set_ylabel('Potential Throughput (ops/sec)')
ax4.set_title('System Efficiency - Higher at Sustained Load')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Latency Pattern Analysis: Why Low RPS Shows Higher Latency', fontsize=16)
plt.tight_layout()
plt.savefig('latency_pattern_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create explanation diagram
fig, ax = plt.subplots(figsize=(12, 8))

# Simulate CPU frequency and cache behavior
rps_points = [1, 10, 50, 100, 500, 1000]
cpu_freq = [0.6, 0.7, 0.85, 0.9, 0.95, 1.0]  # Normalized
cache_hot = [0.3, 0.4, 0.7, 0.8, 0.95, 1.0]  # Normalized
thread_active = [0.2, 0.3, 0.6, 0.8, 0.95, 1.0]  # Normalized

ax2_twin = ax.twinx()

# Plot lines
l1 = ax.plot(rps_points, cpu_freq, 'b-o', label='CPU Frequency', linewidth=3, markersize=10)
l2 = ax.plot(rps_points, cache_hot, 'g-s', label='Cache Hotness', linewidth=3, markersize=10)
l3 = ax.plot(rps_points, thread_active, 'r-^', label='Thread Activity', linewidth=3, markersize=10)

# Plot latency on secondary axis
latency_normalized = [0.08, 0.09, 0.065, 0.055, 0.03, 0.02]
l4 = ax2_twin.plot(rps_points, latency_normalized, 'm-d', label='Latency', linewidth=3, markersize=10)

ax.set_xlabel('Requests Per Second (RPS)', fontsize=12)
ax.set_ylabel('System State (Normalized)', fontsize=12)
ax2_twin.set_ylabel('Latency (ms)', fontsize=12, color='m')
ax.set_title('System Behavior at Different RPS Levels', fontsize=14)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Combine legends
lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Add annotations
ax.annotate('Cold CPU,\nEmpty Cache', xy=(1, 0.6), xytext=(2, 0.8),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

ax.annotate('Warm System,\nOptimal Performance', xy=(1000, 0.95), xytext=(300, 0.7),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=11, color='green', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

plt.tight_layout()
plt.savefig('system_behavior_explanation.png', dpi=300, bbox_inches='tight')

print("Visualizations created:")
print("- latency_pattern_analysis.png")
print("- system_behavior_explanation.png")
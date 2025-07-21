#!/usr/bin/env python3
"""
Analyze why latency is higher at low RPS
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['rps_benchmarks'])

# Analyze latency patterns
print("LATENCY ANALYSIS: Why Higher Latency at Low RPS?")
print("=" * 60)

# Show average latency by RPS for each expression type
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    print(f"\n{expr_type.upper()}:")
    print("RPS  | Avg Latency | P95 Latency | P99 Latency")
    print("-" * 50)
    for _, row in expr_data.iterrows():
        print(f"{row['rps']:4d} | {row['avg_latency_ms']:11.3f} | {row['p95_latency_ms']:11.3f} | {row['p99_latency_ms']:11.3f}")

# Visualize the pattern
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Average latency vs RPS
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    ax1.plot(expr_data['rps'], expr_data['avg_latency_ms'], 'o-', 
             label=expr_type.replace('_', ' ').title(), linewidth=2, markersize=8)

ax1.set_xlabel('Requests Per Second (RPS)')
ax1.set_ylabel('Average Latency (ms)')
ax1.set_title('Average Latency vs RPS - Notice Higher Latency at Low RPS')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Latency ratio (compared to 1000 RPS baseline)
for expr_type in df['test_name'].unique():
    expr_data = df[df['test_name'] == expr_type].sort_values('rps')
    baseline = expr_data[expr_data['rps'] == 1000]['avg_latency_ms'].values[0]
    ratios = expr_data['avg_latency_ms'] / baseline
    ax2.plot(expr_data['rps'], ratios, 'o-', 
             label=expr_type.replace('_', ' ').title(), linewidth=2, markersize=8)

ax2.set_xlabel('Requests Per Second (RPS)')
ax2.set_ylabel('Latency Ratio (compared to 1000 RPS)')
ax2.set_title('Latency Overhead at Different RPS Levels')
ax2.set_xscale('log')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latency_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n\nPOSSIBLE REASONS FOR HIGHER LATENCY AT LOW RPS:")
print("=" * 60)
print("""
1. THREAD POOL COLD START & CONTEXT SWITCHING
   - At low RPS (1-10), threads in the pool may go idle between requests
   - Thread wake-up and context switching add overhead
   - At high RPS, threads stay "warm" and active

2. CPU FREQUENCY SCALING (Power Management)
   - Modern CPUs reduce frequency when idle to save power
   - Low RPS = more idle time = CPU scales down
   - Each request must wait for CPU to scale back up
   - High RPS keeps CPU at high frequency constantly

3. CACHE EFFECTS
   - CPU caches (L1/L2/L3) may be evicted during idle periods
   - Memory pages may be swapped out
   - At high RPS, everything stays in cache

4. PYTHON GC AND JIT OPTIMIZATION
   - Python's garbage collector runs during idle time
   - JIT optimizations in Model2Vec may degrade without constant use
   - High RPS keeps optimizations active

5. OS SCHEDULING
   - At low RPS, OS may deprioritize the process
   - Network stack and I/O buffers may be flushed
   - High RPS keeps process priority high

6. MEASUREMENT OVERHEAD
   - At low RPS, timing overhead is a larger percentage
   - Thread synchronization overhead more noticeable
   - High RPS amortizes these costs
""")

# Calculate specific numbers
low_rps_avg = df[df['rps'] <= 10]['avg_latency_ms'].mean()
high_rps_avg = df[df['rps'] >= 500]['avg_latency_ms'].mean()
overhead_pct = ((low_rps_avg - high_rps_avg) / high_rps_avg) * 100

print(f"\nQUANTITATIVE ANALYSIS:")
print(f"Average latency at 1-10 RPS: {low_rps_avg:.3f} ms")
print(f"Average latency at 500-1000 RPS: {high_rps_avg:.3f} ms")
print(f"Overhead at low RPS: {overhead_pct:.1f}%")

print("\nRECOMMENDATIONS:")
print("- For production use, maintain steady request flow")
print("- Use connection pooling and keep-alive")
print("- Consider request batching for low-volume scenarios")
print("- Implement warm-up period before benchmarking")
print("- Use CPU governor 'performance' mode for consistent results")
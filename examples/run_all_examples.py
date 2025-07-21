#!/usr/bin/env python3
"""
Run all examples to ensure everything works
"""

import sys
import os
import subprocess
import glob

# Get the examples directory
examples_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(examples_dir)

# Find all example files
example_files = glob.glob(os.path.join(examples_dir, "example*.py"))
example_files.extend(glob.glob(os.path.join(examples_dir, "test_*.py")))

# Exclude some files
exclude = ['run_all_examples.py', 'debug_failing_tests.py', 'debug_expression_match.py']
example_files = [f for f in example_files if os.path.basename(f) not in exclude]

print(f"Found {len(example_files)} example files to run\n")

success_count = 0
failed_count = 0
failed_files = []

for example_file in sorted(example_files):
    filename = os.path.basename(example_file)
    print(f"{'='*60}")
    print(f"Running: {filename}")
    print(f"{'='*60}")
    
    try:
        # Run the example
        result = subprocess.run(
            [sys.executable, example_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✓ SUCCESS")
            success_count += 1
            # Show first few lines of output
            output_lines = result.stdout.strip().split('\n')
            if output_lines:
                print("\nOutput preview:")
                for line in output_lines[:5]:
                    print(f"  {line}")
                if len(output_lines) > 5:
                    print(f"  ... ({len(output_lines) - 5} more lines)")
        else:
            print("✗ FAILED")
            failed_count += 1
            failed_files.append(filename)
            print(f"\nError: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT (30s)")
        failed_count += 1
        failed_files.append(filename)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        failed_count += 1
        failed_files.append(filename)
    
    print()

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: {success_count} passed, {failed_count} failed")
print(f"{'='*60}")

if failed_files:
    print("\nFailed examples:")
    for f in failed_files:
        print(f"  - {f}")
else:
    print("\n✅ All examples ran successfully!")

sys.exit(0 if failed_count == 0 else 1)
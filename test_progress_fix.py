#!/usr/bin/env python3
"""
Test script to verify progress bar fix for sus_failure_probability.py
"""
import numpy as np
import multiprocessing
from numba import jit
from tqdm import tqdm
import time

# Import the functions from the main script
import sys
sys.path.append('/root/Case_Study_Log')
from sus_failure_probability import run_subset_simulation

def test_progress_bars():
    """Test the progress bar display with a small number of simulations."""
    print("Testing progress bar fix...")
    print(f"System has {multiprocessing.cpu_count()} CPU cores")
    
    # Load a small subset of data for testing
    try:
        res_no_adpt = np.load("/root/Case_Study_Log/res_no_adpt.npy", allow_pickle=True).item()
        print("Successfully loaded test data")
    except:
        print("Error loading test data")
        return
    
    # Set up test parameters
    SHOW_PROGRESS = True
    n_test_runs = 4  # Small number for testing
    k_test = 1.0     # Single k value for testing
    
    print(f"\nRunning {n_test_runs} test simulations...")
    print("You should see progress bars P00, P01, P02, P03 (or similar)")
    
    # Prepare test arguments
    test_args = []
    for j in range(n_test_runs):
        # Use smaller N for faster testing
        test_args.append((100, 0.1, res_no_adpt['u'], res_no_adpt['time_grid'], k_test, np.random.randint(0, 100000)))
    
    start_time = time.time()
    
    # Run test simulations
    with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
        results = pool.map(run_subset_simulation, test_args)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nTest completed in {elapsed_time:.1f} seconds")
    print(f"Results: {[f'{r:.6f}' for r in results]}")
    print("If you saw progress bars labeled P00, P01, etc., the fix is working!")

if __name__ == "__main__":
    # Temporarily set the global variable for testing
    import sus_failure_probability
    sus_failure_probability.SHOW_PROGRESS = True
    
    test_progress_bars()

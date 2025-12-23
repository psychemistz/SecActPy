#!/usr/bin/env python3
"""
Step-by-step comparison of Python GSL RNG with R's GSL.

Run this script and compare output with R's GSL output.

Usage:
    python tests/test_gsl_debug.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy.rng import GSLRNG, MT19937Pure


def main():
    print("=" * 70)
    print("Python GSL MT19937 - Step by Step Debug")
    print("=" * 70)
    
    # ==========================================================================
    # 1. Raw MT19937 output (first 10 values with seed=0)
    # ==========================================================================
    print("\n[1] First 10 raw MT19937 values (seed=0):")
    mt = MT19937Pure(0)
    for i in range(10):
        val = mt.genrand_int32()
        print(f"    {i}: {val}")
    
    # ==========================================================================
    # 2. uniform_int output 
    # ==========================================================================
    print("\n[2] uniform_int(n) for various n (seed=0):")
    rng = GSLRNG(0)
    
    # Test with n=10
    print("    uniform_int(10), first 10 values:")
    vals = [rng.uniform_int(10) for _ in range(10)]
    print(f"    {vals}")
    
    # Reset and test with n=7720 (actual data size)
    rng.reset(0)
    print("\n    uniform_int(7720), first 10 values:")
    vals = [rng.uniform_int(7720) for _ in range(10)]
    print(f"    {vals}")
    
    # ==========================================================================
    # 3. Single shuffle of [0..9]
    # ==========================================================================
    print("\n[3] Fisher-Yates shuffle of [0,1,2,...,9] (seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(10, dtype=np.int32)
    print(f"    Before: {arr.tolist()}")
    rng.shuffle_inplace(arr)
    print(f"    After:  {arr.tolist()}")
    
    # ==========================================================================
    # 4. Cumulative shuffles (as used in permutation table)
    # ==========================================================================
    print("\n[4] Cumulative shuffles of [0..9], 5 permutations (seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(10, dtype=np.int32)
    for i in range(5):
        rng.shuffle_inplace(arr)
        print(f"    Perm {i}: {arr.tolist()}")
    
    # ==========================================================================
    # 5. Permutation table for actual data size
    # ==========================================================================
    print("\n[5] Permutation table (n=7720, n_perm=3, seed=0):")
    rng = GSLRNG(0)
    arr = np.arange(7720, dtype=np.int32)
    
    for perm_idx in range(3):
        rng.shuffle_inplace(arr)
        print(f"    Perm {perm_idx} first 10: {arr[:10].tolist()}")
        print(f"    Perm {perm_idx} last 10:  {arr[-10:].tolist()}")
    
    # ==========================================================================
    # Print R code for comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("R CODE TO COMPARE (run in R with RidgeR loaded)")
    print("=" * 70)
    
    r_code = '''
# Option 1: If RidgeR has a debug function
# RidgeR::debug_gsl_permutations(n=10, n_perm=5, seed=0)

# Option 2: Add this to RidgeR C code and recompile:
/*
#include <gsl/gsl_rng.h>
#include <stdio.h>

void debug_gsl_output(void) {
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 0);
    
    // [1] Raw MT19937 values
    printf("[1] First 10 raw MT19937 values (seed=0):\\n");
    for (int i = 0; i < 10; i++) {
        printf("    %d: %lu\\n", i, gsl_rng_get(rng));
    }
    
    // [2] uniform_int(10)
    gsl_rng_set(rng, 0);
    printf("\\n[2] uniform_int(10), first 10 values:\\n    ");
    for (int i = 0; i < 10; i++) {
        printf("%lu ", gsl_rng_uniform_int(rng, 10));
    }
    printf("\\n");
    
    // [3] Shuffle [0..9]
    gsl_rng_set(rng, 0);
    int arr[10] = {0,1,2,3,4,5,6,7,8,9};
    
    printf("\\n[3] Fisher-Yates shuffle of [0..9] (seed=0):\\n");
    printf("    Before: ");
    for (int i = 0; i < 10; i++) printf("%d ", arr[i]);
    printf("\\n");
    
    // Fisher-Yates shuffle (matching your shuffle_array_gsl)
    for (int i = 0; i < 9; i++) {
        int j = i + (int)gsl_rng_uniform_int(rng, (unsigned long)(10 - i));
        int tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
    
    printf("    After:  ");
    for (int i = 0; i < 10; i++) printf("%d ", arr[i]);
    printf("\\n");
    
    // [4] Cumulative shuffles
    gsl_rng_set(rng, 0);
    int arr2[10] = {0,1,2,3,4,5,6,7,8,9};
    printf("\\n[4] Cumulative shuffles of [0..9], 5 permutations (seed=0):\\n");
    
    for (int perm = 0; perm < 5; perm++) {
        for (int i = 0; i < 9; i++) {
            int j = i + (int)gsl_rng_uniform_int(rng, (unsigned long)(10 - i));
            int tmp = arr2[j];
            arr2[j] = arr2[i];
            arr2[i] = tmp;
        }
        printf("    Perm %d: ", perm);
        for (int i = 0; i < 10; i++) printf("%d ", arr2[i]);
        printf("\\n");
    }
    
    gsl_rng_free(rng);
}
*/

# Option 3: Create a simple R wrapper that calls GSL
# (requires Rcpp with GSL)
'''
    print(r_code)
    print("=" * 70)


if __name__ == "__main__":
    main()

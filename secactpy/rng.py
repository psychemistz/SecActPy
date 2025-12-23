"""
GSL-compatible Mersenne Twister RNG for R/RidgeR reproducibility.

This module provides an RNG that produces identical permutation sequences
to GSL's MT19937 as used in RidgeR. This is essential for reproducing
exact p-values and z-scores from the R package.

RidgeR Algorithm (from C code):
-------------------------------
1. Initialize MT19937 with seed (default: 0)
2. Initialize array [0, 1, 2, ..., n-1]
3. For each permutation:
   a. Fisher-Yates shuffle the array (cumulative - each shuffle starts from previous state)
   b. Copy the shuffled array to permutation table

GSL's Fisher-Yates shuffle:
---------------------------
    for i in range(n-1):
        j = i + gsl_rng_uniform_int(rng, n - i)
        swap(array[i], array[j])

Key insight: RidgeR uses CUMULATIVE shuffling - each permutation is a
shuffle of the previous permutation, not a fresh shuffle of 0..n-1.

Usage:
------
    >>> from secactpy.rng import GSLRNG, generate_permutation_table
    >>> 
    >>> # Generate permutation table matching RidgeR
    >>> rng = GSLRNG(seed=0)
    >>> table = rng.permutation_table(n=1000, n_perm=1000)
    >>> 
    >>> # Or use convenience function
    >>> table = generate_permutation_table(n=1000, n_perm=1000, seed=0)

References:
-----------
- GSL: https://www.gnu.org/software/gsl/doc/html/rng.html
- MT19937: https://en.wikipedia.org/wiki/Mersenne_Twister
- RidgeR source: generate_permutation_table() in ridger.c
"""

import numpy as np
from typing import Optional, Tuple

__all__ = ['GSLRNG', 'generate_permutation_table']


# =============================================================================
# MT19937 Constants (from reference implementation)
# =============================================================================

MT_N = 624                    # State size
MT_M = 397                    # Middle word
MT_MATRIX_A = 0x9908b0df      # Constant vector a
MT_UPPER_MASK = 0x80000000    # Most significant bit  
MT_LOWER_MASK = 0x7fffffff    # Least significant 31 bits
MT_TEMPERING_MASK_B = 0x9d2c5680
MT_TEMPERING_MASK_C = 0xefc60000
MT19937_MAX = 0xFFFFFFFF      # 2^32 - 1


# =============================================================================
# MT19937 Implementation - Pure Python (for correctness)
# =============================================================================

class MT19937Pure:
    """
    Pure Python MT19937 matching the reference implementation exactly.
    
    This matches the original Matsumoto & Nishimura implementation and
    GSL's gsl_rng_mt19937 exactly.
    """
    
    __slots__ = ('mt', 'mti')
    
    def __init__(self, seed: int = 0):
        """Initialize with seed using the reference algorithm."""
        self.mt = [0] * MT_N
        self.mti = MT_N + 1
        self._init_genrand(seed)
    
    def _init_genrand(self, seed: int) -> None:
        """Initialize with a single seed (reference init_genrand)."""
        self.mt[0] = seed & MT19937_MAX
        for i in range(1, MT_N):
            self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & MT19937_MAX
        self.mti = MT_N
    
    def genrand_int32(self) -> int:
        """Generate a random 32-bit unsigned integer."""
        mag01 = (0, MT_MATRIX_A)
        
        # Generate MT_N words at one time
        if self.mti >= MT_N:
            mt = self.mt  # Local reference for speed
            
            for kk in range(MT_N - MT_M):
                y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK)
                mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1]
            
            for kk in range(MT_N - MT_M, MT_N - 1):
                y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK)
                mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1]
            
            y = (mt[MT_N - 1] & MT_UPPER_MASK) | (mt[0] & MT_LOWER_MASK)
            mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1]
            
            self.mti = 0
        
        y = self.mt[self.mti]
        self.mti += 1
        
        # Tempering
        y ^= (y >> 11)
        y ^= (y << 7) & MT_TEMPERING_MASK_B
        y ^= (y << 15) & MT_TEMPERING_MASK_C
        y ^= (y >> 18)
        
        return y


# =============================================================================
# GSL-Compatible RNG
# =============================================================================

class GSLRNG:
    """
    GSL-compatible Mersenne Twister RNG.
    
    Produces identical sequences to GSL's gsl_rng_mt19937 when using
    the same seed. Uses the exact GSL algorithm for bounded integer generation.
    
    Parameters
    ----------
    seed : int, optional
        Random seed. Default is 0 (matching RidgeR default).
    
    Examples
    --------
    >>> rng = GSLRNG(seed=0)
    >>> perm_table = rng.permutation_table(n=100, n_perm=1000)
    >>> perm_table.shape
    (1000, 100)
    
    >>> # Verify it's a valid permutation
    >>> np.all(np.sort(perm_table[0]) == np.arange(100))
    True
    
    Notes
    -----
    This class matches GSL's behavior exactly, including:
    - Seed initialization via MT19937 reference algorithm
    - Bounded integer generation via GSL's rejection sampling
    - Fisher-Yates shuffle with cumulative state
    """
    
    __slots__ = ('_seed', '_mt')
    
    def __init__(self, seed: int = 0):
        """
        Initialize RNG with seed.
        
        Parameters
        ----------
        seed : int
            Seed value. Use 0 for RidgeR compatibility.
            
        Notes
        -----
        GSL treats seed=0 specially by using 4357 as the default seed.
        This matches that behavior exactly.
        """
        self._seed = seed
        # GSL uses 4357 as default when seed=0
        actual_seed = 4357 if seed == 0 else seed
        self._mt = MT19937Pure(actual_seed)
    
    def _get_raw(self) -> int:
        """Get next raw 32-bit unsigned integer."""
        return self._mt.genrand_int32()
    
    def uniform_int(self, n: int) -> int:
        """
        Generate random integer in [0, n-1] using GSL's algorithm.
        
        This implements GSL's gsl_rng_uniform_int exactly:
        - Uses rejection sampling to avoid modulo bias
        - Identical to GSL when given same MT19937 state
        
        Parameters
        ----------
        n : int
            Upper bound (exclusive). Must be > 0.
        
        Returns
        -------
        int
            Random integer in [0, n-1]
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n > MT19937_MAX:
            raise ValueError(f"n must be <= {MT19937_MAX}")
        
        # GSL algorithm: rejection sampling with scaling
        scale = MT19937_MAX // n
        
        while True:
            k = self._get_raw() // scale
            if k < n:
                return k
    
    def shuffle_inplace(self, arr: np.ndarray) -> None:
        """
        Fisher-Yates shuffle matching GSL implementation.
        
        Modifies array in-place using the exact same algorithm as GSL.
        
        Parameters
        ----------
        arr : np.ndarray
            Array to shuffle. Modified in-place.
        """
        n = len(arr)
        for i in range(n - 1):
            j = i + self.uniform_int(n - i)
            arr[i], arr[j] = arr[j], arr[i]
    
    def permutation_table(self, n: int, n_perm: int) -> np.ndarray:
        """
        Generate permutation table matching RidgeR exactly.
        
        IMPORTANT: Uses cumulative shuffling like RidgeR - each permutation
        is a shuffle of the previous state, not a fresh shuffle of 0..n-1.
        
        Parameters
        ----------
        n : int
            Number of elements to permute (e.g., number of genes).
        n_perm : int
            Number of permutations to generate.
        
        Returns
        -------
        np.ndarray, shape (n_perm, n), dtype int32
            Each row is a permutation of 0..n-1.
        """
        table = np.zeros((n_perm, n), dtype=np.int32)
        arr = np.arange(n, dtype=np.int32)
        
        for i in range(n_perm):
            self.shuffle_inplace(arr)
            table[i] = arr.copy()
        
        return table
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset RNG to initial state.
        
        Parameters
        ----------
        seed : int, optional
            New seed. If None, uses original seed.
        """
        if seed is not None:
            self._seed = seed
        # GSL uses 4357 as default when seed=0
        actual_seed = 4357 if self._seed == 0 else self._seed
        self._mt = MT19937Pure(actual_seed)


# =============================================================================
# Convenience Function
# =============================================================================

def generate_permutation_table(
    n: int,
    n_perm: int,
    seed: int = 0
) -> np.ndarray:
    """
    Generate permutation table matching RidgeR.
    
    Convenience function wrapping GSLRNG.permutation_table().
    
    Parameters
    ----------
    n : int
        Number of elements to permute.
    n_perm : int
        Number of permutations.
    seed : int, default=0
        Random seed. Use 0 for RidgeR compatibility.
    
    Returns
    -------
    np.ndarray, shape (n_perm, n)
        Permutation table.
    
    Examples
    --------
    >>> table = generate_permutation_table(100, 1000, seed=0)
    >>> table.shape
    (1000, 100)
    """
    rng = GSLRNG(seed)
    return rng.permutation_table(n, n_perm)


# =============================================================================
# Validation Functions
# =============================================================================

# Reference values for validation (from GSL with seed=0)
REFERENCE_MT19937_SEED0_FIRST10 = [
    2357136044, 2546248239, 3071714933, 3626093760, 2588848963,
    3684848379, 2340255427, 3638918503, 1819583497, 2678185683
]

REFERENCE_MT19937_SEED5489_FIRST5 = [
    3499211612, 581869302, 3890346734, 3586334585, 545404204
]


def validate_mt19937() -> Tuple[bool, str]:
    """
    Validate MT19937 implementation against reference values.
    
    Returns
    -------
    tuple
        (success: bool, message: str)
    """
    # Test with standard seed 5489
    mt = MT19937Pure(5489)
    vals = [mt.genrand_int32() for _ in range(5)]
    
    if vals != REFERENCE_MT19937_SEED5489_FIRST5:
        return False, f"MT19937(5489) mismatch: got {vals}, expected {REFERENCE_MT19937_SEED5489_FIRST5}"
    
    # Test with seed 0
    mt = MT19937Pure(0)
    vals = [mt.genrand_int32() for _ in range(10)]
    
    if vals != REFERENCE_MT19937_SEED0_FIRST10:
        return False, f"MT19937(0) mismatch: got {vals}, expected {REFERENCE_MT19937_SEED0_FIRST10}"
    
    return True, "MT19937 implementation validated successfully"


def validate_gslrng() -> Tuple[bool, str]:
    """
    Validate GSLRNG implementation.
    
    Returns
    -------
    tuple
        (success: bool, message: str)
    """
    # Test uniform_int bounds
    rng = GSLRNG(seed=0)
    n = 100
    
    for _ in range(1000):
        v = rng.uniform_int(n)
        if v < 0 or v >= n:
            return False, f"uniform_int({n}) out of bounds: {v}"
    
    # Validate permutation table structure
    rng.reset()
    table = rng.permutation_table(10, 5)
    
    for i, row in enumerate(table):
        if set(row) != set(range(10)):
            return False, f"Permutation {i} is not a valid permutation of 0..9"
    
    return True, "GSLRNG implementation validated successfully"


def generate_c_validation_code() -> str:
    """
    Generate C code to produce reference permutation table from GSL.
    
    Compile with: gcc -o gen_perm gen_perm.c -lgsl -lgslcblas
    """
    return '''
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>

void shuffle_array(gsl_rng *rng, int *array, int n) {
    for (int i = 0; i < n - 1; i++) {
        int j = i + (int)gsl_rng_uniform_int(rng, (unsigned long)(n - i));
        int tmp = array[j];
        array[j] = array[i];
        array[i] = tmp;
    }
}

int main() {
    int n = 10;
    int n_perm = 5;
    unsigned long seed = 0;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    
    int *array = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) array[i] = i;
    
    printf("REFERENCE_PERM_TABLE = np.array([\\n");
    for (int p = 0; p < n_perm; p++) {
        shuffle_array(rng, array, n);
        printf("    [");
        for (int i = 0; i < n; i++) {
            printf("%d", array[i]);
            if (i < n - 1) printf(", ");
        }
        printf("],  # perm %d\\n", p);
    }
    printf("], dtype=np.int32)\\n");
    
    gsl_rng_free(rng);
    free(array);
    return 0;
}
'''


# =============================================================================
# Main - Testing
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("SecActPy RNG Module - Validation and Testing")
    print("=" * 60)
    
    # 1. Validate MT19937
    print("\n1. Validating MT19937 implementation...")
    success, msg = validate_mt19937()
    print(f"   {'✓' if success else '✗'} {msg}")
    
    # 2. Validate GSLRNG
    print("\n2. Validating GSLRNG implementation...")
    success, msg = validate_gslrng()
    print(f"   {'✓' if success else '✗'} {msg}")
    
    # 3. Test permutation table generation
    print("\n3. Testing permutation table generation...")
    rng = GSLRNG(seed=0)
    table = rng.permutation_table(10, 5)
    print(f"   Permutation table (n=10, n_perm=5):")
    for i, row in enumerate(table):
        print(f"   perm[{i}] = {row.tolist()}")
    
    # 4. Performance benchmark
    print("\n4. Performance benchmark...")
    for n, n_perm in [(100, 100), (1000, 1000), (100, 10000)]:
        rng = GSLRNG(seed=0)
        start = time.time()
        _ = rng.permutation_table(n, n_perm)
        elapsed = time.time() - start
        print(f"   n={n:4d}, n_perm={n_perm:5d}: {elapsed:.3f}s")
    
    # 5. Show C validation code
    print("\n5. To generate reference data from GSL, compile and run:")
    print("   " + "-" * 50)
    print("   Save the following to gen_perm.c and run:")
    print("   gcc -o gen_perm gen_perm.c -lgsl -lgslcblas && ./gen_perm")
    print("   " + "-" * 50)
    
    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)

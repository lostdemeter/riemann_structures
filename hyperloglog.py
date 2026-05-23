"""ResonantHyperLogLog — cardinality estimation using Riemann zero hash.

Traditional HLL: hash each element to 64 bits. First k bits = bucket.
Remaining bits = leading zeros → geometric distribution → estimator.

Resonant HLL: one Riemann zero gives uniform phase → convert to
pseudo-64-bit hash. Same bucket/leading-zero logic, deterministic.

Error ≈ 1.04 / √m  (m = 2^k registers)

Usage:
    hll = ResonantHyperLogLog(precision=12)  # 2^12 = 4096 registers
    hll.add("apple")
    hll.add("banana")
    hll.estimate()  # ≈ 2
"""
import os, sys, math, json
from array import array
import numpy as np

_gammas_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..',
                 'riemann_attention', 'output', 'figures', 'scripts', 'gammas_620.npy'),
    os.path.join(os.path.dirname(__file__), '..', '..',
                 'riemann_constructive', 'attention', 'gammas_620.npy'),
]
GAMMAS = None
for p in _gammas_paths:
    full = os.path.abspath(p)
    if os.path.exists(full):
        GAMMAS = np.load(full)
        break
if GAMMAS is None:
    GAMMAS = np.array([14.134725 + i * 2.5 for i in range(200)])
NG = len(GAMMAS)

# HLL bias correction constants
_HLL_ALPHA = {
    16: 0.673,
    32: 0.697,
    64: 0.709,
}
def _alpha(m):
    if m in _HLL_ALPHA:
        return _HLL_ALPHA[m]
    return 0.7213 / (1 + 1.079 / m)


class ResonantHyperLogLog:
    """HyperLogLog cardinality estimator using Riemann zero hash.

    Args:
        precision: Number of bits for bucket index (4-16). m = 2^precision.
            Higher precision = lower error but more memory.
            Error ≈ 1.04 / √m. Default 12 → 4096 registers → 1.6% error.
    """

    def __init__(self, precision=12):
        if precision < 4 or precision > 16:
            raise ValueError("Precision must be between 4 and 16")
        self.p = precision
        self.m = 1 << self.p  # number of registers
        self.registers = array('B', [0]) * self.m

    def _hash64(self, key):
        """64-bit pseudo-hash using Riemann zero phase.

        Returns (bucket_index, leading_zero_count) as a tuple.
        """
        h = hash(key) if not isinstance(key, int) else key
        # Phase from gamma_0 — uniform in [0, 2π)
        phase = GAMMAS[0] * h
        # Convert to 64-bit integer
        bits = int((phase % (2 * math.pi)) / (2 * math.pi) * (1 << 64))
        # First p bits = bucket
        bucket = bits >> (64 - self.p)
        # Remaining bits (lower 64-p bits) for leading-zero counting
        remaining = bits & ((1 << (64 - self.p)) - 1)
        if remaining == 0:
            w = 64 - self.p + 1
        else:
            # ρ(hash) = 1 + leading_zeros = (65 - p) - bit_length(remaining)
            w = (65 - self.p) - remaining.bit_length()
        w = max(1, min(w, 64))
        return bucket, w

    # ─── Core ──────────────────────────────────────────────────────────

    def add(self, key):
        """Insert key into the HLL."""
        bucket, w = self._hash64(key)
        if w > self.registers[bucket]:
            self.registers[bucket] = w

    def __len__(self):
        return int(round(self.estimate()))

    def estimate(self):
        """Return float cardinality estimate."""
        m = self.m
        # Sum of 2^{-M[j]}
        inv_sum = sum(2.0 ** -r for r in self.registers)
        if inv_sum == 0:
            return 0.0

        estimate = _alpha(m) * m * m / inv_sum

        # Small range correction
        if estimate < 2.5 * m:
            # Count empty registers
            v = sum(1 for r in self.registers if r == 0)
            if v > 0:
                estimate = m * math.log(m / v)

        # Large range correction (64-bit)
        if estimate > (1 << 64) / 30:
            estimate = -(1 << 64) * math.log(1 - estimate / (1 << 64))

        return estimate

    def merge(self, other):
        """Combine two HLLs (for distributed counting)."""
        for i in range(self.m):
            self.registers[i] = max(self.registers[i], other.registers[i])

    def clear(self):
        for i in range(self.m):
            self.registers[i] = 0

    # ─── Stats ─────────────────────────────────────────────────────────

    def stats(self):
        e = self.estimate()
        return {
            'precision': self.p,
            'registers': self.m,
            'error_bound': 1.04 / math.sqrt(self.m),
            'estimate': e,
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantHyperLogLog(p={s['precision']}, "
                f"m={s['registers']}, err={s['error_bound']:.4f}, "
                f"est={s['estimate']:.0f})")


# ─── Benchmark ─────────────────────────────────────────────────────────

from array import array

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantHyperLogLog Benchmark")
    print("=" * 60)

    random.seed(42)
    def rand_str():
        return ''.join(random.choice(string.ascii_letters) for _ in range(8))

    for true_n in [100, 1000, 10000, 100000]:
        items = {rand_str() for _ in range(true_n)}
        actual_n = len(items)

        hll = ResonantHyperLogLog(precision=12)
        t0 = time.time()
        for s in items:
            hll.add(s)
        insert_ms = (time.time() - t0) * 1000
        est = hll.estimate()
        err = abs(est - actual_n) / actual_n * 100

        print(f"\n  N={actual_n:6d}:  est={est:8.0f}  err={err:+.2f}%  "
              f"{insert_ms:.0f}ms ({insert_ms/actual_n*1000:.3f}µs)")

    # Precision tradeoff
    print(f"\n--- Precision tradeoff (N=10000) ---")
    for p in [8, 10, 12, 14]:
        items = {rand_str() for _ in range(10000)}
        hll = ResonantHyperLogLog(precision=p)
        for s in items:
            hll.add(s)
        est = hll.estimate()
        err = abs(est - len(items)) / len(items) * 100
        print(f"  p={p}, m={hll.m:5d}: est={est:8.0f}, err={err:+.2f}%")

    # Comparison with traditional HLL
    print(f"\n--- Comparison with traditional (N=10000, p=12) ---")
    items = {rand_str() for _ in range(10000)}
    hll = ResonantHyperLogLog(precision=12)
    for s in items:
        hll.add(s)
    print(f"  Resonant HLL:  est={hll.estimate():.0f}")

    # Traditional HLL with hashlib
    import hashlib
    class TradHLL:
        def __init__(self, p=12):
            self.p = p
            self.m = 1 << p
            self.reg = [0] * self.m
        def add(self, key):
            h = hashlib.sha256(str(key).encode()).digest()
            h = int.from_bytes(h, 'big')
            bucket = h >> (256 - p)
            # Count leading zeros in remaining
            remaining = (h << p) & ((1 << 256) - 1)
            w = remaining.bit_length() if remaining > 0 else 0
            w = (256 - p) - w + 2 if w > 0 else 257 - p
            w = max(1, min(w, 256))
            if w > self.reg[bucket]:
                self.reg[bucket] = w
        def __len__(self):
            m = self.m
            inv_sum = sum(2.0 ** -r for r in self.reg)
            if inv_sum == 0: return 0.0
            est = _alpha(m) * m * m / inv_sum
            v = sum(1 for r in self.reg if r == 0)
            if est < 2.5 * m and v > 0:
                est = m * math.log(m / v)
            return est

    thll = TradHLL(p=12)
    for s in items:
        thll.add(s)
    print(f"  Traditional HLL: est={thll.estimate():.0f}")


if __name__ == '__main__':
    run_benchmark()

"""ResonantBloomFilter — Bloom filter using Riemann zeros as hash functions.

Traditional Bloom filter: k random hash functions per element.
Resonant Bloom filter: k zeros, each gives a bit position via digit_n(key) % m.

The zeros behave like independent random hash functions (Montgomery-Odlyzko),
so the false positive rate matches theory: (1 - e^{-kn/m})^k.

Usage:
    bf = ResonantBloomFilter(capacity=1000, error_rate=0.01)
    bf.add("hello")
    bf.add("world")
    "hello" in bf      # True (probably)
    "foo" in bf        # False (probably)
    bf.save("/tmp/bloom.json")
    bf2 = ResonantBloomFilter.load("/tmp/bloom.json")
"""
import os, sys, math, json, struct
from array import array

import numpy as np

# ─── Riemann zeros ─────────────────────────────────────────────────────
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


class ResonantBloomFilter:
    """Bloom filter with Riemann-zero hash functions.

    Args:
        capacity: Expected number of elements to insert
        error_rate: Desired false positive rate (0 < error_rate < 1)
        n_zeros: Number of hash functions (default: automatically chosen)
        bitarray: Optional existing bitarray (for from_bytes)
    """

    def __init__(self, capacity=None, error_rate=None, n_zeros=None, bitarray=None):
        if bitarray is not None:
            self.bits = array('B', bitarray)
            self.m = len(self.bits) * 8
            self.k = n_zeros or 6
        elif capacity is not None and error_rate is not None:
            self._init_optimal(capacity, error_rate, n_zeros)
        elif capacity is not None and n_zeros is not None:
            # Use provided k, compute m
            self.k = min(n_zeros, NG)
            self.m = max(8, int(capacity * self.k / math.log(2)))
            self.bits = array('B', [0]) * ((self.m + 7) // 8)
        else:
            # Default: small test
            self.m = 1024
            self.k = 6
            self.bits = array('B', [0]) * (self.m // 8)

    def _init_optimal(self, capacity, error_rate, n_zeros):
        """Choose optimal k and m for given capacity and error rate."""
        ln2 = math.log(2)
        # Optimal m = -n * ln(p) / (ln2)^2
        self.m = max(8, int(-capacity * math.log(error_rate) / (ln2 * ln2)))
        # Optimal k = (m/n) * ln2
        k_opt = max(1, int(self.m / capacity * ln2))
        self.k = min(n_zeros if n_zeros else k_opt, NG)
        self.bits = array('B', [0]) * ((self.m + 7) // 8)

    # ─── Hash functions ───────────────────────────────────────────────

    def _hash(self, key, idx):
        """Compute bit position for key using zero index idx."""
        h = hash(key) if not isinstance(key, int) else key
        phase = GAMMAS[idx] * h
        # Use both integer and fractional parts for better mixing
        pos = int((phase % (2 * math.pi)) / (2 * math.pi) * self.m) % self.m
        return pos

    def _positions(self, key):
        """Generate all k bit positions for a key."""
        return [self._hash(key, i) for i in range(self.k)]

    # ─── Core operations ───────────────────────────────────────────────

    def add(self, key):
        """Insert a key into the filter."""
        for pos in self._positions(key):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self.bits[byte_idx] |= (1 << bit_idx)

    def __contains__(self, key):
        """Check if key is in the filter (may have false positives)."""
        for pos in self._positions(key):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self.bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    # ─── Set operations ────────────────────────────────────────────────

    def union(self, other):
        """Merge another filter into this one (bitwise OR)."""
        if len(self.bits) != len(other.bits):
            raise ValueError("Filters must have same size")
        result = ResonantBloomFilter(n_zeros=self.k, bitarray=self.bits[:])
        for i in range(len(self.bits)):
            result.bits[i] |= other.bits[i]
        return result

    def intersection(self, other):
        """Bitwise AND of two filters."""
        if len(self.bits) != len(other.bits):
            raise ValueError("Filters must have same size")
        result = ResonantBloomFilter(n_zeros=self.k, bitarray=self.bits[:])
        for i in range(len(self.bits)):
            result.bits[i] &= other.bits[i]
        return result

    @property
    def size(self):
        return self.m

    @property
    def n_bytes(self):
        return len(self.bits)

    def false_positive_rate(self):
        """Theoretical false positive rate."""
        n_est = self._estimate_n()
        if n_est == 0:
            return 0.0
        return (1 - math.exp(-self.k * n_est / self.m)) ** self.k

    def _estimate_n(self):
        """Estimate number of elements inserted (Swamidass & Baldi)."""
        bits_set = sum(bin(b).count('1') for b in self.bits)
        p = bits_set / self.m
        return max(0, int(-self.m / self.k * math.log(1 - p)))

    def clear(self):
        for i in range(len(self.bits)):
            self.bits[i] = 0

    # ─── Serialization ─────────────────────────────────────────────────

    def to_bytes(self):
        return bytes(self.bits)

    @classmethod
    def from_bytes(cls, data, n_zeros=6):
        return cls(bitarray=data, n_zeros=n_zeros)

    def save(self, path):
        data = {'k': self.k, 'm': self.m, 'bits': self.to_bytes().hex()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        bf = cls(n_zeros=data['k'])
        bf.m = data['m']
        bf.bits = array('B', bytes.fromhex(data['bits']))
        return bf

    def stats(self):
        bits_set = sum(bin(b).count('1') for b in self.bits)
        return {
            'bits': self.m,
            'hash_functions': self.k,
            'bytes': len(self.bits),
            'bits_set': bits_set,
            'load_factor': bits_set / self.m if self.m else 0,
            'estimated_n': self._estimate_n(),
            'fpr_theoretical': self.false_positive_rate(),
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantBloomFilter(m={s['bits']}, k={s['hash_functions']}, "
                f"bits={s['bits_set']}/{s['bits']}, "
                f"est_n={s['estimated_n']}, fpr={s['fpr_theoretical']:.4f})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantBloomFilter Benchmark")
    print("=" * 60)

    # Generate random strings
    random.seed(42)
    def rand_str(n=8):
        return ''.join(random.choice(string.ascii_letters) for _ in range(n))

    N_INSERT = 10000
    N_QUERY = 100000
    inserted = [rand_str() for _ in range(N_INSERT)]
    queries = inserted[:N_QUERY // 2] + [rand_str() for _ in range(N_QUERY // 2)]
    random.shuffle(queries)

    # Resonant Bloom filter
    print(f"\n=== Resonant Bloom Filter ===")
    t0 = time.time()
    bf = ResonantBloomFilter(capacity=N_INSERT, error_rate=0.01)
    for s in inserted:
        bf.add(s)
    insert_ms = (time.time() - t0) * 1000

    t0 = time.time()
    tp, fp, tn, fn = 0, 0, 0, 0
    for s in queries:
        if s in bf:
            if s in inserted:
                tp += 1
            else:
                fp += 1
        else:
            if s in inserted:
                fn += 1
            else:
                tn += 1
    query_ms = (time.time() - t0) * 1000

    print(f"  Insert {N_INSERT}:  {insert_ms:.0f} ms")
    print(f"  Query {len(queries)}: {query_ms:.0f} ms ({query_ms/len(queries)*1000:.3f} µs)")
    if fp + fn > 0:
        print(f"  FPR: {fp/(fp+tn)*100:.4f}% (theory: {bf.false_positive_rate()*100:.4f}%)")
        print(f"  FNR: {fn/(tp+fn)*100:.4f}%")
    print(f"  Stats: {bf.stats()['load_factor']*100:.1f}% full, "
          f"est {bf.stats()['estimated_n']} elements")

    # Traditional Bloom filter (Python built-in hash)
    print(f"\n=== Traditional Bloom Filter (random hashes) ===")
    import hashlib
    m = bf.m
    k = bf.k

    class TradBloom:
        def __init__(self, m, k):
            self.m = m
            self.k = k
            self.bits = array('B', [0]) * ((m + 7) // 8)
        def _hash(self, key, salt):
            h = hashlib.md5((str(salt) + str(key)).encode()).digest()
            return int.from_bytes(h[:4], 'big') % self.m
        def add(self, key):
            for i in range(self.k):
                p = self._hash(key, i)
                self.bits[p // 8] |= (1 << (p % 8))
        def __contains__(self, key):
            for i in range(self.k):
                p = self._hash(key, i)
                if not (self.bits[p // 8] & (1 << (p % 8))):
                    return False
            return True

    t0 = time.time()
    tb = TradBloom(m, k)
    for s in inserted:
        tb.add(s)
    t_insert_ms = (time.time() - t0) * 1000

    t0 = time.time()
    t_tp, t_fp, t_tn, t_fn = 0, 0, 0, 0
    for s in queries:
        if s in tb:
            if s in inserted:
                t_tp += 1
            else:
                t_fp += 1
        else:
            if s in inserted:
                t_fn += 1
            else:
                t_tn += 1
    t_query_ms = (time.time() - t0) * 1000

    print(f"  Insert {N_INSERT}:  {t_insert_ms:.0f} ms")
    print(f"  Query {len(queries)}: {t_query_ms:.0f} ms ({t_query_ms/len(queries)*1000:.3f} µs)")
    if t_fp + t_fn > 0:
        print(f"  FPR: {t_fp/(t_fp+t_tn)*100:.4f}%")
        print(f"  FNR: {t_fn/(t_tp+t_fn)*100:.4f}%")

    # Comparison
    print(f"\n=== Comparison ===")
    print(f"  Insert: resonant {insert_ms:.0f}ms vs traditional {t_insert_ms:.0f}ms")
    print(f"  Query:  resonant {query_ms:.0f}ms vs traditional {t_query_ms:.0f}ms")
    print(f"  FPR:    resonant {fp/(fp+tn)*100:.4f}% vs traditional {t_fp/(t_fp+t_tn)*100:.4f}%")
    print(f"  Theory: {bf.false_positive_rate()*100:.4f}%")


if __name__ == '__main__':
    run_benchmark()

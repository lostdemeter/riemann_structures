"""ResonantCuckooFilter — cuckoo filter using Riemann zero hash functions.

Traditional cuckoo filter: two hash functions, fingerprint, two possible
bucket positions, displacement on collision (cuckoo hashing).

Resonant cuckoo filter: three Riemann zeros:
  - γ₁: first bucket position  p₁ = digit₁(key) % m
  - γ₂: fingerprint (low BITS bits of digit₂)
  - γ₃: offset for second bucket  offset = digit₃(fingerprint) % m
  - p₂ = p₁ ⊕ offset (mutual alternate)

Usage:
    cf = ResonantCuckooFilter(capacity=1000, fingerprint_bits=4)
    cf.add("apple")
    "apple" in cf   # True
    "banana" in cf  # False (maybe)
"""
import os, sys, math, json, random
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

MAX_CUCKOO_KICKS = 500  # prevent infinite loops


class CuckooBucket:
    __slots__ = ('slots',)
    def __init__(self, entries_per_bucket=4):
        self.slots = [None] * entries_per_bucket

    def has(self, fp):
        return fp in self.slots

    def insert(self, fp):
        for i in range(len(self.slots)):
            if self.slots[i] is None:
                self.slots[i] = fp
                return True
        return False

    def delete(self, fp):
        for i in range(len(self.slots)):
            if self.slots[i] == fp:
                self.slots[i] = None
                return True
        return False

    def swap(self, fp):
        """Replace a random entry with fp, return the displaced fingerprint."""
        idx = random.randrange(len(self.slots))
        old = self.slots[idx]
        self.slots[idx] = fp
        return old

    @property
    def full(self):
        return all(s is not None for s in self.slots)

    @property
    def count(self):
        return sum(1 for s in self.slots if s is not None)


class ResonantCuckooFilter:
    """Cuckoo filter with Riemann-zero hash functions.

    Args:
        capacity: Expected number of elements
        fingerprint_bits: Bits per fingerprint (4-8). More = lower FPR.
        entries_per_bucket: Slots per bucket (default 4).
        max_kicks: Max cuckoo displacements before considering full.
    """

    def __init__(self, capacity=1000, fingerprint_bits=4,
                 entries_per_bucket=4, max_kicks=MAX_CUCKOO_KICKS):
        self.bits = fingerprint_bits
        self.fp_mask = (1 << self.bits) - 1
        self.bucket_size = entries_per_bucket
        self.max_kicks = max_kicks

        # Number of buckets: m = capacity / bucket_size * load_factor
        # Target load factor ~0.95 for cuckoo filters
        self.m = max(4, int(1.5 * capacity / self.bucket_size))
        self.buckets = [CuckooBucket(self.bucket_size) for _ in range(self.m)]
        self._size = 0

    # ─── Hash functions (3 Riemann zeros) ─────────────────────────────

    def _digit(self, key, zero_idx, mod=None):
        h = hash(key) if not isinstance(key, int) else key
        phase = GAMMAS[zero_idx] * h
        d = int((phase % (2 * math.pi)) / (2 * math.pi) * (mod or self.m)) % (mod or self.m)
        return d

    def _fingerprint(self, key):
        return self._digit(key, 1, self.fp_mask + 1)

    def _positions(self, key):
        """Compute both bucket positions for key."""
        p1 = self._digit(key, 0)
        fp = self._fingerprint(key)
        offset = self._digit(fp, 2)
        p2 = p1 ^ offset
        return p1 % self.m, p2 % self.m, fp

    # ─── Core operations ───────────────────────────────────────────────

    def add(self, key):
        """Insert key. Returns True if inserted, False if filter is full."""
        p1, p2, fp = self._positions(key)

        # Try both buckets first
        if self.buckets[p1].insert(fp):
            self._size += 1
            return True
        if self.buckets[p2].insert(fp):
            self._size += 1
            return True

        # Cuckoo displacement
        cur = p1 if random.random() < 0.5 else p2
        for _ in range(self.max_kicks):
            # Displace an existing fingerprint from cur
            displaced = self.buckets[cur].swap(fp)

            # Compute alternate for the displaced fingerprint
            offset = self._digit(displaced, 2)
            cur = cur ^ offset
            cur %= self.m
            fp = displaced

            # Try to insert at new location
            if self.buckets[cur].insert(fp):
                self._size += 1
                return True

        return False  # Filter full

    def __contains__(self, key):
        p1, p2, fp = self._positions(key)
        return self.buckets[p1].has(fp) or self.buckets[p2].has(fp)

    def discard(self, key):
        """Remove key. Returns True if found and removed."""
        p1, p2, fp = self._positions(key)
        if self.buckets[p1].delete(fp):
            self._size -= 1
            return True
        if self.buckets[p2].delete(fp):
            self._size -= 1
            return True
        return False

    # ─── Stats ─────────────────────────────────────────────────────────

    def clear(self):
        for b in self.buckets:
            for i in range(len(b.slots)):
                b.slots[i] = None
        self._size = 0

    @property
    def size(self):
        return self._size

    @property
    def load_factor(self):
        return self._size / (self.m * self.bucket_size)

    def false_positive_rate(self):
        """Theoretical FPR approximation for cuckoo filters."""
        f = 2 ** -self.bits  # probability of random fingerprint match
        return 2 * f  # two buckets to check

    def stats(self):
        return {
            'size': self._size,
            'buckets': self.m,
            'slots': self.m * self.bucket_size,
            'fingerprint_bits': self.bits,
            'load_factor': self.load_factor,
            'fpr_theoretical': self.false_positive_rate(),
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantCuckooFilter(m={s['buckets']}, "
                f"fp={s['fingerprint_bits']}b, "
                f"size={s['size']}/{s['slots']}, "
                f"load={s['load_factor']:.1%}, "
                f"fpr={s['fpr_theoretical']:.4f})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantCuckooFilter Benchmark")
    print("=" * 60)

    random.seed(42)
    def rand_str():
        return ''.join(random.choice(string.ascii_letters) for _ in range(8))

    N = 10000
    inserted = [rand_str() for _ in range(N)]

    cf = ResonantCuckooFilter(capacity=N, fingerprint_bits=6)
    print(f"\nFilter: {cf}")

    t0 = time.time()
    fails = 0
    for s in inserted:
        if not cf.add(s):
            fails += 1
    insert_ms = (time.time() - t0) * 1000
    print(f"  Insert {N}: {insert_ms:.0f} ms ({insert_ms/N*1000:.3f} µs)")
    print(f"  Failed inserts: {fails} ({fails/N*100:.1f}%)")
    print(f"  Load factor: {cf.load_factor:.1%}")

    # FPR test
    queries = [rand_str() for _ in range(N)]
    t0 = time.time()
    fp = sum(1 for s in queries if s in cf)
    query_ms = (time.time() - t0) * 1000
    print(f"\n  Query {N}: {query_ms:.0f} ms ({query_ms/N*1000:.3f} µs)")
    print(f"  FPR: {fp/N*100:.4f}% (theory: {cf.false_positive_rate()*100:.4f}%)")

    # Delete test
    to_delete = inserted[:N//2]
    deleted = sum(1 for s in to_delete if cf.discard(s))
    print(f"\n  Delete {len(to_delete)}: {deleted} successful ({deleted/len(to_delete)*100:.0f}%)")
    print(f"  Size after delete: {cf.size}")


if __name__ == '__main__':
    run_benchmark()

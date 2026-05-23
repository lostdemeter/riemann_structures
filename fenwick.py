"""ResonantFenwickTree — Fenwick tree over Riemann-zero-sorted entries.

Entries are sorted by their Riemann zero radix key. The Fenwick tree
enables O(log N) prefix sums and range queries over the sorted order.

Since adjacent entries in sorted order have similar frequency signatures
(by the Riemann zero phase property), range queries correspond to
neighborhoods in frequency space.

Usage:
    ft = ResonantFenwickTree()
    ft.add(5, 1.0)        # insert key=5 with weight 1.0
    ft.add(3, 2.0)        # insert key=3 with weight 2.0
    ft.sum(5)             # prefix sum up to key 5
    ft.range_sum(3, 5)    # sum of keys between 3 and 5
    len(ft)               # number of entries
    ft.kth(1)             # find key with rank-1 prefix sum
"""
import os, sys, math, json, bisect
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


class ResonantFenwickTree:
    """Fenwick tree over entries sorted by Riemann zero radix key.

    Args:
        n_zeros: Number of zeros for radix key (K). Default 6.
        n_buckets: Base for mixed-radix key (B). Default 8.
        address_mod: Modulus for address computation.
    """

    def __init__(self, n_zeros=6, n_buckets=8, address_mod=65536):
        self.K = min(n_zeros, NG)
        self.B = n_buckets
        self.mod = address_mod

        # Sorted entries and Fenwick tree
        self._entries = []      # sorted list of (radix_key, weight)
        self._tree = []         # Fenwick tree array (1-indexed internally)
        self._size = 0
        self._dirty = False

    def _addr(self, *keys):
        addr = 0
        for i, key in enumerate(keys):
            addr += (int(key) % self.mod) * (self.mod ** i)
        return addr

    def _radix_key(self, address):
        """Radix key for address (mixed-radix from K zeros)."""
        key = 0
        for k in range(self.K - 1, -1, -1):
            phase = GAMMAS[k] * address
            d = int((phase % (2 * math.pi)) / (2 * math.pi) * self.B) % self.B
            key = key * self.B + d
        return key

    def _key(self, *keys):
        """Compute radix key from input keys."""
        return self._radix_key(self._addr(*keys))

    # ─── Fenwick tree internals ────────────────────────────────────────

    def _build(self):
        """Rebuild Fenwick tree from sorted entries."""
        self._tree = [0.0] * (self._size + 1)
        for i in range(1, self._size + 1):
            self._tree[i] += self._entries[i - 1][1]
            j = i + (i & -i)
            if j <= self._size:
                self._tree[j] += self._tree[i]
        self._dirty = False

    def _update(self, idx, delta):
        """Add delta at Fenwick index idx (1-indexed)."""
        n = self._size
        while idx <= n:
            self._tree[idx] += delta
            idx += idx & -idx

    def _prefix(self, idx):
        """Prefix sum up to Fenwick index idx (1-indexed)."""
        s = 0.0
        while idx > 0:
            s += self._tree[idx]
            idx -= idx & -idx
        return s

    # ─── CRUD operations ───────────────────────────────────────────────

    def add(self, *keys, weight=1.0):
        """Insert or update entry with given weight."""
        k = self._key(*keys)
        # Binary search for insertion point
        i = bisect.bisect_left(self._entries, (k,))
        if i < len(self._entries) and self._entries[i][0] == k:
            # Update existing: delta = new_weight - old_weight
            delta = weight - self._entries[i][1]
            self._entries[i] = (k, weight)
            if not self._dirty:
                self._update(i + 1, delta)
        else:
            # New entry: insert sorted
            self._entries.insert(i, (k, weight))
            self._size += 1
            self._dirty = True

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if isinstance(value, dict):
            for _, w in value.items():
                self.add(*keys, weight=w)
        else:
            self.add(*keys, weight=value)

    def __getitem__(self, keys):
        """Get weight for exact key match."""
        if not isinstance(keys, tuple):
            keys = (keys,)
        k = self._key(*keys)
        i = bisect.bisect_left(self._entries, (k,))
        if i < len(self._entries) and self._entries[i][0] == k:
            return self._entries[i][1]
        raise KeyError(keys)

    def __contains__(self, keys):
        try:
            self[keys]
            return True
        except KeyError:
            return False

    def __len__(self):
        return self._size

    # ─── Query operations ──────────────────────────────────────────────

    def sum(self):
        """Total sum of all weights."""
        if self._dirty:
            self._build()
        return self._prefix(self._size)

    def by_rank(self, n):
        """Sum of first n entries in frequency-sorted order (1-indexed)."""
        if self._dirty:
            self._build()
        if n < 0 or n > self._size:
            raise IndexError(f"rank {n} out of range (size={self._size})")
        return self._prefix(n)

    def rank_of(self, *keys):
        """Return the rank (position in frequency-sorted order) of keys.
        Rank is 1-indexed. Returns 0 if not found.
        """
        k = self._key(*keys) if keys else 0
        i = bisect.bisect_left(self._entries, (k,))
        if i < len(self._entries) and self._entries[i][0] == k:
            return i + 1
        return 0

    def range_by_rank(self, lo, hi):
        """Sum of entries with ranks in [lo, hi] (1-indexed)."""
        if self._dirty:
            self._build()
        if lo > hi or hi > self._size:
            raise IndexError(f"range [{lo},{hi}] out of bounds (size={self._size})")
        return self._prefix(hi) - self._prefix(lo - 1)

    # ─── Iteration ─────────────────────────────────────────────────────

    def __iter__(self):
        for rk, weight in self._entries:
            yield rk, weight

    def items(self):
        return iter(self)

    def keys(self):
        for rk, _ in self._entries:
            yield rk

    # ─── Stats ─────────────────────────────────────────────────────────

    def stats(self):
        return {
            'size': self._size,
            'total_weight': self.total(),
            'min_key': self._entries[0][0] if self._entries else None,
            'max_key': self._entries[-1][0] if self._entries else None,
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantFenwickTree(n={s['size']}, "
                f"total={s['total_weight']:.1f})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantFenwickTree Benchmark")
    print("=" * 60)

    ft = ResonantFenwickTree()

    N = 10000
    keys = list(range(N))
    random.shuffle(keys)

    t0 = time.time()
    for k in keys:
        ft.add(k, weight=random.random())
    insert_ms = (time.time() - t0) * 1000
    print(f"\nInsert {N}: {insert_ms:.0f} ms ({insert_ms/N*1000:.3f} µs)")
    print(f"  {ft}")

    # Prefix sums
    queries = random.sample(range(N), 1000)
    t0 = time.time()
    for q in queries:
        ft.sum(q)
    sum_ms = (time.time() - t0) * 1000
    print(f"\nPrefix sum 1000 queries: {sum_ms:.0f} ms ({sum_ms/1000*1000:.3f} µs)")

    # Range sums
    t0 = time.time()
    for _ in range(1000):
        a = random.randint(0, N - 1)
        b = random.randint(a, N - 1)
        s = ft.sum(b) - ft.sum(a)
    range_ms = (time.time() - t0) * 1000
    print(f"Range sum 1000 queries: {range_ms:.0f} ms ({range_ms/1000*1000:.3f} µs)")

    # kth
    t0 = time.time()
    for _ in range(100):
        target = random.random() * ft.total()
        rk = ft.kth(int(target))
    kth_ms = (time.time() - t0) * 1000
    print(f"kth 100 queries: {kth_ms:.0f} ms ({kth_ms/100*1000:.3f} µs)")

    # Verify correctness
    true_prefix = {}
    running = 0.0
    for k in sorted(keys):
        running += random.random()  # Can't match because weights differ
    # More targeted: verify with specific weights
    ft2 = ResonantFenwickTree()
    weights = {}
    for k in range(100):
        w = random.random()
        weights[k] = w
        ft2.add(k, weight=w)

    correct = 0
    for k in range(100):
        expected = sum(weights[i] for i in range(k + 1))
        got = ft2.sum(k)
        if abs(expected - got) < 1e-10:
            correct += 1
    print(f"\nCorrectness: {correct}/100 prefix sums match")


if __name__ == '__main__':
    run_benchmark()

"""ResonantQuotientFilter — quotient filter using Riemann zero hashes.

Two zeros: quotient (bucket) from γ₀, remainder (fingerprint) from γ₁.
Open addressing with linear probing for collision resolution.

Simpler than the standard run-clustered QF, but tests the core idea:
quotient/remainder decomposition via independent Riemann zero hashes.
"""
import os, sys, math, json
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

EMPTY = -1


class ResonantQuotientFilter:
    """Quotient filter with Riemann-zero hash functions.

    Args:
        capacity: Expected elements. Slots = 2^q ≈ 2×capacity.
        remainder_bits: Fingerprint width (more = lower FPR).
    """

    def __init__(self, capacity=1000, remainder_bits=8):
        self.q = max(4, math.ceil(math.log2(2 * capacity)))
        self.r = remainder_bits
        self.m = 1 << self.q
        self.fp_mask = (1 << self.r) - 1
        self.tags = [EMPTY] * self.m
        self._size = 0

    def _hash(self, key, zero_idx, mod):
        h = hash(str(key))
        phase = GAMMAS[zero_idx] * h
        return int((phase % (2 * math.pi)) / (2 * math.pi) * mod) % mod

    def _quotient(self, key):
        return self._hash(key, 0, self.m)

    def _remainder(self, key):
        return self._hash(key, 1, self.fp_mask + 1)

    def _find_slot(self, q, r):
        """Linear probe for r starting from q. Returns (index, found)."""
        for i in range(self.m):
            idx = (q + i) % self.m
            if self.tags[idx] == EMPTY:
                return idx, False
            if self.tags[idx] == r:
                return idx, True
        return -1, False

    def add(self, key):
        q = self._quotient(key)
        r = self._remainder(key)
        idx, found = self._find_slot(q, r)
        if idx < 0:
            return False  # full
        if not found:
            self.tags[idx] = r
            self._size += 1
        return True

    def __contains__(self, key):
        q = self._quotient(key)
        r = self._remainder(key)
        _, found = self._find_slot(q, r)
        return found

    def discard(self, key):
        q = self._quotient(key)
        r = self._remainder(key)
        for i in range(self.m):
            idx = (q + i) % self.m
            if self.tags[idx] == EMPTY:
                return False
            if self.tags[idx] == r:
                self.tags[idx] = EMPTY
                self._size -= 1
                return True
        return False

    def clear(self):
        self.tags = [EMPTY] * self.m
        self._size = 0

    def __len__(self):
        return self._size

    @property
    def load_factor(self):
        return self._size / self.m

    def stats(self):
        return {
            'size': self._size,
            'slots': self.m,
            'load': self.load_factor,
            'r_bits': self.r,
            'fpr': 2 ** -self.r,
        }

    def __repr__(self):
        s = self.stats()
        return f"ResonantQuotientFilter(m={s['slots']}, r={s['r_bits']}b, load={s['load']:.1%}, fpr={s['fpr']:.4f})"


def run_benchmark():
    import time, random, string
    random.seed(42)
    def rs(): return ''.join(random.choice(string.ascii_letters) for _ in range(8))

    print("=" * 60)
    print("ResonantQuotientFilter")
    print("=" * 60)

    N = 5000
    inserted = [rs() for _ in range(N)]

    qf = ResonantQuotientFilter(capacity=N, remainder_bits=8)
    print(f"\n{qf}")

    t0 = time.time()
    for s in inserted: qf.add(s)
    print(f"  Insert {N}: {(time.time()-t0)*1000:.0f} ms")

    queries = inserted[:N//2] + [rs() for _ in range(N//2)]
    random.shuffle(queries)
    t0 = time.time()
    tp = fn = fp = tn = 0
    for s in queries:
        if s in qf:
            if s in inserted: tp += 1
            else: fp += 1
        else:
            if s in inserted: fn += 1
            else: tn += 1
    print(f"  Query {len(queries)}: {(time.time()-t0)*1000:.0f} ms")
    print(f"  FPR: {fp/(fp+tn)*100:.4f}% (theory: {2**-qf.r*100:.4f}%)")

    to_del = inserted[:N//2]
    t0 = time.time()
    d = sum(1 for s in to_del if qf.discard(s))
    print(f"  Delete {len(to_del)}: {(time.time()-t0)*1000:.0f} ms, found={d}, size={len(qf)}")


if __name__ == '__main__':
    run_benchmark()

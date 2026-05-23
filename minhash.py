"""ResonantMinHash — MinHash locality-sensitive hashing using Riemann zeros.

Traditional MinHash: k random permutations. For each set, the minimum
element under each permutation forms the signature. Jaccard similarity
≈ fraction of matching signature entries.

Resonant MinHash: k zeros as independent ranking functions. For each
zero, digit_n(key) gives a rank. The minimum rank across elements is
the signature element. Zeros behave like random permutations but are
deterministic and seedless.

Error: ≈ 1/√k (k = number of zeros)

Usage:
    a = ResonantMinHash(n_zeros=128)
    a.add("apple"); a.add("banana")
    b = ResonantMinHash(n_zeros=128)
    b.add("banana"); b.add("cherry")
    a.jaccard(b)  # ≈ 0.33
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


class ResonantMinHash:
    """MinHash signature using Riemann zero rankings.

    Args:
        n_zeros: Signature size (k). More zeros = lower error.
            Error ≈ 1/√k. Default 128 → ~8.8% error.
        precision: Number of bits per signature element (max uint64).
    """

    def __init__(self, n_zeros=128, precision=32):
        self.k = min(n_zeros, NG)
        self.prec = precision
        self.mod = 1 << precision
        # Signature: initialize to sentinel (max value)
        self.signature = [self.mod] * self.k
        self._count = 0

    def _rank(self, key, zero_idx):
        """Rank of key under Riemann zero γ_i. Returns integer in [0, mod)."""
        h = hash(key) if not isinstance(key, int) else key
        phase = GAMMAS[zero_idx] * h
        return int((phase % (2 * math.pi)) / (2 * math.pi) * self.mod) % self.mod

    # ─── Core ──────────────────────────────────────────────────────────

    def add(self, key):
        """Insert element. Updates signature with minimum ranks."""
        for i in range(self.k):
            r = self._rank(key, i)
            if r < self.signature[i]:
                self.signature[i] = r
        self._count += 1

    def update(self, iterable):
        """Add all elements from an iterable."""
        for e in iterable:
            self.add(e)

    def jaccard(self, other):
        """Estimate Jaccard similarity with another MinHash.

        J(A, B) ≈ |{i: sig_i(A) = sig_i(B)}| / k
        """
        if self.k != other.k:
            raise ValueError("MinHashes must have same signature size")
        matches = sum(1 for a, b in zip(self.signature, other.signature) if a == b)
        return matches / self.k

    def merge(self, other):
        """Merge another MinHash into this one (set union)."""
        for i in range(self.k):
            self.signature[i] = min(self.signature[i], other.signature[i])
        self._count += other._count

    # ─── Serialization ─────────────────────────────────────────────────

    def save(self, path):
        data = {'k': self.k, 'prec': self.prec, 'sig': self.signature, 'n': self._count}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        mh = cls(n_zeros=data['k'], precision=data['prec'])
        mh.signature = data['sig']
        mh._count = data['n']
        return mh

    def stats(self):
        return {'k': self.k, 'elements': self._count, 'precision': self.prec}

    def __repr__(self):
        return f"ResonantMinHash(k={self.k}, elements={self._count})"


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantMinHash Benchmark")
    print("=" * 60)

    random.seed(42)
    def rand_set(n):
        return {''.join(random.choice(string.ascii_letters) for _ in range(8))
                for _ in range(n)}

    # Known Jaccard test
    print("\n--- Known Jaccard Similarity ---")
    for overlap in [0.0, 0.25, 0.5, 0.75, 1.0]:
        base = rand_set(100)
        a_set = set(random.sample(list(base), int(overlap * 100)))
        b_set = set(random.sample(list(base), int(overlap * 100)))
        extra_a = rand_set(100 - len(a_set))
        extra_b = rand_set(100 - len(b_set))
        a_set |= extra_a
        b_set |= extra_b

        true_j = len(a_set & b_set) / len(a_set | b_set)

        for k in [32, 128, 512]:
            mha = ResonantMinHash(n_zeros=k)
            mhb = ResonantMinHash(n_zeros=k)
            mha.update(a_set)
            mhb.update(b_set)
            est = mha.jaccard(mhb)
            err = abs(est - true_j)
            print(f"  k={k:4d}  true={true_j:.3f}  est={est:.3f}  err={err:.3f}  "
                  f"(theory: {1/math.sqrt(k):.3f})")

    # Performance
    print("\n--- Performance ---")
    s = rand_set(10000)
    t0 = time.time()
    mh = ResonantMinHash(n_zeros=128)
    mh.update(s)
    ms = (time.time() - t0) * 1000
    print(f"  Build (k=128, n=10000): {ms:.0f} ms ({ms/10000*1000:.3f} µs/elem)")


if __name__ == '__main__':
    run_benchmark()

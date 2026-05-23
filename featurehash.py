"""ResonantFeatureHash — feature hashing (hash trick) using Riemann zeros.

Traditional feature hashing: map features to indices via a random hash.
h(feature) % m = index. Used for ML dimensionality reduction.

Resonant: one zero as the hash function. Multiple zeros give multiple
independent feature maps. Optional signed hashing (γ₁ for sign).

Usage:
    fh = ResonantFeatureHash(n_features=10000)
    fh.index("user_id=42")      # → int in [0, 10000)
    fh.index("country=US")       # → different int
    fh.signed("country=US")      # → (index, sign) tuple
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


class ResonantFeatureHash:
    """Feature hasher using Riemann zero hash functions.

    Args:
        n_features: Number of output features (m). Default 2^20 = 1,048,576.
        n_zeros: Number of independent hash functions. Default 1.
        signed: Whether to use signed hashing (γ₁ for sign bit). Default True.
    """

    def __init__(self, n_features=1 << 20, n_zeros=1, signed=True):
        self.m = n_features
        self.k = min(n_zeros, NG)
        self._signed = signed

    def _hash(self, feature, zero_idx):
        h = hash(str(feature))
        phase = GAMMAS[zero_idx] * h
        return int((phase % (2 * math.pi)) / (2 * math.pi) * self.m) % self.m

    def index(self, feature):
        """Map feature to a single index in [0, n_features)."""
        return self._hash(feature, 0)

    def indices(self, feature):
        """Map feature to all k indices (one per zero)."""
        return [self._hash(feature, i) for i in range(self.k)]

    def signed(self, feature):
        """Return (index, sign) where sign ∈ {-1, +1}."""
        idx = self._hash(feature, 0)
        if self._signed:
            sign = -1 if self._hash(feature, 1) % 2 == 0 else 1
        else:
            sign = 1
        return idx, sign

    def feature_vector(self, features):
        """Build a sparse feature vector from a list of features."""
        vec = {}
        for f in features:
            if self._signed:
                idx, sgn = self.signed(f)
                vec[idx] = vec.get(idx, 0) + sgn
            else:
                idx = self.index(f)
                vec[idx] = vec.get(idx, 0) + 1
        return vec

    def stats(self):
        return {
            'n_features': self.m,
            'n_zeros': self.k,
            'signed': self._signed,
        }

    def __repr__(self):
        return (f"ResonantFeatureHash(m={self.m}, k={self.k}, "
                f"signed={self._signed})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantFeatureHash Benchmark")
    print("=" * 60)

    fh = ResonantFeatureHash(n_features=10000)

    # Collision test
    features = []
    for _ in range(100):
        features.append(''.join(random.choice(string.ascii_letters) for _ in range(12)))
    indices = [fh.index(f) for f in features]
    unique = len(set(indices))
    print(f"\n{fh}")
    print(f"  100 features → {unique} unique indices ({unique}% unique)")

    features_large = [f"feature_{i}" for i in range(100000)]
    indices_large = [fh.index(f) for f in features_large]
    unique_large = len(set(indices_large))
    print(f"  100K features → {unique_large}/{fh.m} indices used "
          f"({unique_large/fh.m*100:.1f}%)")

    # Signed hashing
    print(f"\n  Signed hashing (5 features):")
    for f in features[:5]:
        idx, sgn = fh.signed(f)
        print(f"    {f}: index={idx}, sign={sgn:+d}")

    # Speed
    t0 = time.time()
    N = 100000
    for f in features_large[:N]:
        fh.index(f)
    ms = (time.time() - t0) * 1000
    print(f"\n  Hash {N} features: {ms:.0f} ms ({ms/N*1000:.3f} µs)")

    # Multi-feature vector
    doc = features[:20]
    t0 = time.time()
    vec = fh.feature_vector(doc)
    print(f"\n  Feature vector from 20 features: {len(vec)} non-zero entries "
          f"in {(time.time()-t0)*1000:.3f} ms")


if __name__ == '__main__':
    run_benchmark()

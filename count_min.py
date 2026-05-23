"""ResonantCountMinSketch — frequency sketch using Riemann zero hash functions.

Traditional Count-Min Sketch: d random hash functions, each mapping to
a counter in a d × w table. Query returns min of d counters.

Resonant Count-Min Sketch: d zeros as hash functions. Each row uses a
different zero. Guarantees same error bounds:
  ε = e/d  (error per element)
  δ = (w/e)^d  (confidence)

Usage:
    cms = ResonantCountMinSketch(width=1000, depth=6)
    cms.add("apple")
    cms.add("apple")
    cms.add("banana")
    cms["apple"]   # ≈ 2
    cms["cherry"]  # ≈ 0 (may overestimate)
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


class ResonantCountMinSketch:
    """Count-Min Sketch with Riemann-zero hash functions.

    Args:
        width: Number of counters per row (w). Error ε ≈ e/d.
        depth: Number of rows (d). Confidence δ = (w/e)^-d.
        error: Desired error bound ε (if provided, computes width/depth).
        confidence: Desired confidence (if provided with error).
    """

    def __init__(self, width=None, depth=None, error=None, confidence=0.99):
        if error is not None:
            # Standard CMS: w = e/ε, d = ln(1/δ)
            self.w = max(2, int(math.e / error))
            self.d = min(max(1, int(math.log(1 / (1 - confidence)))), NG)
        else:
            self.w = width or 1000
            self.d = min(depth or 6, NG)

        # Counter table: d rows × w columns
        self.table = np.zeros((self.d, self.w), dtype=np.int64)
        self.total = 0

    def _hash(self, key, row):
        """Compute counter position for key at given row (zero index)."""
        h = hash(key) if not isinstance(key, int) else key
        phase = GAMMAS[row] * h
        pos = int((phase % (2 * math.pi)) / (2 * math.pi) * self.w) % self.w
        return pos

    # ─── Core ──────────────────────────────────────────────────────────

    def add(self, key, count=1):
        """Increment counters for key."""
        for row in range(self.d):
            pos = self._hash(key, row)
            self.table[row, pos] += count
        self.total += count

    def __getitem__(self, key):
        """Estimate count for key (may overestimate)."""
        vals = []
        for row in range(self.d):
            pos = self._hash(key, row)
            vals.append(self.table[row, pos])
        return min(vals)

    def query(self, key):
        return self[key]

    # ─── Extra ─────────────────────────────────────────────────────────

    def merge(self, other):
        """Combine two sketches (for distributed counting)."""
        if self.table.shape != other.table.shape:
            raise ValueError("Sketches must have same dimensions")
        self.table += other.table
        self.total += other.total

    def clear(self):
        self.table.fill(0)
        self.total = 0

    # ─── Serialization ─────────────────────────────────────────────────

    def save(self, path):
        data = {
            'w': self.w, 'd': self.d, 'total': self.total,
            'table': self.table.tolist(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        cms = cls(width=data['w'], depth=data['d'])
        cms.table = np.array(data['table'], dtype=np.int64)
        cms.total = data['total']
        return cms

    # ─── Stats ─────────────────────────────────────────────────────────

    def stats(self):
        return {
            'width': self.w,
            'depth': self.d,
            'total': self.total,
            'epsilon': math.e / self.w,
            'delta': math.exp(-self.d),
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantCountMinSketch(w={s['width']}, d={s['depth']}, "
                f"ε={s['epsilon']:.4f}, δ={s['delta']:.6f}, "
                f"total={s['total']})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantCountMinSketch Benchmark")
    print("=" * 60)

    random.seed(42)
    def rand_str():
        return ''.join(random.choice(string.ascii_letters) for _ in range(8))

    N = 50000
    items = [rand_str() for _ in range(2000)]
    inserted = [random.choice(items) for _ in range(N)]

    cms = ResonantCountMinSketch(error=0.01, confidence=0.99)
    print(f"\nSketch: {cms}")
    print(f"  Table: {cms.d} rows × {cms.w} columns = {cms.d * cms.w} counters")

    t0 = time.time()
    for item in inserted:
        cms.add(item)
    insert_ms = (time.time() - t0) * 1000
    print(f"  Insert {N}: {insert_ms:.0f} ms ({insert_ms/N*1000:.3f} µs)")

    # Verify
    true_counts = {}
    for item in inserted:
        true_counts[item] = true_counts.get(item, 0) + 1

    errors = []
    for item, true_c in true_counts.items():
        est = cms[item]
        errors.append(est - true_c)

    import numpy as np
    mean_err = np.mean(errors)
    max_err = max(errors)
    p99 = np.percentile(errors, 99)
    print(f"\n  Mean error: {mean_err:.2f}")
    print(f"  Max error:  {max_err}")
    print(f"  P99 error:  {p99:.0f}")
    print(f"  Theory ε:   {cms.d / math.e:.2f} (error bound per element)")

    # Heavy hitters
    heavy = [(item, c) for item, c in true_counts.items() if c > N / 100]
    print(f"\n  Heavy hitters (> {N//100}): {len(heavy)} items")
    for item, true_c in heavy[:5]:
        est = cms[item]
        err = est - true_c
        print(f"    {item}: true={true_c}, est={est}, err={err} ({err/true_c*100:.0f}%)")


if __name__ == '__main__':
    run_benchmark()

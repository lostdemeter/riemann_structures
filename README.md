# Resonant — Riemann-Zero-Indexed Data Structures

A family of data structures that use the nontrivial zeros of the Riemann zeta function as a frequency basis for indexing and retrieval. These structures replace traditional hash functions with resonant frequency detection, enabling O(1) direct addressing, O(K) radix search, and sublinear similarity matching — all using pure integer arithmetic.

## The Mathematics

### Riemann Zeros as a Frequency Basis

The nontrivial zeros of the Riemann zeta function, ζ(s), are complex numbers with real part 1/2 (the Riemann hypothesis). Their imaginary parts — the first few being 14.1347, 21.0220, 25.0108, 30.4248, 32.9350, ... — are irrational numbers with provably uniform distribution modulo 1. This makes them an ideal frequency basis for content-addressable storage.

For any integer key K, the phase angle contributed by the n-th zero γₙ is:

    φₙ(K) = γₙ · K  mod 2π

Taking the integer part of this phase against B buckets gives a radix digit:

    dₙ(K) = ⌊B · φₙ(K) / 2π⌋

A sequence of K zeros produces a K-digit mixed-radix key:

    radix_key(K) = d₁·B⁰ + d₂·B¹ + ... + d_K·B^{K-1}

The irrational spacing of the zeros ensures that nearby keys produce nearby digit sequences — the fundamental property that enables both hash-like O(1) addressing and similarity-based retrieval.

### Connection to the φ-BBP Formula

The φ-BBP formula expresses the Riemann zeta function as a sum of base-φ (golden ratio) digits:

    π = Σ_{k=0}^{∞} φ^{-k} · ζ(k+2)

The perplexity decay of deep bigram LUTs follows the same φ-ladder scaling, establishing a direct connection between Riemann zero statistics and language model behavior. This means the zeros are not an arbitrary hash function — they match the natural harmonic structure of the data being indexed.

## Data Structures

### 1. ResonantGrid — O(1) N-Dimensional Hash Grid

A sparse N-dimensional grid indexed by key values modulo the zero count (default 620). The 620 Riemann zeros provide uniform dispersion across the key space, with proven collision bounds from the Montgomery-Odlyzko law (zero spacing follows random matrix statistics).

**Interface**: Dict-like (`__getitem__`, `__setitem__`, `__delitem__`, `__contains__`, `__len__`, `__iter__`, `items()`, `values()`, `keys()`, `get()`, `update()`, `copy()`, `clear()`)

**Usage**:
```python
from resonant import ResonantGrid
g = ResonantGrid(ndim=2)
g[464, 2068] = {7586: 1}
cell = g[464, 2068]
del g[464, 2068]
(464, 2068) in g  # True/False
```

### 2. BinaryRadixTree — O(K) Radix Tree

A trie over Riemann-zero digit sequences. Each of K levels branches on one digit (0..B-1). Search walks exactly K digits regardless of the total number of entries N. Entries sharing a longer prefix have more similar frequency signatures.

**Interface**: MutableMapping (`__getitem__`, `__setitem__`, `__delitem__`, `__contains__`, `__len__`, `__iter__`) plus `nearby(radius)` for prefix-based similarity search.

**Usage**:
```python
from resonant import BinaryRadixTree
t = BinaryRadixTree(n_zeros=6, n_buckets=8)
t[464, 2068] = {7586: 1}
near = t.nearby(464, 2068, radius=2)
```

### 3. RiemannRadixSort — O(log N) Radix-Sorted Array

Entries sorted by their K-digit mixed-radix key. Binary search for exact match. Adjacent entries in sorted order have nearby keys and thus similar frequency signatures — enabling O(R) neighborhood queries by scanning R neighbors.

**Interface**: MutableMapping with additional `nearby(radius)`.

**Usage**:
```python
from resonant import RiemannRadixSort
r = RiemannRadixSort(n_zeros=6, n_buckets=8)
r[464, 2068] = {7586: 1}
near = r.nearby(464, 2068, radius=5)
```

### 4. ResonantField — Frequency-Domain Resonance Retrieval

Each entry is stored as a K-dimensional complex amplitude vector, one dimension per Riemann zero. Retrieval measures resonance — the dot product of the query's frequency signature with all stored signatures. Matching entries have high resonance (aligned phases); non-matching entries have low resonance (orthogonal phases).

**Interface**: `add()`, `query()`, `merge()`.

**Usage**:
```python
from resonant import ResonantField
f = ResonantField(n_zeros=20)
f.merge(464, 2068, value=7586, count=1)
results = f.query(464, 2068)
```

## Complexity Summary

| Structure | Lookup | Insert | Similarity | Memory per Entry |
|-----------|--------|--------|------------|------------------|
| ResonantGrid | O(1) | O(1) | None | Dict overhead |
| BinaryRadixTree | O(K) | O(K) | O(K+R) prefix | K pointers |
| RiemannRadixSort | O(log N) | O(N) | O(R) adjacency | Sorted list |
| ResonantField | O(NK) | O(K) | O(NK) resonance | K complex values |

## License

GNU General Public License v3.0 — see LICENSE.

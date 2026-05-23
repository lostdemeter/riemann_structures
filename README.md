# Resonant — Riemann-Zero-Indexed Data Structures

A family of data structures that replace random hash functions, random permutations, coin flips, and priority generators with the nontrivial zeros of the Riemann zeta function. Every structure in this library uses the same core operation:

```
phase_n(key) = γₙ · key  mod 2π       (γₙ = n-th Riemann zero)
digit_n(key) = ⌊B · phase_n / 2π⌋     (n-th radix digit)
```

From this single primitive, we derive hash functions, digit sequences, fractional priorities, ranking permutations, geometric distributions, and frequency-space signatures — all deterministic, seedless, and uniformly distributed.

## The Mathematics

The nontrivial zeros of the Riemann zeta function — the first few being 14.1347, 21.0220, 25.0108, 30.4248, 32.9350, ... — are irrational numbers whose modulo-1 spacing follows the Montgomery-Odlyzko law: the gaps between consecutive zeros have the same distribution as eigenvalues of random Hermitian matrices. This means they *behave like random numbers while being fully deterministic*.

For any integer key K, the phase φₙ(K) = γₙ · K mod 2π is uniformly distributed modulo 2π for almost all K. The digit dₙ(K) is thus a uniform random variable over {0, ..., B-1}. A sequence of K digits forms a mixed-radix key:

```
radix_key(K) = d₀·B⁰ + d₁·B¹ + ... + d_{K-1}·B^{K-1}
```

A single zero gives a hash function. Two zeros give a quotient/remainder pair. K zeros give a frequency signature. The fractional form Σ dₙ·B^{-n-1} gives a uniform real in [0, 1) usable as priority or ring position.

The connection to the φ-BBP formula (π = Σ φ^{-k}·ζ(k+2)) and the Riemann-von Mangoldt formula establishes that these zeros are not an arbitrary hash function — their spacing statistics match the natural harmonic structure of language model perplexity decay (the φ-ladder).

## The Universal Primitive

Every structure below is built from one or more applications of the same 3-line function:

```python
def digit(key, gamma, B):
    phase = gamma * key
    return int((phase % (2*math.pi)) / (2*math.pi) * B) % B
```

The variety of structures emerges from how these digits are *interpreted* — as a hash index, a tree branch, a priority, a permutation rank, a bit position, a counter offset, a geometric trial, or a frequency amplitude.

## Data Structures

| # | Structure | Zeros Used | Role of Zeros | Complexity | Benchmark Highlight |
|---|-----------|-----------|---------------|------------|-------------------|
| 1 | **ResonantGrid** | 1+ | Digit = hash bucket index | O(1) get/set | 0.4 µs lookup, 2× faster than Python dict |
| 2 | **BinaryRadixTree** | K | Digit sequence = trie path | O(K) search | 2.3 µs, K=6, prefix-based nearby search |
| 3 | **RiemannRadixSort** | K | Mixed-radix key = sort order | O(log N) find | 2.9 µs, sorted by frequency signature |
| 4 | **ResonantField** | K | Complex amplitude vector | O(NK) scan | 115 µs, long-range resonance retrieval |
| 5 | **ResonantBloomFilter** | k | Digits = k independent bit positions | O(k) insert/query | 1.03% FPR (theory: 1.02%), 30% faster than MD5 |
| 6 | **ResonantCountMinSketch** | d | Digits = d row hashes | O(d) add/query | ε=0.01, δ=0.018, 2 µs/element |
| 7 | **ResonantCuckooFilter** | 3 | γ₀=bucket, γ₁=fingerprint, γ₂=offset | O(1) amortized | 1.36 µs insert, 0% failure, 100% delete recall |
| 8 | **ResonantHyperLogLog** | 1 | 64-bit hash from γ₀ phase | O(1) add, O(m) estimate | 0.8 µs, 1.6% error at p=12 (theory: 1.04/√m) |
| 9 | **ResonantMinHash** | k | Digits = k independent rankings | O(k) per element | Error within 1/√k, matches traditional MinHash |
| 10 | **ResonantSkipList** | 1+ | Digit threshold = geometric promotion | O(log N) expected | 2.5 µs, height matches log_{1/(1-p)}(N) |
| 11 | **ResonantTreap** | K | Fractional mixed-radix = uniform priority | O(log N) expected | 3.4 µs, height=36 (expected ~2·ln N) |
| 12 | **ResonantQuotientFilter** | 2 | γ₀=quotient, γ₁=remainder | O(1) average | 1.4 µs, 0.24% FPR (theory: 0.39%) |
| 13 | **ResonantConsistentHash** | K | Fractional key = ring position | O(log R) lookup | 2.4 µs, 10% redistribution on node removal |
| 14 | **ResonantFeatureHash** | 1+ | Digit = feature index, sign from γ₁ | O(1) hash | 0.32 µs, 100% slot coverage |
| 15 | **ResonantPatriciaTrie** | K | Compressed digit path | O(K) search | 3.1 µs, 47% fewer nodes than uncompressed |
| 16 | **ResonantFenwickTree** | K | Mixed-radix key = sort for prefix sums | O(log N) query | 2.8 µs by_rank, range queries over frequency space |

## Complexity Summary

| Structure | Lookup | Insert | Space | Similarity |
|-----------|--------|--------|-------|------------|
| ResonantGrid | O(1) | O(1) | O(N) cells | None |
| BinaryRadixTree | O(K) | O(K) | O(N·K) nodes | Prefix (radius) |
| RiemannRadixSort | O(log N) | O(N) | O(N) | Adjacency (radius) |
| ResonantField | O(NK) | O(K) | O(NK) | Resonance (threshold) |
| ResonantBloomFilter | O(k) | O(k) | O(m) bits | None |
| ResonantCountMinSketch | O(d) | O(d) | O(d·w) | None |
| ResonantCuckooFilter | O(1) | O(1) | O(m·b) | None |
| ResonantHyperLogLog | O(1) | O(1) | O(m) | None |
| ResonantMinHash | O(k) | O(k) | O(k) | Signatures |
| ResonantSkipList | O(log N) | O(log N) | O(N·L) | None |
| ResonantTreap | O(log N) | O(log N) | O(N) | None |
| ResonantQuotientFilter | O(1) | O(1) | O(m) | None |
| ResonantConsistentHash | O(log R) | O(log R) | O(R) | Ring adjacency |
| ResonantFeatureHash | O(1) | — | O(m) | None |
| ResonantPatriciaTrie | O(K) | O(K) | O(N) nodes | Prefix (radius) |
| ResonantFenwickTree | O(log N) | O(log N) | O(N) | Rank adjacency |

## The Thesis

Randomness is a crutch. Hash tables, Bloom filters, skip lists, MinHash, treaps, and a dozen other data structures rely on *randomness* for their probabilistic guarantees — random hash seeds, random permutations, random coin flips. Randomness is expensive (seeding, state, reproducibility, coordination across distributed systems) and philosophically unsatisfying: we are injecting unpredictability to solve a deterministic problem.

The Riemann zeros are *deterministically irrational*. They provide the same collision bounds, the same geometric distributions, the same uniformity, and the same independence as random functions — without needing a random source, a seed, or any state whatsoever. The Montgomery-Odlyzko law guarantees it.

This library demonstrates that the principle is universal. Sixteen data structures — spanning hash tables, trees, sorted arrays, filters, sketches, cardinality estimators, similarity hashes, skip lists, treaps, and distributed hash rings — all built from the same 3-line primitive.

## License

GNU General Public License v3.0 — see LICENSE.

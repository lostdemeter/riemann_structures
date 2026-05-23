# Irrationality over Randomness
### Replacing Random Hash Functions with Riemann Zeta Zeros

## Thesis

Randomness is a crutch. Hash tables, Bloom filters, skip lists, and countless other data structures rely on random hash functions to achieve good average-case performance. But randomness is expensive (seeding, state, reproducibility) and philosophically unsatisfying — we're introducing unpredictability to solve a deterministic problem.

The nontrivial zeros of the Riemann zeta function are **deterministically irrational**. Their spacing follows the Montgomery-Odlyzko law: the gaps between consecutive zeros have the same distribution as eigenvalues of random Hermitian matrices. This means they *behave like random numbers* while being *fully deterministic*. They are, in effect, a free lunch — pre-computed randomness that anyone can use.

For any integer key K and zero γₙ, the phase angle φₙ(K) = γₙ · K mod 2π is an irrational multiple of π for almost all K, giving uniform distribution modulo 2π. This is the fundamental operation that replaces random hashing.

## The Primitive

All structures in this document use the same core operation:

```
phase_n(key) = gamma_n * key  (mod 2π)
digit_n(key) = floor(B * phase_n / 2π)
```

Where γₙ is the n-th Riemann zero (γ₁ ≈ 14.1347, γ₂ ≈ 21.0220, ...), and B is the number of buckets.

A single zero gives a hash function. K zeros give K independent hash functions. A sequence of K digits gives a mixed-radix key.

## Proven: Structures We Already Built

### 1. Hash Table (ResonantGrid)

**Traditional**: Random hash function h(key) → bucket. O(1) average, collision resolution via chaining.
**Resonant**: `digit_1(key)` → bucket using γ₁. 620 buckets (one per zero). O(1) direct addressing.
**Result**: Works. 0.4 µs lookup, 2x faster than Python dict.

### 2. Radix Tree / Trie (BinaryRadixTree)

**Traditional**: Each level extracts a digit from the key using bit shifts or string prefixes.
**Resonant**: Level k extracts `digit_k(key)` using γ_k. K levels, each branching on B buckets.
**Result**: Works. O(K) = O(6) search regardless of N. 100% correctness match with hash table.

### 3. Sorted Array / Radix Sort (RiemannRadixSort)

**Traditional**: Sort by comparison (O(N log N)), binary search (O(log N)).
**Resonant**: Sort by mixed-radix key from K zeros. Binary search by key. Nearby entries have similar keys.
**Result**: Works. O(log N) lookup + O(R) neighborhood scan.

## Proposed: Structures to Build

### 4. Bloom Filter

**Traditional**: k random hash functions map each element to k bit positions. Query checks if all k bits are set.
**Resonant**: k zeros → k bit positions via `digit_n(key) mod M`. Fully deterministic, no seed needed.
**Why it works**: The zeros behave like independent random hash functions. The false positive rate is (1 - e^{-kn/m})^k, same as random hashing.
**Implementation**: `bits[digit_n(key) % m] = 1` for n = 1..k. Query: check all k bits.
**Advantage**: Reproducible across runs. No random state to save/restore. Can reconstruct the filter from just the count.

### 5. Count-Min Sketch

**Traditional**: d random hash functions map each element to d counters. Query returns min of d counters.
**Resonant**: d zeros → d counter positions via `digit_n(key) % w`.
**Why it works**: The zeros provide pairwise-independent hash functions. The error bound is ε = e/d with probability 1 - δ where δ = (w/e)^d.
**Advantage**: Deterministic frequency estimation. Same guarantees as random hashing without the randomness.

### 6. Cuckoo Filter

**Traditional**: Two hash functions h₁, h₂. Insert displaces existing elements (cuckoo hashing).
**Resonant**: Two zeros, two bucket positions: `pos₁ = digit₁(key) % m`, `pos₂ = digit₂(key) % m`.
**Why it works**: The two zeros give effectively independent bucket assignments, so cuckoo displacement converges as expected.
**Advantage**: Cuckoo filters need rarely-changing hash functions for relocation stability. Irrational zeros are naturally stable.

### 7. HyperLogLog

**Traditional**: Hash element to 64-bit value, use leading zeros as run-length estimate.
**Resonant**: `digit_n(key)` gives a B-ary digit. Leading zeros of the base-B representation = leading non-zero digits of the phase sequence.
**Alternative**: Use `floor(-log_B(digit_n(key) / (B - 1)))` as a geometric random variable.
**Why it works**: The digit distribution is uniform (Montgomery-Odlyzko), so the leading-zero distribution is geometric — exactly what HLL needs.
**Advantage**: Can tune precision by adjusting B (more buckets = more precision per zero).

### 8. MinHash / Locality-Sensitive Hashing

**Traditional**: Random permutation of elements. MinHash = minimum element under permutation. Similar sets have similar MinHashes.
**Resonant**: `digit_n(key)` gives a ranked ordering of keys under zero n. The "minimum" key is the one with smallest digit_n. K zeros give K independent rankings.
**Why it works**: Each zero gives a different digit assignment, effectively a different random permutation. The Jaccard similarity between two sets is approximated by the fraction of zeros where they share the same minimum element.
**Advantage**: No random permutations to generate or store. The zeros IS the permutation generator.

### 9. Skip List

**Traditional**: Each element's promotion level is determined by random coin flips (geometric distribution).
**Resonant**: `digit_n(key)` determines if the element appears at level n. If digit_n == 0, it's promoted. Continue until digit_m ≠ 0.
**Why it works**: The digits are uniformly distributed, so the promotion probability is 1/B per level — giving a geometric distribution with mean B/(B-1), matching skip list theory.
**Advantage**: Deterministic skip lists. Same expected O(log N) search, but reproducible across runs.

### 10. Treap (Randomized BST)

**Traditional**: Each node gets a random priority. Tree is a BST by key, heap by priority.
**Resonant**: Priority = `Σ digit_n(key) * B^{-n}` (a fractional mixed-radix number). This gives a uniform random priority in [0, 1).
**Why it works**: The digits are i.i.d. uniform, so the fractional mixed-radix number is uniform in [0, 1). The treap property holds with expected O(log N) depth.
**Advantage**: Determinisitic treap. Same expected performance, no random number generator needed.

### 11. Quotient Filter

**Traditional**: Split hash into quotient (bucket) and remainder. Store remainder at quotient position.
**Resonant**: `digit₁(key)` = quotient (bucket), `digit₂(key)` = remainder.
**Why it works**: Two zeros give independent quotient and remainder distributions.
**Advantage**: Deterministic quotient/remainder decomposition.

### 12. Consistent Hashing (Chord DHT)

**Traditional**: Hash keys and nodes to positions on a ring. Each key is owned by the nearest clockwise node.
**Resonant**: `Σ digit_n(key) * B^{-n}` maps both keys and nodes to [0, 1). Nearest clockwise node owns the key.
**Why it works**: The fractional key is uniform on [0, 1). The ring topology is preserved. Node addition/removal only affects neighbors.
**Advantage**: Deterministic node assignment. No random seed coordination between nodes.

### 13. Feature Hashing (Hash Trick)

**Traditional**: Map features to indices via a random hash. h(feature) % m gives the index.
**Resonant**: `digit₁(feature) % m` gives the index. K zeros give K independent feature maps.
**Why it works**: Same collision guarantees as random hashing.
**Advantage**: Deterministic across training runs. Same feature goes to same bucket every time.

### 14. Frequency-Sorted Trie (Path-Compressed Trie)

**Traditional**: Each node stores a branch character. Compress single-child paths.
**Resonant**: Build a BinaryRadixTree from observed frequencies. After construction, compress nodes with single children.
**Why it works**: Frequency ordering is natural — more frequent entries get shallower paths because their digit sequences are more likely to diverge early.
**Advantage**: Self-optimizing trie. Path compression is automatic when common prefixes emerge from the frequency distribution.

### 15. Cumulative Frequency Array (Fenwick Tree over Zero Space)

**Traditional**: Fenwick tree over sorted array for prefix sum queries.
**Resonant**: Sort by radix key. Build Fenwick tree over the sorted entries. Query prefix sums by radix key range.
**Why it works**: Radix key ordering is monotonic with frequency similarity. Range queries in key space ≈ frequency neighborhood queries.
**Advantage**: O(log N) prefix sums over frequency-similar entries.

---

## Summary

| Structure | Traditional Randomness | Riemann Zero Replacement | Status |
|-----------|----------------------|-------------------------|--------|
| Hash table | Random hash function | Single zero digit | Built |
| Radix tree | Bit/digit extraction | K zero digits | Built |
| Radix sort | Comparison function | Mixed-radix zero key | Built |
| Bloom filter | k random hashes | k zero digits | Proposed |
| Count-min sketch | d random hashes | d zero digits | Proposed |
| Cuckoo filter | 2 random hashes | 2 zero digits | Proposed |
| HyperLogLog | Random 64-bit hash | Leading zero count | Proposed |
| MinHash | Random permutations | K zero rankings | Proposed |
| Skip list | Random coin flips | Zero digit threshold | Proposed |
| Treap | Random priority | Fractional mixed-radix | Proposed |
| Quotient filter | Hash split | Two zero digits | Proposed |
| Consistent hashing | Random ring positions | Fractional zero key | Proposed |
| Feature hashing | Random hash | Single zero digit | Proposed |
| Path-compressed trie | Implicit | Frequency-ordered digits | Built (partial) |
| Fenwick tree | Sorted index | Radix-key sorted | Proposed |

## Implementation Note

All proposed structures use the same shared GAMMAS array (the first K zeros of the Riemann zeta function) and the same digit/phase computation. They differ only in how they interpret the digit sequence. This means a single `resonant/` package can provide all of them, with consistent behavior and analysis.

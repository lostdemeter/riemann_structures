"""ResonantMinHash demo."""
from resonant.minhash import ResonantMinHash

a = ResonantMinHash(n_zeros=128)
a.update(["apple", "banana", "cherry"])

b = ResonantMinHash(n_zeros=128)
b.update(["banana", "cherry", "date"])

print(f"J(A,B): {a.jaccard(b):.3f}  (true: 2/4 = 0.500)")

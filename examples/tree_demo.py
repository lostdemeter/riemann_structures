"""BinaryRadixTree demo — dict interface + nearby search."""
from resonant import BinaryRadixTree

t = BinaryRadixTree(n_zeros=6, n_buckets=8)

# Insert some bigrams
for phrase, expected in [
    ((464, 2068), 7586),   # "The quick" → " brown"
    ((2068, 7586), 21831), # " quick brown" → " fox"
    ((7586, 21831), 18045),# " brown fox" → " jumps"
]:
    t[phrase] = {expected: 1}

print(f"t = {t}")
print(f"t[464, 2068] = {t[464, 2068]}")
print(f"len(t) = {len(t)}")

# Dict methods
print(f"get: {t.get(464, 2068)}")
print(f"get (missing): {t.get(999, 999)}")
print(f"contains: {(464, 2068) in t}, {(999, 999) in t}")

# Iteration
for entry in t:
    print(f"  entry: {entry}")

# Nearby (entries with similar digit sequences)
print(f"\nNearby (464, 2068):")
for addr, values in t.nearby(464, 2068, radius=2):
    print(f"  address={addr} values={values}")

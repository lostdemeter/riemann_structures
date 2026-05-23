"""RiemannRadixSort demo — sorted key-value store."""
from resonant import RiemannRadixSort

r = RiemannRadixSort(n_zeros=6, n_buckets=8)

# Dict-like interface
r[464, 2068] = {7586: 1}
r[2068, 7586] = {21831: 1}
r[7586, 21831] = {18045: 1}

print(f"r = {r}")
print(f"r[464, 2068] = {r[464, 2068]}")
print(f"len(r) = {len(r)}")

# Delete
del r[464, 2068]
print(f"After del: len = {len(r)}, contains (464,2068) = {(464,2068) in r}")

# Re-add
r[464, 2068] = {7586: 1}

# Sorted iteration
print("\nSorted entries:")
for keys, cell in r.items():
    print(f"  {keys} → {cell}")

# Nearby (sorted adjacency)
print(f"\nNearby (464, 2068, radius=3):")
for near in r.nearby(464, 2068, radius=3):
    print(f"  address={near['address']} dist={near['key_distance']} values={near['values']}")

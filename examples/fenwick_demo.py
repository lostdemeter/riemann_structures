"""ResonantFenwickTree demo."""
from resonant.fenwick import ResonantFenwickTree

ft = ResonantFenwickTree()
for k in range(10):
    ft.add(k, weight=k + 1)

print(ft)
print(f"Total: {ft.sum()}")
print(f"by_rank(5): {ft.by_rank(5)} (first 5 entries, 1-indexed)")
print(f"range_by_rank(3, 7): {ft.range_by_rank(3, 7)}")
print(f"rank_of key 5: {ft.rank_of(5)} (position in frequency-sorted order)")

"""ResonantField demo — frequency-domain similarity search."""
from resonant import ResonantField

f = ResonantField(n_zeros=20)

# Insert bigram observations
for a, b, c, count in [
    (464, 2068, 7586, 3),
    (2068, 7586, 21831, 2),
    (7586, 21831, 18045, 1),
    (464, 2068, 999, 1),  # another possible continuation
]:
    f.merge(a, b, value=c, count=count)

print(f"f = {f}")

# Query
results = f.query(464, 2068)
print(f"\nQuery (464, 2068): {len(results)} matches")
for cell, score in results:
    print(f"  cell={cell} resonance={score:.3f}")

# Direct access via merge
f.merge(464, 2068, value=7586, count=1)
results = f.query(464, 2068)
print(f"\nAfter merge: {len(results)} matches")
for cell, score in results:
    print(f"  cell={cell} resonance={score:.3f}")

"""ResonantGrid demo — dictionary-like usage."""
from resonant import ResonantGrid

# Create a 2D grid (for bigram storage)
g = ResonantGrid(ndim=2)

# Dict-like set/get
g[464, 2068] = {7586: 1}   # store "The quick" → " brown"
g[2068, 7586] = {21831: 1}  # " quick brown" → " fox"

print(f"g = {g}")
print(f"g[464, 2068] = {g[464, 2068]}")
print(f"(464, 2068) in g = {(464, 2068) in g}")
print(f"len(g) = {len(g)}")

# Iteration
for keys, cell in g.items():
    print(f"  {keys} → {cell}")

# Update with merge function
g.update(464, 2068, merge_fn=lambda c: c.update({999: 2}))
print(f"After merge: {g[464, 2068]}")

# Copy and clear
g2 = g.copy()
g.clear()
print(f"After clear: len(g) = {len(g)}, copy len = {len(g2)}")

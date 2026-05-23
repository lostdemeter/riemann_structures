"""ResonantCountMinSketch demo."""
from resonant.count_min import ResonantCountMinSketch

cms = ResonantCountMinSketch(error=0.01, confidence=0.99)

data = ["apple", "banana", "apple", "cherry", "apple", "banana"]
for w in data:
    cms.add(w)

print(f"Sketch: {cms}")
for w in set(data):
    print(f"  '{w}': true={data.count(w)}, est={cms[w]}")

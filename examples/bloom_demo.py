"""ResonantBloomFilter demo."""
from resonant.bloom import ResonantBloomFilter

bf = ResonantBloomFilter(capacity=100, error_rate=0.01)
words = ["apple", "banana", "cherry", "date", "elderberry"]
for w in words:
    bf.add(w)

print(f"Bloom filter: {bf}")
for w in words + ["fig", "grape", "honeydew"]:
    print(f"  '{w}' in filter: {w in bf}")

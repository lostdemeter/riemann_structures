"""ResonantCuckooFilter demo."""
from resonant.cuckoo import ResonantCuckooFilter

cf = ResonantCuckooFilter(capacity=20, fingerprint_bits=4)
fruits = ["apple", "banana", "cherry", "date", "elderberry"]
for f in fruits:
    cf.add(f)

print(cf)
for f in fruits + ["fig", "grape"]:
    print(f"  '{f}' in filter: {f in cf}")

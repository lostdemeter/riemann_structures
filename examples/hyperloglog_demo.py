"""ResonantHyperLogLog demo."""
from resonant.hyperloglog import ResonantHyperLogLog

hll = ResonantHyperLogLog(precision=12)
fruits = ["apple", "banana", "apple", "cherry", "banana", "date"]
for f in fruits:
    hll.add(f)

print(f"Distinct fruits: {len(hll)} (true: {len(set(fruits))})")
print(f"Estimate: {hll.estimate():.1f}")

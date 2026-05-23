"""ResonantQuotientFilter demo."""
from resonant.quotient import ResonantQuotientFilter

qf = ResonantQuotientFilter(capacity=20, remainder_bits=4)
fruits = ["apple", "banana", "cherry", "date"]
for f in fruits:
    qf.add(f)

print(qf)
for f in fruits + ["fig"]:
    print(f"  '{f}': {f in qf}")

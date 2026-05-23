"""ResonantTreap demo."""
from resonant.treap import ResonantTreap

t = ResonantTreap()
t[5] = "five"
t[3] = "three"
t[7] = "seven"
t[1] = "one"
t[5] = "FIVE"  # update

print(f"t[5] = {t[5]}")
print(f"Keys in order: {list(t.keys())}")
print(f"len = {len(t)}")
del t[7]
print(f"After del 7: keys = {list(t.keys())}")

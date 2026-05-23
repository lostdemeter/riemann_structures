"""ResonantSkipList demo."""
from resonant.skiplist import ResonantSkipList

sl = ResonantSkipList(promotion_prob=0.5)
sl[3] = "three"
sl[1] = "one"
sl[4] = "four"
sl[1] = "ONE"  # update

print(f"sl[1] = {sl[1]}")
print(f"len(sl) = {len(sl)}")
print(f"Keys in order: {list(sl.keys())}")
del sl[3]
print(f"After del sl[3]: keys = {list(sl.keys())}")

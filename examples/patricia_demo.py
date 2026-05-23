"""ResonantPatriciaTrie demo."""
from resonant.patricia import ResonantPatriciaTrie

pt = ResonantPatriciaTrie()
pt.insert(5, 3, value="some value")
pt.insert(5, 7, value="other value")
print(f"pt = {pt}")
print(f"pt[5, 3] = {pt[5, 3]}")
print(f"len(pt) = {len(pt)}")
for addr, val in pt:
    print(f"  entry: addr={addr} val={val}")

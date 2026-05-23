"""ResonantConsistentHash demo."""
from resonant.consistenthash import ResonantConsistentHash

ch = ResonantConsistentHash(virtual_nodes=50)
ch.add_node("server-A")
ch.add_node("server-B")
ch.add_node("server-C")

for key in ["user:1", "user:2", "user:3", "user:4", "user:5"]:
    print(f"  {key} → {ch[key]}")

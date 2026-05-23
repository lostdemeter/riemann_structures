"""ResonantConsistentHash — consistent hashing using Riemann zero positions.

Traditional consistent hashing: hash keys and nodes to a ring [0,1).
Each key owned by nearest clockwise node. Minimal redistribution on
node join/leave.

Resonant: both keys and nodes positioned via fractional mixed-radix key.
Same method: position = Σ digit_n(key) * B^{-n-1}. Uniform in [0,1).

Usage:
    ch = ResonantConsistentHash()
    ch.add_node("server-A")
    ch.add_node("server-B")
    ch["my-key"]         # → "server-A" (or "server-B")
    ch.add_node("server-C")
    ch.remove_node("server-A")
"""
import os, sys, math, json, bisect
import numpy as np

_gammas_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..',
                 'riemann_attention', 'output', 'figures', 'scripts', 'gammas_620.npy'),
    os.path.join(os.path.dirname(__file__), '..', '..',
                 'riemann_constructive', 'attention', 'gammas_620.npy'),
]
GAMMAS = None
for p in _gammas_paths:
    full = os.path.abspath(p)
    if os.path.exists(full):
        GAMMAS = np.load(full)
        break
if GAMMAS is None:
    GAMMAS = np.array([14.134725 + i * 2.5 for i in range(200)])
NG = len(GAMMAS)


class ResonantConsistentHash:
    """Consistent hash ring with Riemann-zero positions.

    Args:
        n_zeros: Number of zeros for position. More = finer granularity.
        n_buckets: Base for mixed-radix priority. Default 256.
        virtual_nodes: Replicas per physical node (for balance). Default 1.
    """

    def __init__(self, n_zeros=6, n_buckets=256, virtual_nodes=1):
        self.k = min(n_zeros, NG)
        self.B = n_buckets
        self.V = virtual_nodes
        self.ring = []      # sorted list of (position, node_id)
        self.nodes = {}     # node_id -> [positions]

    def _position(self, key):
        """Compute position on [0, 1) from Riemann zero digits."""
        key_str = str(key)
        h = hash(key_str)
        pos = 0.0
        factor = 1.0 / self.B
        for i in range(self.k):
            phase = GAMMAS[i] * h
            d = int((phase % (2 * math.pi)) / (2 * math.pi) * self.B) % self.B
            pos += d * factor
            factor /= self.B
        return pos

    def add_node(self, node_id):
        positions = []
        for v in range(self.V):
            pos = self._position(f"{node_id}:v{v}")
            positions.append(pos)
            bisect.insort(self.ring, (pos, node_id))
        self.nodes[node_id] = positions

    def remove_node(self, node_id):
        if node_id not in self.nodes:
            return
        for pos in self.nodes[node_id]:
            # Find and remove (pos, node_id) from ring
            i = bisect.bisect_left(self.ring, (pos, node_id))
            while i < len(self.ring) and self.ring[i] == (pos, node_id):
                del self.ring[i]
                break
        del self.nodes[node_id]

    def get_node(self, key):
        """Find the node responsible for key."""
        if not self.ring:
            return None
        pos = self._position(key)
        # Find nearest clockwise node
        i = bisect.bisect_left(self.ring, (pos,))
        if i >= len(self.ring):
            i = 0  # wrap around
        return self.ring[i][1]

    def __getitem__(self, key):
        return self.get_node(key)

    def __contains__(self, node_id):
        return node_id in self.nodes

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def keys(self):
        return iter(self)

    # ─── Stats ─────────────────────────────────────────────────────────

    def key_distribution(self, n_keys=10000):
        """Count how many keys map to each node."""
        import random
        counts = {n: 0 for n in self.nodes}
        for _ in range(n_keys):
            node = self.get_node(random.random())
            if node in counts:
                counts[node] += 1
        return counts

    def stats(self):
        dist = self.key_distribution(10000) if self.nodes else {}
        if dist:
            vals = list(dist.values())
            ideal = 10000 / len(dist)
            imbalance = max(abs(v - ideal) / ideal for v in vals) if ideal > 0 else 0
        else:
            imbalance = 0
        return {
            'nodes': len(self.nodes),
            'ring_size': len(self.ring),
            'virtual_nodes': self.V,
            'imbalance': imbalance,
            'distribution': dist,
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantConsistentHash(nodes={s['nodes']}, "
                f"ring={s['ring_size']}, "
                f"imbalance={s['imbalance']:.3f}, "
                f"V={s['virtual_nodes']})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random, string

    print("=" * 60)
    print("ResonantConsistentHash Benchmark")
    print("=" * 60)

    # Build ring
    ch = ResonantConsistentHash(virtual_nodes=10)
    nodes = [f"server-{i}" for i in range(10)]
    for n in nodes:
        ch.add_node(n)

    print(f"\nRing: {ch}")
    print(f"  Positions per node: {ch.V}")

    # Query
    N = 100000
    t0 = time.time()
    for _ in range(N):
        ch.get_node(random.random())
    ms = (time.time() - t0) * 1000
    print(f"\n  Query {N}: {ms:.0f} ms ({ms/N*1000:.3f} µs)")

    # Distribution
    print(f"\n  Key distribution (10K keys):")
    for node, count in sorted(ch.key_distribution(10000).items()):
        print(f"    {node}: {count} keys ({count/10000*100:.1f}%)")

    # Node removal impact
    ch2 = ResonantConsistentHash(virtual_nodes=10)
    for n in nodes:
        ch2.add_node(n)

    keys = [random.random() for _ in range(10000)]
    before = {k: ch2.get_node(k) for k in keys}

    ch2.remove_node("server-0")
    after = {k: ch2.get_node(k) for k in keys}

    redistributed = sum(1 for k in keys if before[k] != after[k])
    print(f"\n  After removing server-0: {redistributed/len(keys)*100:.1f}% keys "
          f"redistributed (ideal for 10→9 nodes: {100/10:.1f}%)")


if __name__ == '__main__':
    run_benchmark()

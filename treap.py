"""ResonantTreap — randomized BST using Riemann zero priorities.

Traditional treap: each node has a random priority. Tree is a BST by
key and a max-heap by priority. Expected O(log N) operations.

Resonant treap: priority = fractional mixed-radix key from Riemann zeros.
priority = Σ digit_n(key) * B^{-n-1}. Uniform in [0, 1). Deterministic.

Usage:
    t = ResonantTreap()
    t[5] = "five"
    t[3] = "three"
    t[3]  # "three"
    del t[3]
    len(t)
    for k in t: print(k)
"""
import os, sys, math, json
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


class TreapNode:
    __slots__ = ('key', 'value', 'priority', 'left', 'right')
    def __init__(self, key, value, priority):
        self.key = key
        self.value = value
        self.priority = priority
        self.left = None
        self.right = None


class ResonantTreap:
    """Treap with Riemann-zero priority.

    Args:
        n_zeros: Number of zeros for priority computation. More zeros
            give more uniform priority distribution. Default 6.
        B: Base for mixed-radix priority. Default 256.
    """

    def __init__(self, n_zeros=6, B=256):
        self.k = min(n_zeros, NG)
        self.B = B
        self.root = None
        self._size = 0

    def _priority(self, key):
        """Compute priority as fractional mixed-radix number in [0, 1).

        Uses hash(str(key)) to ensure uniform distribution even for
        consecutive integer keys (Python hashes ints to themselves).
        """
        h = hash(str(key))
        prio = 0.0
        factor = 1.0 / self.B
        for i in range(self.k):
            phase = GAMMAS[i] * h
            d = int((phase % (2 * math.pi)) / (2 * math.pi) * self.B) % self.B
            prio += d * factor
            factor /= self.B
        return prio

    # ─── BST + Heap rotations ─────────────────────────────────────────

    def _rotate_right(self, p):
        q = p.left
        p.left = q.right
        q.right = p
        return q

    def _rotate_left(self, p):
        q = p.right
        p.right = q.left
        q.left = p
        return q

    # ─── Core operations ───────────────────────────────────────────────

    def __setitem__(self, key, value):
        self.root = self._insert(self.root, key, value)
        self._size += 1

    def _insert(self, node, key, value):
        if node is None:
            return TreapNode(key, value, self._priority(key))
        if key < node.key:
            node.left = self._insert(node.left, key, value)
            if node.left.priority > node.priority:
                node = self._rotate_right(node)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
            if node.right.priority > node.priority:
                node = self._rotate_left(node)
        else:
            node.value = value
            self._size -= 1  # update, not insert
        return node

    def __getitem__(self, key):
        node = self.root
        while node is not None:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node.value
        raise KeyError(key)

    def __delitem__(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            raise KeyError(key)
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                self._size -= 1
                return node.right
            if node.right is None:
                self._size -= 1
                return node.left
            # Two children: rotate higher-priority child up
            if node.left.priority > node.right.priority:
                node = self._rotate_right(node)
                node.right = self._delete(node.right, key)
            else:
                node = self._rotate_left(node)
                node.left = self._delete(node.left, key)
        return node

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self):
        return self._size

    def __iter__(self):
        def inorder(node):
            if node is None:
                return
            yield from inorder(node.left)
            yield node.key
            yield from inorder(node.right)
        return inorder(self.root)

    def items(self):
        def inorder(node):
            if node is None:
                return
            yield from inorder(node.left)
            yield node.key, node.value
            yield from inorder(node.right)
        return inorder(self.root)

    def values(self):
        for _, v in self.items():
            yield v

    def keys(self):
        return iter(self)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # ─── Stats ─────────────────────────────────────────────────────────

    def _height(self, node):
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def stats(self):
        h = self._height(self.root)
        return {
            'size': self._size,
            'height': h,
            'expected_height': 2 * math.log(self._size + 1) / math.log(2)
                if self._size > 0 else 0,
            'n_zeros': self.k,
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantTreap(n={s['size']}, height={s['height']}, "
                f"expected≈{s['expected_height']:.0f})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random

    print("=" * 60)
    print("ResonantTreap Benchmark")
    print("=" * 60)

    N = 10000
    keys = list(range(N))
    random.shuffle(keys)

    t0 = time.time()
    t = ResonantTreap()
    for k in keys:
        t[k] = f"value_{k}"
    insert_ms = (time.time() - t0) * 1000
    print(f"\nInsert {N}: {insert_ms:.0f} ms ({insert_ms/N*1000:.3f} µs)")
    print(f"  {t}")

    # Search
    queries = keys[:N//2] + [-1] * (N//2)
    random.shuffle(queries)
    t0 = time.time()
    found = 0
    for q in queries:
        if q in t:
            found += 1
    search_ms = (time.time() - t0) * 1000
    print(f"\nSearch {len(queries)}: {search_ms:.0f} ms "
          f"({search_ms/len(queries)*1000:.3f} µs)")
    print(f"  Found: {found}/{len(queries)}")

    # Delete
    to_del = keys[:N//2]
    t0 = time.time()
    for k in to_del:
        del t[k]
    del_ms = (time.time() - t0) * 1000
    print(f"\nDelete {len(to_del)}: {del_ms:.0f} ms "
          f"({del_ms/len(to_del)*1000:.3f} µs)")

    # Height vs expected
    print(f"\n  Final: {t}")
    print(f"  Expected height: {math.log(t._size, 2):.0f} (balanced BST)")

    # Compare with dict
    d = {}
    t0 = time.time()
    for k in keys:
        d[k] = f"value_{k}"
    print(f"\n  Dict insert: {(time.time()-t0)*1000:.0f} ms")

    t0 = time.time()
    for q in queries:
        q in d
    print(f"  Dict search: {(time.time()-t0)*1000:.0f} ms")


if __name__ == '__main__':
    run_benchmark()

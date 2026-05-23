"""ResonantSkipList — skip list using Riemann zero digit promotions.

Traditional skip list: each element's height determined by random coin
flips (geometric distribution). Expected O(log N) search/insert/delete.

Resonant skip list: promotion level determined by digit_n(key) threshold.
digit_n(key) < B*p → promoted to level n. Deterministic, seedless.

Expected height after insertion is geometric with mean 1/p.
Search/insert/delete: O(log N) with high probability.

Usage:
    sl = ResonantSkipList(promotion_prob=0.5)
    sl[5] = "five"
    sl[3] = "three"
    sl[7] = "seven"
    sl[5]  # "five"
    del sl[3]
    len(sl)  # 2
    for k in sl: print(k, sl[k])
"""
import os, sys, math, json, random
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

MAX_LEVEL = 64  # max possible height (limited by available zeros)


class SkipNode:
    __slots__ = ('key', 'value', 'forward')
    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)


class ResonantSkipList:
    """Skip list with Riemann-zero promotion.

    Args:
        promotion_prob: Probability of promotion per level (default 0.5).
            Controls expected height: E[height] = 1/p.
        max_level: Maximum node height. Default 32.
    """

    def __init__(self, promotion_prob=0.5, max_level=32):
        self.p = promotion_prob
        self.max_level = min(max_level, NG)
        self.header = SkipNode(None, None, self.max_level)
        self._level = 0  # current max level in use
        self._size = 0

    def _digit(self, key, idx):
        h = hash(key) if not isinstance(key, int) else key
        phase = GAMMAS[idx] * h
        return int((phase % (2 * math.pi)) / (2 * math.pi) * 256)

    def _random_level(self, key):
        """Determine node height from digit sequence.
        
        digit_n(key) < threshold → promoted to level n.
        threshold = int(256 * p). Geometric(1-p) distribution.
        """
        threshold = int(256 * self.p)
        level = 0
        for i in range(self.max_level):
            if self._digit(key, i) < threshold:
                level += 1
            else:
                break
        return level

    # ─── Core ──────────────────────────────────────────────────────────

    def _find_predecessors(self, key):
        """Find predecessors at each level for key.
        
        Returns list of nodes: update[i] = node where forward[i] >= key.
        """
        update = [None] * (self.max_level + 1)
        cur = self.header
        for i in range(self._level, -1, -1):
            while cur.forward[i] is not None and cur.forward[i].key < key:
                cur = cur.forward[i]
            update[i] = cur
        return update

    def __setitem__(self, key, value):
        update = self._find_predecessors(key)
        cur = update[0].forward[0]

        if cur is not None and cur.key == key:
            cur.value = value
            return

        level = self._random_level(key)
        if level > self._level:
            for i in range(self._level + 1, level + 1):
                update[i] = self.header
            self._level = level

        node = SkipNode(key, value, level)
        for i in range(level + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
        self._size += 1

    def __getitem__(self, key):
        cur = self.header
        for i in range(self._level, -1, -1):
            while cur.forward[i] is not None and cur.forward[i].key < key:
                cur = cur.forward[i]
        cur = cur.forward[0]
        if cur is not None and cur.key == key:
            return cur.value
        raise KeyError(key)

    def __delitem__(self, key):
        update = self._find_predecessors(key)
        cur = update[0].forward[0]
        if cur is None or cur.key != key:
            raise KeyError(key)

        for i in range(len(cur.forward)):
            if update[i].forward[i] is not cur:
                break
            update[i].forward[i] = cur.forward[i]

        while self._level > 0 and self.header.forward[self._level] is None:
            self._level -= 1
        self._size -= 1

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self):
        return self._size

    def __iter__(self):
        cur = self.header.forward[0]
        while cur is not None:
            yield cur.key
            cur = cur.forward[0]

    def items(self):
        cur = self.header.forward[0]
        while cur is not None:
            yield cur.key, cur.value
            cur = cur.forward[0]

    def values(self):
        cur = self.header.forward[0]
        while cur is not None:
            yield cur.value
            cur = cur.forward[0]

    def keys(self):
        return iter(self)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # ─── Stats ─────────────────────────────────────────────────────────

    def stats(self):
        # Compute actual levels
        levels = {}
        cur = self.header.forward[0]
        while cur is not None:
            lvl = len(cur.forward) - 1
            levels[lvl] = levels.get(lvl, 0) + 1
            cur = cur.forward[0]
        return {
            'size': self._size,
            'max_level': self._level,
            'promotion_prob': self.p,
            'level_distribution': levels,
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantSkipList(n={s['size']}, "
                f"L={s['max_level']}, p={s['promotion_prob']})")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random

    print("=" * 60)
    print("ResonantSkipList Benchmark")
    print("=" * 60)

    N = 10000

    # Insert
    keys = list(range(N))
    random.shuffle(keys)

    sl = ResonantSkipList(promotion_prob=0.25)
    t0 = time.time()
    for k in keys:
        sl[k] = f"value_{k}"
    insert_ms = (time.time() - t0) * 1000
    s = sl.stats()
    print(f"\nInsert {N}: {insert_ms:.0f} ms ({insert_ms/N*1000:.3f} µs)")
    print(f"  Max level: {s['max_level']}, "
          f"Expected: {math.log(N, 1/(1-sl.p)):.0f}")
    print(f"  Level dist: {dict(sorted(s['level_distribution'].items()))}")

    # Search
    random.shuffle(keys)
    queries = keys[:N//2] + [-1] * (N//2)  # half existing, half missing
    random.shuffle(queries)
    t0 = time.time()
    found = 0
    for q in queries:
        if q in sl:
            found += 1
    search_ms = (time.time() - t0) * 1000
    print(f"\nSearch {len(queries)}: {search_ms:.0f} ms "
          f"({search_ms/len(queries)*1000:.3f} µs)")
    print(f"  Found: {found}/{len(queries)}")

    # Delete
    to_del = keys[:N//2]
    t0 = time.time()
    for k in to_del:
        del sl[k]
    del_ms = (time.time() - t0) * 1000
    print(f"\nDelete {len(to_del)}: {del_ms:.0f} ms "
          f"({del_ms/len(to_del)*1000:.3f} µs)")
    print(f"  Size after delete: {len(sl)}")

    # Compare with dict
    print(f"\n--- Comparison with Python dict ---")
    d = {}
    t0 = time.time()
    for k in keys:
        d[k] = f"value_{k}"
    d_insert = (time.time() - t0) * 1000

    t0 = time.time()
    for q in queries:
        q in d
    d_search = (time.time() - t0) * 1000

    print(f"  Insert: skip_list {insert_ms:.0f}ms vs dict {d_insert:.0f}ms")
    print(f"  Search: skip_list {search_ms:.0f}ms vs dict {d_search:.0f}ms")


if __name__ == '__main__':
    run_benchmark()

"""BinaryRadixTree — radix tree over Riemann zero digit prefixes.

Each level branches on one digit from a Riemann-zero-derived phase.
Search walks K digits = O(K) regardless of N. Supports prefix-based
similarity search (nearby entries with similar digit sequences).

Usage:
    tree = BinaryRadixTree(n_zeros=6, n_buckets=8)
    tree[464, 2068] = {7586: 1}      # insert
    cell = tree[464, 2068]            # lookup (raises KeyError if missing)
    cell = tree.get(464, 2068)        # lookup (returns None if missing)
    del tree[464, 2068]               # delete
    (464, 2068) in tree               # membership
    len(tree)                         # number of entries
    for entry in tree:                # iterate all entries
    tree.nearby(464, 2068, radius=2)  # prefix similarity
"""
import os, sys, math, json
from collections.abc import MutableMapping

import numpy as np

# ─── Riemann zeros ─────────────────────────────────────────────────────
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
        GAMMAS = list(np.load(full))
        break
if GAMMAS is None:
    GAMMAS = [14.134725 + i * 2.5 for i in range(200)]
NG = len(GAMMAS)


class RadixNode:
    __slots__ = ('children', 'entries', 'depth')
    def __init__(self, depth=0):
        self.children = {}
        self.entries = []
        self.depth = depth


class BinaryRadixTree(MutableMapping):
    """Radix tree over Riemann zero digit sequences.

    Args:
        n_zeros: Number of zeros (tree depth K)
        n_buckets: Buckets per digit (B). Total cells = B^K.
        address_mod: Address computation modulus
    """

    def __init__(self, n_zeros=6, n_buckets=8, address_mod=65536):
        self.K = min(n_zeros, NG)
        self.B = n_buckets
        self.mod = address_mod
        self.root = RadixNode(depth=0)
        self._len = 0

    def _addr(self, *keys):
        addr = 0
        for i, key in enumerate(keys):
            addr += (int(key) % self.mod) * (self.mod ** i)
        return addr

    def _digits(self, address):
        digs = []
        for k in range(self.K):
            phase = GAMMAS[k] * address
            d = int((phase % (2 * math.pi)) / (2 * math.pi) * self.B) % self.B
            digs.append(d)
        return digs

    def _resolve(self, *keys):
        """Walk to the leaf node. Returns (node, digs, addr) or raises KeyError."""
        addr = self._addr(*keys)
        digs = self._digits(addr)
        node = self.root
        for d in digs:
            if d not in node.children:
                raise KeyError(keys)
            node = node.children[d]
        return node, digs, addr

    def _ensure(self, *keys):
        """Walk to the leaf node, creating nodes as needed."""
        addr = self._addr(*keys)
        digs = self._digits(addr)
        node = self.root
        for d in digs:
            if d not in node.children:
                node.children[d] = RadixNode(depth=node.depth + 1)
            node = node.children[d]
        return node, addr

    # ─── MutableMapping interface ─────────────────────────────────────

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        node, _, addr = self._resolve(*keys)
        for e_addr, values in node.entries:
            if e_addr == addr:
                return dict(values)
        raise KeyError(keys)

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        node, addr = self._ensure(*keys)
        for e in node.entries:
            if e[0] == addr:
                old_values = e[1]
                if isinstance(value, dict):
                    e[1] = dict(value)
                else:
                    e[1][value] = e[1].get(value, 0) + 1
                return
        if isinstance(value, dict):
            node.entries.append([addr, dict(value)])
        else:
            node.entries.append([addr, {value: 1}])
        self._len += 1

    def __delitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        node, _, addr = self._resolve(*keys)
        for i, (e_addr, _) in enumerate(node.entries):
            if e_addr == addr:
                del node.entries[i]
                self._len -= 1
                return
        raise KeyError(keys)

    def __contains__(self, keys):
        try:
            self._resolve(*keys)
            return True
        except KeyError:
            return False

    def __len__(self):
        return self._len

    def __iter__(self):
        def dfs(node):
            for e_addr, values in node.entries:
                yield e_addr, dict(values)
            for child in node.children.values():
                yield from dfs(child)
        return dfs(self.root)

    # ─── Extra API ─────────────────────────────────────────────────────

    def get(self, *keys, default=None):
        try:
            return self[keys]
        except KeyError:
            return default

    def insert(self, *keys, value, count=1):
        """Increment count for value at keys (bigram-style)."""
        if not isinstance(keys, tuple):
            keys = (keys,)
        node, addr = self._ensure(*keys)
        for e in node.entries:
            if e[0] == addr:
                e[1][value] = e[1].get(value, 0) + count
                return
        node.entries.append([addr, {value: count}])
        self._len += 1

    def nearby(self, *keys, radius=2):
        """Find entries with similar digit sequences.

        Walks up the tree: matching prefixes of length K, K-1, ..., K-radius.
        Entries sharing a longer prefix have more similar frequency signatures.
        """
        addr = self._addr(*keys)
        digs = self._digits(addr)
        results = []

        def collect(node):
            for e_addr, values in node.entries:
                if e_addr != addr:
                    results.append((e_addr, dict(values)))

        for prefix_len in range(self.K, self.K - radius - 1, -1):
            node = self.root
            for k in range(min(prefix_len, self.K)):
                if digs[k] not in node.children:
                    node = None
                    break
                node = node.children[digs[k]]
            if node:
                def dfs(n):
                    collect(n)
                    for c in n.children.values():
                        dfs(c)
                dfs(node)
            if len(results) >= 10:
                break
        return results

    def stats(self):
        def count_nodes(node):
            n = 1
            for c in node.children.values():
                n += count_nodes(c)
            return n
        return {
            'entries': self._len,
            'nodes': count_nodes(self.root),
            'depth': self.K,
            'buckets': self.B,
        }

    def save(self, path):
        def node_to_dict(node):
            return {
                'entries': [[e[0], dict(e[1])] for e in node.entries],
                'children': {str(d): node_to_dict(c) for d, c in node.children.items()},
            }
        data = {'K': self.K, 'B': self.B, 'mod': self.mod, '_len': self._len,
                'root': node_to_dict(self.root)}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        from collections import defaultdict
        with open(path) as f:
            data = json.load(f)
        tree = cls(n_zeros=data['K'], n_buckets=data['B'], address_mod=data['mod'])
        tree._len = data['_len']

        def dict_to_node(d):
            node = RadixNode()
            node.entries = [[e[0], {int(k): int(v) for k, v in e[1].items()}]
                           for e in d['entries']]
            node.children = {int(ds): dict_to_node(cd)
                           for ds, cd in d['children'].items()}
            return node

        tree.root = dict_to_node(data['root'])
        return tree

    def __repr__(self):
        s = self.stats()
        return (f"BinaryRadixTree(K={s['depth']}, B={s['buckets']}, "
                f"entries={s['entries']}, nodes={s['nodes']})")

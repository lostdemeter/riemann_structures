"""ResonantPatriciaTrie — path-compressed trie over Riemann zero digits.

Standard trie: each level = one digit.
Path-compressed trie: consecutive single-child nodes merge into one path.
Fewer nodes, shallower depth, faster search.

Each node stores a compressed path (list of digits). Single-child paths
are merged into the parent during insertion.

Usage:
    pt = ResonantPatriciaTrie()
    pt[464, 2068] = {7586: 1}
    cell = pt[464, 2068]
    len(pt)
    for entry in pt: ...
"""
import os, sys, math, json
from collections.abc import MutableMapping

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


class PatriciaNode:
    __slots__ = ('path', 'children', 'entries')
    def __init__(self, path=None):
        self.path = path or []  # list of digits from parent to this node
        self.children = {}      # first_digit -> PatriciaNode
        self.entries = []       # [(address, {value: count})]


class ResonantPatriciaTrie(MutableMapping):
    """Path-compressed trie over Riemann zero digit sequences.

    Args:
        n_zeros: Tree depth (K). Default 6.
        n_buckets: Digits per level (B). Default 8.
        address_mod: Address computation modulus.
    """

    def __init__(self, n_zeros=6, n_buckets=8, address_mod=65536):
        self.K = min(n_zeros, NG)
        self.B = n_buckets
        self.mod = address_mod
        self.root = PatriciaNode()
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

    # ─── Insert ────────────────────────────────────────────────────────

    def insert(self, *keys, value, count=1):
        addr = self._addr(*keys)
        digs = self._digits(addr)
        self._insert_into(self.root, digs, 0, addr, value, count)

    def _insert_into(self, node, digs, depth, addr, value, count):
        if depth >= len(digs):
            # Reached leaf depth — store/update entry
            for e in node.entries:
                if e[0] == addr:
                    e[1][value] = e[1].get(value, 0) + count
                    return
            node.entries.append([addr, {value: count}])
            self._len += 1
            return

        # No children — create one
        if not node.children:
            child = PatriciaNode(digs[depth:])
            self._insert_into(child, digs, len(digs), addr, value, count)
            node.children[digs[depth]] = child
            return

        # Check if first digit matches any child path
        first_d = digs[depth]
        if first_d in node.children:
            child = node.children[first_d]
            cp = child.path

            # Find shared prefix length
            shared = 0
            for i in range(min(len(cp), len(digs) - depth)):
                if cp[i] == digs[depth + i]:
                    shared += 1
                else:
                    break

            if shared == len(cp):
                # Full path match — continue into child
                self._insert_into(child, digs, depth + shared, addr, value, count)
            else:
                # Partial match — split the child
                # Create split node with shared prefix
                split = PatriciaNode(cp[:shared])
                # Child keeps its remaining path
                child.path = cp[shared:]
                split.children[child.path[0]] = child
                # Create new sibling for the remaining query digits
                remaining = digs[depth + shared:]
                if remaining:
                    new_child = PatriciaNode(remaining)
                    self._insert_into(new_child, digs, len(digs), addr, value, count)
                    split.children[remaining[0]] = new_child
                else:
                    # Query ended at this node
                    split.entries.append([addr, {value: count}])
                    self._len += 1

                node.children[first_d] = split
        else:
            # New child with remaining digits as path
            child = PatriciaNode(digs[depth:])
            self._insert_into(child, digs, len(digs), addr, value, count)
            node.children[first_d] = child

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if isinstance(value, dict):
            for k, v in value.items():
                self.insert(*keys, value=k, count=v)
        else:
            self.insert(*keys, value=value)

    # ─── Lookup ────────────────────────────────────────────────────────

    def _find(self, *keys):
        addr = self._addr(*keys)
        digs = self._digits(addr)
        return self._find_in(self.root, digs, 0, addr)

    def _find_in(self, node, digs, depth, addr):
        if depth >= len(digs):
            # Check entries at this node
            for e_addr, values in node.entries:
                if e_addr == addr:
                    return dict(values)
            return None

        first_d = digs[depth]
        if first_d not in node.children:
            return None

        child = node.children[first_d]
        cp = child.path

        # Try to match path
        if len(digs) - depth < len(cp):
            return None
        for i in range(len(cp)):
            if cp[i] != digs[depth + i]:
                return None

        return self._find_in(child, digs, depth + len(cp), addr)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = self._find(*keys)
        if result is None:
            raise KeyError(keys)
        return result

    def __contains__(self, keys):
        try:
            self[keys]
            return True
        except KeyError:
            return False

    def __delitem__(self, keys):
        raise NotImplementedError("Delete not yet implemented for PatriciaTrie")

    def __len__(self):
        return self._len

    def __iter__(self):
        def dfs(node):
            for e_addr, values in node.entries:
                yield e_addr, dict(values)
            for child in node.children.values():
                yield from dfs(child)
        return dfs(self.root)

    def get(self, *keys, default=None):
        try:
            return self[keys]
        except KeyError:
            return default

    def stats(self):
        def node_count(node):
            n = 1
            for c in node.children.values():
                n += node_count(c)
            return n

        def path_stats(node, depth=0, paths=None):
            if paths is None:
                paths = []
            if not node.children and node.entries:
                paths.append(depth + len(node.path) if node.path else depth)
            for c in node.children.values():
                path_stats(c, depth + len(node.path or []), paths)
            return paths

        total = node_count(self.root)
        depths = path_stats(self.root)
        return {
            'entries': self._len,
            'nodes': total,
            'compression_ratio': self.K * self._len / total if total else 0,
            'leaf_depths': depths,
            'avg_leaf_depth': sum(depths) / len(depths) if depths else 0,
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantPatriciaTrie(n={s['entries']}, "
                f"nodes={s['nodes']}, "
                f"avg_depth={s['avg_leaf_depth']:.1f}, "
                f"compress={s['compression_ratio']:.1f}x)")


# ─── Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    import time, random
    from text.radix_tree import BinaryRadixTree

    print("=" * 60)
    print("ResonantPatriciaTrie vs BinaryRadixTree")
    print("=" * 60)

    # Build small tree
    pt = ResonantPatriciaTrie(n_zeros=6, n_buckets=8)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    text = "the quick brown fox jumps over the lazy dog"
    toks = enc.encode(text)
    for i in range(len(toks) - 2):
        pt.insert(toks[i], toks[i + 1], value=toks[i + 2])

    print(f"\nPatriciaTrie: {pt}")

    # Full corpus
    from knowledge.corpus import CorpusStore
    corpus = CorpusStore()
    pages = corpus.list_pages()[:200]

    t0 = time.time()
    pt2 = ResonantPatriciaTrie(n_zeros=6, n_buckets=8)
    bt = BinaryRadixTree(n_zeros=6, n_buckets=8)
    for title in pages:
        if title in corpus.pages and corpus.pages[title].raw_text:
            toks = enc.encode(corpus.pages[title].raw_text[:2000])
            for i in range(len(toks) - 2):
                pt2.insert(toks[i], toks[i + 1], value=toks[i + 2])
                bt.insert(toks[i], toks[i + 1], value=toks[i + 2])
    build_ms = (time.time() - t0) * 1000
    print(f"\nBuild {len(pages)} pages: {build_ms:.0f} ms")
    print(f"  Patricia: {pt2.stats()}")
    print(f"  Binary:   entries={bt.stats()['entries']}, "
          f"nodes={bt.stats()['nodes']}")

    # Correctness
    queries = []
    for title in pages[:10]:
        if title in corpus.pages and corpus.pages[title].raw_text:
            toks = enc.encode(corpus.pages[title].raw_text[:200])
            for i in range(min(10, len(toks) - 2)):
                queries.append((toks[i], toks[i + 1]))
    correct = 0
    for a, b in queries:
        pv = pt2.get(a, b) or {}
        bv = bt.find(a, b) or {}
        if pv == bv:
            correct += 1
    print(f"\nCorrectness: {correct}/{len(queries)}")

    # Speed
    import random as rnd
    rnd.shuffle(queries)
    N = 500
    t0 = time.time()
    for _ in range(N):
        for a, b in queries:
            pt2.get(a, b)
    pt_ms = (time.time() - t0) * 1000
    t0 = time.time()
    for _ in range(N):
        for a, b in queries:
            bt.find(a, b)
    bt_ms = (time.time() - t0) * 1000
    print(f"\nLookup ({len(queries)} queries, {N} passes):")
    print(f"  Patricia: {pt_ms:.0f} ms ({pt_ms/(N*len(queries))*1000:.3f} µs)")
    print(f"  Binary:   {bt_ms:.0f} ms ({bt_ms/(N*len(queries))*1000:.3f} µs)")


if __name__ == '__main__':
    run_benchmark()

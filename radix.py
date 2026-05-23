"""RiemannRadixSort — sorted array indexed by Riemann zero radix key.

O(log N) binary search by radix key. Adjacent entries have similar
frequency signatures. Supports dict-like interface + nearby().

Usage:
    r = RiemannRadixSort()
    r[464, 2068] = {7586: 1}
    cell = r[464, 2068]
    del r[464, 2068]
    len(r)
    for keys, cell in r.items():
    r.nearby(464, 2068, radius=5)
"""
import os, sys, math, json, bisect
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


class RiemannRadixSort(MutableMapping):
    """Sorted kv-store indexed by Riemann-zero radix key.

    Args:
        n_zeros: Radix key digits (K)
        n_buckets: Buckets per digit (B)
        address_mod: Address modulus
    """

    def __init__(self, n_zeros=6, n_buckets=8, address_mod=65536):
        self.K = min(n_zeros, NG)
        self.B = n_buckets
        self.mod = address_mod
        self._entries = []
        self._keys = []
        self._dirty = False

    def _addr(self, *keys):
        addr = 0
        for i, key in enumerate(keys):
            addr += (int(key) % self.mod) * (self.mod ** i)
        return addr

    def _radix_key(self, address):
        key = 0
        for k in range(self.K - 1, -1, -1):
            phase = GAMMAS[k] * address
            d = int((phase % (2 * math.pi)) / (2 * math.pi) * self.B) % self.B
            key = key * self.B + d
        return key

    def _sort(self):
        if self._dirty and len(self._entries) > 1:
            self._entries.sort(key=lambda e: self._radix_key(e[0]))
            self._dirty = False
        self._keys = [self._radix_key(e[0]) for e in self._entries]

    def _find(self, *keys):
        """Binary search for entry. Returns (index, addr, target_key) or raises KeyError."""
        self._sort()
        addr = self._addr(*keys)
        target = self._radix_key(addr)
        if not self._keys:
            raise KeyError(keys)
        idx = bisect.bisect_left(self._keys, target)
        # Walk collisions
        for i in range(idx, len(self._keys)):
            if self._keys[i] != target:
                break
            if self._entries[i][0] == addr:
                return i, addr, target
        for i in range(idx - 1, -1, -1):
            if self._keys[i] != target:
                break
            if self._entries[i][0] == addr:
                return i, addr, target
        raise KeyError(keys)

    # ─── MutableMapping ────────────────────────────────────────────────

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        i, _, _ = self._find(*keys)
        return dict(self._entries[i][1])

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        addr = self._addr(*keys)
        for i, (ea, _) in enumerate(self._entries):
            if ea == addr:
                self._entries[i] = (addr, dict(value) if isinstance(value, dict) else {value: 1})
                self._dirty = True
                return
        self._entries.append((addr, dict(value) if isinstance(value, dict) else {value: 1}))
        self._dirty = True

    def __delitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        i, _, _ = self._find(*keys)
        del self._entries[i]
        self._dirty = True

    def __contains__(self, keys):
        try:
            self._find(*keys)
            return True
        except KeyError:
            return False

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        self._sort()
        for ea, values in self._entries:
            yield ea, dict(values)

    # ─── Extra ─────────────────────────────────────────────────────────

    def get(self, *keys, default=None):
        try:
            return self[keys]
        except (KeyError, IndexError):
            return default

    def nearby(self, *keys, radius=5):
        self._sort()
        addr = self._addr(*keys)
        target = self._radix_key(addr)
        idx = bisect.bisect_left(self._keys, target)
        results = []
        for i in range(max(0, idx - radius), min(len(self._entries), idx + radius + 1)):
            if self._entries[i][0] != addr:
                results.append({
                    'address': self._entries[i][0],
                    'values': dict(self._entries[i][1]),
                    'key_distance': abs(self._keys[i] - target),
                })
        return sorted(results, key=lambda x: x['key_distance'])

    def items(self):
        self._sort()
        for ea, values in self._entries:
            # Decode address back to keys (lossy: mod collision)
            keys = []
            remaining = ea
            for _ in range(2):
                keys.append(remaining % self.mod)
                remaining //= self.mod
            yield tuple(keys), dict(values)

    def stats(self):
        return {'entries': len(self._entries), 'K': self.K, 'B': self.B}

    def __repr__(self):
        return f"RiemannRadixSort(K={self.K}, B={self.B}, entries={len(self._entries)})"

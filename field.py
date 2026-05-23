"""ResonantField — frequency-domain retrieval via Riemann zero basis.

Stores entries as K-dimensional complex amplitude vectors. Retrieval
via resonance detection (dot product in frequency space). Designed
for fuzzy/similarity search where exact key matching isn't enough.

Usage:
    field = ResonantField(n_zeros=20)
    field[464, 2068, 7586] += 1       # increment count
    results = field[464, 2068]         # find matching entries
    results = field.query(464, 2068)   # explicit query
"""
import os, sys, math
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


class ResonantField:
    """Frequency-domain retrieval via Riemann zero basis.

    Args:
        n_zeros: Number of Riemann zeros for signature (K)
        address_mod: Modulus for address computation
    """

    def __init__(self, n_zeros=20, address_mod=65536):
        self.K = min(n_zeros, NG)
        self.mod = address_mod
        self._entries = []
        self._signatures = None
        self._addresses = np.array([], dtype=np.int64)

    def _addr(self, *keys):
        addr = 0
        for i, key in enumerate(keys):
            addr += (int(key) % self.mod) * (self.mod ** i)
        return addr

    def _signature(self, address):
        phi = 2 * math.pi * (address % 1000000) / 1000000.0
        phases = np.array(GAMMAS[:self.K]) * phi
        return np.exp(1j * phases)

    def _index(self, addr):
        for i, ea in enumerate(self._addresses):
            if ea == addr:
                return i
        return -1

    # ─── CRUD ──────────────────────────────────────────────────────────

    def add(self, *keys, value=None, count=1):
        """Add an observation. For bigrams: add(a, b, c) stores (a,b)->c."""
        if len(keys) < 2:
            return
        addr = self._addr(*keys[:2])
        idx = self._index(addr)
        if idx >= 0:
            # Update
            if value is not None:
                cell = self._entries[idx]
                cell[value] = cell.get(value, 0) + count
            return
        # New entry
        cell = {keys[2]: count} if len(keys) >= 3 and value is None else ({value: count} if value is not None else {})
        self._addresses = np.append(self._addresses, np.int64(addr))
        self._entries.append(cell)
        sig = self._signature(addr)
        if self._signatures is None or len(self._signatures) == 0:
            self._signatures = sig.reshape(1, -1)
        else:
            self._signatures = np.vstack([self._signatures, sig.reshape(1, -1)])

    def __getitem__(self, keys):
        """Return (cell_dict, resonance_score) for matching entries."""
        if not isinstance(keys, tuple):
            keys = (keys,)
        results = self.query(*keys)
        if not results:
            raise KeyError(keys)
        return results

    def query(self, *keys, threshold=0.5):
        """Find entries resonating with keys. Returns list of (cell, score)."""
        if len(self._entries) == 0:
            return []
        addr = self._addr(*keys)
        sig = self._signature(addr)
        scores = np.abs(self._signatures @ np.conj(sig)) / self.K
        results = []
        for i in range(len(self._entries)):
            if scores[i] > threshold and self._entries[i]:
                results.append((dict(self._entries[i]), float(scores[i])))
        return sorted(results, key=lambda x: -x[1])

    def merge(self, *keys, value, count=1):
        """Atomically increment count for value at keys."""
        addr = self._addr(*keys)
        idx = self._index(addr)
        if idx >= 0:
            self._entries[idx][value] = self._entries[idx].get(value, 0) + count
        else:
            self.add(*keys, value=value, count=count)

    def stats(self):
        return {
            'entries': len(self._entries),
            'K': self.K,
            'signature_dim': self.K,
        }

    def __repr__(self):
        return f"ResonantField(K={self.K}, entries={len(self._entries)})"

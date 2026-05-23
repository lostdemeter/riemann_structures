"""ResonantGrid — N-dimensional grid indexed by resonant frequency.

Dictionary-like interface. O(1) get/set/delete via mixed-radix address
computed from key values modulo n_frequencies.

Usage:
    grid = ResonantGrid(ndim=2)
    grid[464, 2068] = {7586: 1}     # set cell
    cell = grid[464, 2068]           # get cell (returns dict)
    del grid[464, 2068]              # delete cell
    (464, 2068) in grid              # membership test
    len(grid)                        # number of non-empty cells
    for keys, cell in grid.items():  # iterate
    for cell in grid.values():       # values only
"""
import os, sys, json, math
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
        GAMMAS = np.load(full)
        break
if GAMMAS is None:
    GAMMAS = np.array([14.134725 + i * 2.5 for i in range(200)])
NG = len(GAMMAS)
DEFAULT_NF = 620


class ResonantGrid:
    """N-dimensional sparse grid indexed by key % n_frequencies.

    Args:
        ndim: Number of dimensions (1=dict, 2=bigram, 3=trigram)
        n_frequencies: Address modulus. Default 620 (Riemann zero count)
    """

    def __init__(self, ndim=2, n_frequencies=DEFAULT_NF):
        self.ndim = ndim
        self.nf = n_frequencies
        self.size = n_frequencies ** ndim
        self.grid = np.empty(self.size, dtype=object)
        self._used = 0

    def _addr(self, *keys):
        if len(keys) != self.ndim:
            raise KeyError(f"Expected {self.ndim} keys, got {len(keys)}")
        addr = 0
        for i, key in enumerate(keys):
            addr += (int(key) % self.nf) * (self.nf ** i)
        return addr

    def _cell(self, *keys):
        addr = self._addr(*keys)
        cell = self.grid[addr]
        if cell is None:
            cell = {}
            self.grid[addr] = cell
            self._used += 1
        return cell

    # ─── Dict interface ────────────────────────────────────────────────

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cell = self.grid[self._addr(*keys)]
        if cell is None:
            raise KeyError(keys)
        return cell

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        addr = self._addr(*keys)
        old = self.grid[addr]
        if old is None and value:
            self._used += 1
        elif old is not None and not value:
            self._used -= 1
        self.grid[addr] = value if value else None

    def __delitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        addr = self._addr(*keys)
        if self.grid[addr] is not None:
            self._used -= 1
        self.grid[addr] = None

    def __contains__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cell = self.grid[self._addr(*keys)]
        return cell is not None and bool(cell)

    def __len__(self):
        return self._used

    def __iter__(self):
        for addr in range(self.size):
            cell = self.grid[addr]
            if cell is not None and cell:
                keys = []
                remaining = addr
                for _ in range(self.ndim):
                    keys.append(remaining % self.nf)
                    remaining //= self.nf
                yield tuple(keys)

    def keys(self):
        for k in self:
            yield k

    def values(self):
        for addr in range(self.size):
            cell = self.grid[addr]
            if cell is not None and cell:
                yield cell

    def items(self):
        for addr in range(self.size):
            cell = self.grid[addr]
            if cell is not None and cell:
                keys = []
                remaining = addr
                for _ in range(self.ndim):
                    keys.append(remaining % self.nf)
                    remaining //= self.nf
                yield tuple(keys), cell

    def get(self, *keys, default=None):
        cell = self.grid[self._addr(*keys)]
        if cell is None:
            return default
        return cell

    def update(self, *keys, merge_fn):
        """Apply merge_fn(cell_dict) to cell at keys. Cell is auto-created."""
        addr = self._addr(*keys)
        cell = self.grid[addr]
        if cell is None:
            cell = {}
            self.grid[addr] = cell
            self._used += 1
        merge_fn(cell)

    def clear(self):
        self.grid[:] = None
        self._used = 0

    def copy(self):
        g = ResonantGrid(ndim=self.ndim, n_frequencies=self.nf)
        for keys, cell in self.items():
            g[keys] = dict(cell)
        return g

    # ─── Serialization ─────────────────────────────────────────────────

    def to_dict(self):
        data = {}
        for keys, cell in self.items():
            addr = self._addr(*keys)
            data[str(addr)] = dict(cell)
        return {'ndim': self.ndim, 'nf': self.nf, 'data': data}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data):
        g = cls(ndim=data['ndim'], n_frequencies=data['nf'])
        for addr_str, cell in data['data'].items():
            addr = int(addr_str)
            if cell:
                g.grid[addr] = {int(k): int(v) for k, v in cell.items()}
                g._used += 1
        return g

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def stats(self):
        return {
            'ndim': self.ndim,
            'n_frequencies': self.nf,
            'cells': self.size,
            'used': self._used,
            'total_entries': sum(len(c) for c in self.grid if c is not None),
        }

    def __repr__(self):
        s = self.stats()
        return (f"ResonantGrid(ndim={s['ndim']}, nf={s['n_frequencies']}, "
                f"cells={s['cells']}, used={s['used']})")

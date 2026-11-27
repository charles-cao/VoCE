"""VoCE: Voronoi Consistency Ensemble for multi-view anomaly detection.

In every round, psi samples are drawn as anchors shared by all views and every view is split into the Voronoi
cells of these anchors (cells are defined on the computed distances; tied samples belong to all their nearest
anchors). For a sample x and round r:
  Shared-Anchor Voronoi Consistency   I_r(x) = 0 if the tie sets A_1(x), ..., A_V(x) share an anchor, else 1
  Cell-Normalized Transfer            s_v(x, a) = d_v(x, a) / rho_v(a) * mu_v(a) / G_v      (regular cells)
                                      T_r(x) = mean over u of  min over a in A_u(x) of  max over v of s_v(x, a)
  score(x) = mean over rounds of T_r(x) + I_r(x)
rho_v(a) is the cell radius, mu_v(a) the mean member distance and G_v the largest mu_v over cells with at least two
members in the round. Degenerate cells: s = 1 if the cell has at most one member; s = 1[d > 0] if it has two or more
members and radius 0. Distances are computed once per pair of distinct rows (float32 Gram expansion), so identical
rows have identical distances and distance exactly 0 to each other.
"""
import numbers

import numpy as np


def _check_views(views, n_views=None, dims=None):
    """Copy the views to float64 arrays and check that they are a non-empty list of 2-D, finite arrays with the
    same number of rows, and, when given, the number of views and the column counts of the fitted model."""
    if isinstance(views, np.ndarray) or not hasattr(views, '__len__') or len(views) == 0:
        raise ValueError('views must be a non-empty list of 2-D arrays, one per view')
    out = []
    for v, x in enumerate(views):
        x = np.array(x, dtype=np.float64, copy=True)
        if x.ndim != 2:
            raise ValueError(f'view {v} must be a 2-D array (samples x features), got shape {x.shape}')
        if not np.isfinite(x).all():
            raise ValueError(f'view {v} contains NaN or infinite values')
        out.append(x)
    if len({len(x) for x in out}) != 1:
        raise ValueError('all views must have the same number of rows (one row per sample)')
    if n_views is not None and len(out) != n_views:
        raise ValueError(f'the model was fitted on {n_views} views, got {len(out)}')
    if dims is not None and [x.shape[1] for x in out] != list(dims):
        raise ValueError(f'the model was fitted on views with {list(dims)} columns, got {[x.shape[1] for x in out]}')
    return out


def _dist(X, A, Xn, An):
    D = Xn[:, None] + An[None, :] - 2.0 * (X @ A.T)
    np.maximum(D, 0.0, out=D)
    return np.sqrt(D, out=D)


class VoCE:
    """Voronoi Consistency Ensemble.

    Parameters
    ----------
    psi : int, default 16
        Number of anchors sampled in each round (shared by all views).
    t : int, default 200
        Number of rounds.
    random_state : int or None
        Seed of the anchor sampling.

    Usage
    -----
    >>> model = VoCE(psi=16, t=200, random_state=0).fit(views)      # views: list of N x d_v arrays
    >>> score = model.decision_function(views)                       # higher = more anomalous
    >>> T, I = model.decision_components(views)                      # score = T + I
    """

    def __init__(self, psi=16, t=200, random_state=None):
        self.psi, self.t, self.random_state = psi, t, random_state

    @staticmethod
    def _uniq(views):
        """Per view: distinct rows U_v, the map from each row to its distinct row, and squared norms."""
        out = []
        for x in views:
            xc = np.ascontiguousarray(x + 0.0)                      # + 0.0 maps -0.0 to 0.0
            key = xc.view(np.dtype((np.void, xc.dtype.itemsize * xc.shape[1]))).ravel()
            _, first, inv = np.unique(key, return_index=True, return_inverse=True)
            U = xc[first]
            U32 = U.astype(np.float32)
            out.append((U, inv.ravel(), (U32 * U32).sum(1), U32))
        return out

    def _round_dist(self, q, v, it):
        """N x psi distances of the query rows to the anchors of round it in view v.
        Computed between distinct rows only; identical rows get identical values, identical pairs get 0."""
        U, inv, Un, U32 = q[v]
        Au, amap = np.unique(self.anchor_uid[v][it], return_inverse=True)
        A = self.q_fit[v][3][Au]
        Dq = _dist(U32, A, Un, (A * A).sum(1)).astype(np.float64)
        if q is self.q_fit:
            Dq[np.arange(len(U))[:, None] == Au[None, :]] = 0.0
        else:
            for k, a in enumerate(Au):
                Dq[(U == self.U_fit[v][a]).all(1), k] = 0.0
        return Dq[inv][:, amap.ravel()]

    def fit(self, views, indices=None):
        """Sample the anchors of every round and store the cell statistics.

        views   : list of N x d_v arrays, one per view, with the same row order.
        indices : optional list of rounds, each a sequence of distinct row indices used as anchors; it replaces
                  the random sampling (used to reproduce worked examples) and then sets the number of rounds.
        """
        views = _check_views(views)
        N, V = len(views[0]), len(views)
        for name, val in (('psi', self.psi), ('t', self.t)):
            if not isinstance(val, numbers.Integral) or val < 1:
                raise ValueError(f'{name} must be a positive integer, got {val!r}')
        if indices is None:
            if self.psi > N:
                raise ValueError(f'psi = {self.psi} anchors cannot be drawn from N = {N} samples; use psi <= N')
            rng = np.random.default_rng(self.random_state)
            self.indices = [rng.choice(N, self.psi, replace=False) for _ in range(self.t)]
        else:
            self.indices = [np.asarray(i) for i in indices]
            if not self.indices or any(i.ndim != 1 or len(i) == 0 or not np.issubdtype(i.dtype, np.integer)
                                       for i in self.indices):
                raise ValueError('indices must be a non-empty list of non-empty 1-D integer arrays')
            if len({len(i) for i in self.indices}) != 1:
                raise ValueError('every round of indices must have the same number of anchors')
            if any(i.min() < 0 or i.max() >= N or len(np.unique(i)) != len(i) for i in self.indices):
                raise ValueError(f'indices must be distinct row indices in [0, {N}) within every round')
        self.n_rounds_, psi = len(self.indices), len(self.indices[0])
        self.N, self.V, self._fit_views = N, V, views
        self.dims_ = [x.shape[1] for x in views]
        self.q_fit = self._uniq(views)
        self.U_fit = [q[0] for q in self.q_fit]
        self.anchor_uid = [[self.q_fit[v][1][idx] for idx in self.indices] for v in range(V)]
        T_ = self.n_rounds_
        self.cnt = np.zeros((V, T_, psi)); self.cmax = np.zeros((V, T_, psi)); self.cmean = np.zeros((V, T_, psi))
        for it in range(T_):
            for v in range(V):
                D = self._round_dist(self.q_fit, v, it)
                M = D == D.min(1, keepdims=True)
                c = M.sum(0).astype(float)
                self.cnt[v, it] = c
                self.cmean[v, it] = np.divide((D * M).sum(0), c, out=np.zeros(psi), where=c > 0)
                self.cmax[v, it] = np.where(M, D, 0.0).max(0)
        # G: largest cell mean among cells with at least two members, taken separately in every round and view;
        # 1 if no cell has two members (then every cell is degenerate and G is not used)
        has2 = (self.cnt >= 2).any(axis=2)
        self.G = np.where(has2, np.where(self.cnt >= 2, self.cmean, 0.0).max(axis=2), 1.0)
        return self

    def _S(self, v, it, D):
        c = self.cnt[v, it][None, :]; cmax = self.cmax[v, it][None, :]; cmean = self.cmean[v, it][None, :]
        G = self.G[v, it]
        reg = (c >= 2) & (cmax > 0)                                   # implies cmean > 0, hence G > 0
        zero_r = (c >= 2) & (cmax == 0)
        S = np.where(zero_r, (D > 0).astype(float), 1.0)
        return np.where(reg, D / np.where(cmax > 0, cmax, 1.0) * cmean / (G if G > 0 else 1.0), S)

    def _query(self, views):
        if not hasattr(self, 'q_fit'):
            raise RuntimeError('call fit before scoring')
        views = _check_views(views, self.V, self.dims_)
        same = len(views[0]) == self.N and all(np.array_equal(a, b) for a, b in zip(views, self._fit_views))
        return self.q_fit if same else self._uniq(views)

    def round_components(self, views, it, q=None):
        """Round it: transfer score T_r, consistency C_r (I_r = 1 - C_r), tie sets A[v] (N x psi bool) and
        single-view scores S[v] (N x psi)."""
        q = self._query(views) if q is None else q
        V = len(q)
        D = [self._round_dist(q, v, it) for v in range(V)]
        A = [d == d.min(1, keepdims=True) for d in D]
        S = [self._S(v, it, D[v]) for v in range(V)]
        W = np.max(S, axis=0)
        T = np.mean([np.where(A[u], W, np.inf).min(1) for u in range(V)], axis=0)
        C = np.logical_and.reduce(A).any(1)
        return T, C, A, S

    def decision_components(self, views):
        """Mean transfer score T and mean inconsistency I over the rounds; the VoCE score is T + I."""
        q = self._query(views)
        n = len(q[0][1])
        Ts = np.zeros(n); Is = np.zeros(n)
        for it in range(self.n_rounds_):
            T, C, _, _ = self.round_components(None, it, q)
            Ts += T; Is += ~C
        return Ts / self.n_rounds_, Is / self.n_rounds_

    def decision_function(self, views):
        """VoCE score, higher = more anomalous."""
        T, I = self.decision_components(views)
        return T + I

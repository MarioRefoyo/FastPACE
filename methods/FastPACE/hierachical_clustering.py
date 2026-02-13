from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform


class ChannelGrouper(ABC):
    def __init__(self, n_channels: int):
        self.n_channels = n_channels

    def cut(self, channel_pct):
        if channel_pct == 1:
            groups = [list(range(self.n_channels))]
            return groups
        else:
            groups = self._cut_custom(channel_pct)
            return groups

    @abstractmethod
    def _cut_custom(self, channel_pct):
        pass


class NaiveOrder(ChannelGrouper):
    def _cut_custom(self, channel_pct) -> List[np.ndarray]:
        ch_block_len = int(max(1, np.floor(channel_pct * self.n_channels)))
        n_clusters = int(np.ceil(self.n_channels / ch_block_len))
        clusters = []
        for n in range(n_clusters):
            ch_start = n * ch_block_len
            ch_end = min(ch_start + ch_block_len, self.n_channels)
            cluster_channels = list(range(ch_start, ch_end))
            clusters.append(cluster_channels)
        return clusters


class ChannelHierarchy(ChannelGrouper):
    def __init__(
        self,
        n_channels,
        similarity: str = "spearman",
        corr_power: float = 0.75,
        linkage_method: str = "ward",
        standardize: bool = False,
        pct_margin: float = 0.5,
        cohesion_guard_q: float = 0.25,
        repair_small: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__(n_channels)
        self.similarity = similarity.lower()
        assert self.similarity in {"pearson", "spearman"}
        self.corr_power = float(corr_power)
        self.linkage_method = linkage_method
        self.standardize = bool(standardize)
        self.pct_margin = pct_margin
        self.cohesion_guard_q = cohesion_guard_q
        self.repair_small = repair_small
        self.eps = float(eps)

        # Fitted artifacts
        self.C_: Optional[int] = None                 # number of channels
        self.corr_: Optional[np.ndarray] = None       # (C, C)
        self.sim_: Optional[np.ndarray] = None        # (C, C)
        self.dist_: Optional[np.ndarray] = None       # (C, C)
        self.Z_: Optional[np.ndarray] = None          # SciPy linkage matrix (C-1, 4)
        self.heights_: Optional[np.ndarray] = None    # merge heights (C-1,)

        # Tree bookkeeping
        self._children_: Optional[Dict[int, Tuple[int, int]]] = None  # node_id -> (left, right)
        self._sizes_: Optional[Dict[int, int]] = None                  # node_id -> # leaves
        self._is_leaf_: Optional[Dict[int, bool]] = None               # node_id -> bool

    # ---------------------------
    # Fitting: build the tree
    # ---------------------------
    def fit(self, X: np.ndarray) -> "ChannelHierarchy":
        """
        Fit the channel hierarchy on training data.

        Parameters
        ----------
        X : np.ndarray
            Either shape (N, C, T) or (C, T_total).
            We'll flatten across the sample/time axes to a single vector per channel
            and compute channel-by-channel correlation.
        """
        X = np.asarray(X)
        if X.ndim == 3:
            # (N, C, T) -> (C, N*T)
            N, C, T = X.shape
            Xc = X.transpose(1, 0, 2).reshape(C, N * T)
        elif X.ndim == 2:
            # (C, T_total)
            C, _ = X.shape
            Xc = X
        else:
            raise ValueError(f"X must be (N,C,T) or (C,T); got {X.shape}")

        self.C_ = C

        # Rank transform for Spearman
        if self.similarity == "spearman":
            Xr = self._rank_along_axis(Xc, axis=1).astype(np.float64)
            Xuse = Xr
            do_standardize = False  # ranks are scale-free
        else:
            Xuse = Xc.astype(np.float64)
            do_standardize = self.standardize

        if do_standardize:
            Xuse = self._zscore_rows(Xuse, eps=self.eps)  # (C, L)

        # Pearson correlation between channels (rows)
        corr = self._corr_rows(Xuse, eps=self.eps)  # (C, C)
        corr = np.clip(corr, -1.0, 1.0)
        corr[np.isnan(corr)] = 0.0

        sim = np.abs(corr) ** self.corr_power
        np.fill_diagonal(sim, 1.0)

        # Distance for clustering
        dist = 1.0 - sim
        dist = np.clip(dist, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)

        # SciPy linkage expects condensed distance vector
        Z = linkage(squareform(dist, checks=False), method=self.linkage_method)
        heights = Z[:, 2]

        self.corr_ = corr
        self.sim_ = sim
        self.dist_ = dist
        self.Z_ = Z
        self.heights_ = heights

        # Build tree bookkeeping
        self._build_tree_structures()

        return self

    # ---------------------------
    # Cutting: size-aware, cohesion-aware
    # ---------------------------
    def _cut_custom(self, channel_pct) -> List[np.ndarray]:
        """
        Produce a clustering close to a target percentage (or n_clusters), using a
        top-down split with cohesion guard and tiny-cluster repair.

        Returns a list of 1D np.ndarray of channel indices (each array is a cluster).
        """
        self._check_fitted()

        C = self.C_
        target_k, target_size = self._target_from_config(channel_pct, C)

        # Size bounds
        alpha_min = 1-self.pct_margin
        beta_max = 1+self.pct_margin
        m_min = max(1, int(np.floor(alpha_min * target_size)))
        m_max = max(m_min, int(np.ceil(beta_max * target_size)))

        # Cohesion guard
        guard_h = np.quantile(self.heights_, self.cohesion_guard_q) if self.heights_ is not None else 0.0

        # Start from the root node (last row represents root)
        root_id = C + (C - 2)  # node ids: 0..C-1 leaves, C..C+(C-2) internal; root is last
        clusters_nodes = self._top_down_split(root_id, m_max, guard_h)

        # Convert node ids to leaf index arrays
        clusters = [np.array(self._collect_leaves(n), dtype=np.int32) for n in clusters_nodes]
        clusters = [c for c in clusters if c.size > 0]

        # Repair too-small clusters by merging to nearest neighbor
        if self.repair_small:
            clusters = self._merge_tiny_clusters(clusters, min_size=m_min)

        # Optional: if you want roughly target_k clusters, you can lightly split largest, high-height nodes
        # until you approach target_k (staying above m_min), but we keep it soft by default.

        return clusters

    # ---------------------------
    # Internals: tree & utilities
    # ---------------------------

    def _build_tree_structures(self):
        """Build child links, sizes, and leaf flags from SciPy linkage."""
        C = self.C_
        Z = self.Z_
        children = {}
        sizes = {}
        is_leaf = {}

        # Initialize leaves
        for i in range(C):
            is_leaf[i] = True
            sizes[i] = 1

        # Internal nodes
        node_id = C
        for i in range(Z.shape[0]):
            left = int(Z[i, 0])
            right = int(Z[i, 1])
            children[node_id] = (left, right)
            is_leaf[node_id] = False
            sizes[node_id] = sizes[left] + sizes[right]
            node_id += 1

        self._children_ = children
        self._sizes_ = sizes
        self._is_leaf_ = is_leaf

    def _collect_leaves(self, node_id: int) -> List[int]:
        """Return list of leaf indices under node_id."""
        if self._is_leaf_[node_id]:
            return [node_id]
        left, right = self._children_[node_id]
        return self._collect_leaves(left) + self._collect_leaves(right)

    def _node_height(self, node_id: int) -> float:
        """Return the merge height of node_id (0 for leaves)."""
        C = self.C_
        if node_id < C:
            return 0.0
        # internal nodes are ordered C..C+(C-2) in the order of Z rows
        row = node_id - C
        return float(self.Z_[row, 2])

    def _top_down_split(self, root_id: int, m_max: int, guard_h: float) -> List[int]:
        """
        Split nodes whose size exceeds m_max *and* whose height is above guard_h.
        Keep cohesive (low-height) nodes intact even if a bit large.
        """
        clusters = []
        stack = [root_id]
        while stack:
            nid = stack.pop()
            size = self._sizes_[nid]
            height = self._node_height(nid)

            if size <= m_max or height <= guard_h or self._is_leaf_[nid]:
                clusters.append(nid)
            else:
                left, right = self._children_[nid]
                stack.append(right)
                stack.append(left)
        return clusters

    def _merge_tiny_clusters(self, clusters: List[np.ndarray], min_size: int) -> List[np.ndarray]:
        """Greedy merge of clusters with size < min_size to their nearest neighbor (average distance)."""
        if len(clusters) <= 1:
            return clusters

        # Work on a mutable list
        clusters = [c.copy() for c in clusters]
        dist = self.dist_  # (C, C)

        def avg_link(i, j):
            ci, cj = clusters[i], clusters[j]
            # average pairwise distance
            return float(dist[np.ix_(ci, cj)].mean()) if ci.size and cj.size else np.inf

        changed = True
        while changed:
            changed = False
            # Find any tiny cluster
            tiny_idx = [i for i, c in enumerate(clusters) if c.size < min_size]
            if not tiny_idx:
                break
            for i in tiny_idx:
                # Find nearest neighbor j != i
                best_j, best_d = None, np.inf
                for j in range(len(clusters)):
                    if j == i:
                        continue
                    d = avg_link(i, j)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j is None:
                    continue
                # Merge i into j
                merged = np.unique(np.concatenate([clusters[i], clusters[best_j]]))
                clusters[best_j] = merged
                clusters[i] = np.array([], dtype=np.int32)
                changed = True
            # Remove empties
            clusters = [c for c in clusters if c.size > 0]

        return clusters

    @staticmethod
    def _target_from_config(channel_pct, C) -> Tuple[int, float]:
        """Return (target_k, target_size) given C channels and the cut config."""
        if channel_pct is None:
            raise ValueError("Provide exactly one of channel_pct or n_clusters.")

        p = float(channel_pct)
        if not (0.0 < p <= 1.0):
            raise ValueError("channel_pct must be in (0,1].")
        k = max(1, int(round(1.0 / p)))
        m = max(1.0, C / k)
        return k, m

    def _check_fitted(self):
        if self.Z_ is None or self.C_ is None or self.dist_ is None:
            raise RuntimeError("Call fit(X) before cut().")

    # ---------------
    # Stats helpers
    # ---------------
    @staticmethod
    def _zscore_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Z-score each row: (x - mean) / (std + eps)."""
        mu = A.mean(axis=1, keepdims=True)
        sd = A.std(axis=1, keepdims=True)
        return (A - mu) / (sd + eps)

    @staticmethod
    def _rank_along_axis(A: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute ranks along an axis (Spearman preparation), average ties."""
        # argsort twice trick to get integer ranks; for ties, we can approximate with mean rank via a stable argsort pass
        A = np.asarray(A)
        # Simple, fast approximation: rank by argsort twice (ties get consecutive ranks)
        order = np.argsort(A, axis=axis, kind="mergesort")
        ranks = np.argsort(order, axis=axis, kind="mergesort").astype(np.float64)
        return ranks

    @staticmethod
    def _corr_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Row-wise Pearson correlation of A (C x L) -> (C x C)."""
        A = np.asarray(A, dtype=np.float64)
        A = A - A.mean(axis=1, keepdims=True)
        ss = np.sqrt((A * A).sum(axis=1, keepdims=True)) + eps
        A = A / ss
        return np.clip(A @ A.T, -1.0, 1.0)


class ChannelHierarchyFixedK(ChannelHierarchy):
    """
    Same tree construction as ChannelHierarchy, but the cut produces EXACTLY K clusters
    (with K inferred from channel_pct as K=round(1/p)). No margins, guards, or repairs.
    """

    def __init__(
        self,
        n_channels,
        similarity: str = "spearman",
        corr_power: float = 0.75,
        linkage_method: str = "ward",
        standardize: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__(
            n_channels=n_channels,
            similarity=similarity,
            corr_power=corr_power,
            linkage_method=linkage_method,
            standardize=standardize,
            pct_margin=0.0,            # unused here
            cohesion_guard_q=0.0,      # unused here
            repair_small=False,        # unused here
            eps=eps,
        )

    def _cut_custom(self, channel_pct) -> List[np.ndarray]:
        """
        Produce exactly K clusters by traversing the linkage from leaves upward and
        stopping when K components remain. Deterministic and independent of margins.
        """
        self._check_fitted()
        C = self.C_
        Z = self.Z_
        if C is None or Z is None:
            raise RuntimeError("Call fit(X) before cut().")

        # Derive target K from p (e.g., p=0.2 -> K=5). Clamp to [1, C].
        p = float(channel_pct)
        if not (0.0 < p <= 1.0):
            raise ValueError("channel_pct must be in (0,1].")
        K = int(round(1.0 / p))
        K = max(1, min(K, C))

        # Active clusters represented by node ids. Start with leaves 0..C-1.
        # We'll build leaf index arrays for any internal node we materialize.
        clusters = {i: np.array([i], dtype=np.int32) for i in range(C)}  # node_id -> leaf indices
        active = set(range(C))                                           # current frontier nodes

        # Walk linkage rows in ascending distance; each row merges two nodes into a new one.
        # SciPy's convention: row i merges nodes Z[i,0] and Z[i,1] into new node id (C + i).
        for i in range(Z.shape[0]):
            if len(active) <= K:
                break
            left = int(Z[i, 0])
            right = int(Z[i, 1])
            new_id = C + i

            # Fetch/compose leaf sets for the children (if child is an internal id, it must exist in clusters)
            li = clusters[left]
            ri = clusters[right]
            merged = np.concatenate([li, ri])

            # Update frontier: remove children, add parent
            active.discard(left)
            active.discard(right)
            clusters[new_id] = merged
            active.add(new_id)

        # If we still have more than K (can happen if C==K), trim nothing; if fewer than K (unlikely), we stop earlier anyway.
        # Collect final clusters as arrays of leaf indices
        result = [np.sort(clusters[nid]) for nid in active]

        # Ensure exactly K groups: if linkage loop ended early because len(active) == K, we’re done.
        # If by any chance len(active) > K (e.g., degenerate Z), merge smallest until K (rare).
        if len(result) > K:
            # Greedy: merge smallest with its nearest by average-link distance until K
            # (very unlikely with a valid Z; kept for robustness)
            dist = self.dist_
            parts = [r.copy() for r in result]
            while len(parts) > K:
                # find closest pair by average link
                best_i, best_j, best_d = None, None, float("inf")
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts)):
                        d = float(dist[np.ix_(parts[i], parts[j])].mean())
                        if d < best_d:
                            best_d = d
                            best_i, best_j = i, j
                merged = np.unique(np.concatenate([parts[best_i], parts[best_j]]))
                # rebuild list
                new_parts = []
                for t in range(len(parts)):
                    if t not in (best_i, best_j):
                        new_parts.append(parts[t])
                new_parts.append(merged)
                parts = new_parts
            result = [np.sort(r) for r in parts]

        # Sort clusters for determinism (by their minimum index)
        result.sort(key=lambda a: (a.min() if a.size else 10**9))
        return result


class ChannelGreedyGroups(ChannelGrouper):
    def __init__(
            self,
            n_channels,
            similarity: str = "spearman",
            corr_power: float = 0.75,
            linkage_method: str = "ward",
            standardize: bool = False,
            eps: float = 1e-12,
    ):
        super().__init__(n_channels)
        self.similarity = similarity.lower()
        assert self.similarity in {"pearson", "spearman"}
        self.corr_power = float(corr_power)
        self.linkage_method = linkage_method
        self.standardize = bool(standardize)
        self.eps = float(eps)

        # Fitted artifacts
        self.C_: Optional[int] = None                 # number of channels
        self.dist_: Optional[np.ndarray] = None       # (C, C)
        self.Z_: Optional[np.ndarray] = None          # SciPy linkage matrix (C-1, 4)
        self.leaf_order: Optional[np.ndarray] = None

    # ---------------------------
    # Fit: build distances
    # ---------------------------
    def fit(self, X: np.ndarray):
        """
        X: (N, C, T) or (C, T_total)
        """
        X = np.asarray(X)
        if X.ndim == 3:
            # (N, C, T) -> (C, N*T)
            N, C, T = X.shape
            Xc = X.transpose(1, 0, 2).reshape(C, N * T)
        elif X.ndim == 2:
            # (C, T_total)
            C, _ = X.shape
            Xc = X
        else:
            raise ValueError(f"X must be (N,C,T) or (C,T); got {X.shape}")

        self.C_ = C

        # Rank transform for Spearman
        if self.similarity == "spearman":
            Xr = self._rank_along_axis(Xc, axis=1).astype(np.float64)
            Xuse = Xr
            do_standardize = False  # ranks are scale-free
        else:
            Xuse = Xc.astype(np.float64)
            do_standardize = self.standardize

        if do_standardize:
            Xuse = self._zscore_rows(Xuse, eps=self.eps)  # (C, L)

        # Pearson correlation between channels (rows)
        corr = self._corr_rows(Xuse, eps=self.eps)  # (C, C)
        corr = np.clip(corr, -1.0, 1.0)
        corr[np.isnan(corr)] = 0.0

        sim = np.abs(corr) ** self.corr_power
        np.fill_diagonal(sim, 1.0)

        # Distance for clustering
        dist = 1.0 - sim
        dist = np.clip(dist, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        self.dist = dist

        Z = linkage(squareform(dist, checks=False), method=self.linkage_method)
        Zopt = optimal_leaf_ordering(Z, squareform(dist, checks=False))
        self.Z_ = Zopt

        order = leaves_list(Zopt)
        self.leaf_order = order
        return self

    # ---------------------------
    # Public API
    # ---------------------------
    def _cut_custom(self, channel_pct) -> List[np.ndarray]:
        """
        Returns a list of clusters (each as a 1D np.ndarray of channel indices).
        """
        if self.dist is None or self.leaf_order is None:
            raise RuntimeError("Call fit(X, ...) before group().")
        C = self.C_
        order = self.leaf_order

        sizes = self._compute_sizes(C, channel_pct)
        groups = self._slice_by_sizes(order, sizes)

        return groups

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _slice_by_sizes(order: np.ndarray, sizes: List[int]) -> List[np.ndarray]:
        groups: List[np.ndarray] = []
        start = 0
        for s in sizes:
            end = start + s
            groups.append(order[start:end].copy())
            start = end
        return groups

    @staticmethod
    def _compute_sizes(C: int, channel_pct) -> List[int]:
        """
        Decide exact cluster sizes to cover all C channels.
        - If cluster_size is given: use it; last cluster may be <= cluster_size.
        - Else use channel_pct with size_mode to get a base size s, then pack C as:
            [s, s, ..., s, C - s*(K-1)] with K=ceil(C/s).
        """

        if channel_pct is None:
            raise ValueError("Provide channel_pct or cluster_size.")
        p = float(channel_pct)
        if not (0.0 < p <= 1.0):
            raise ValueError("channel_pct must be in (0,1].")
        raw = p * C

        s = max(1, int(np.ceil(raw)))

        # Build sizes to cover all channels
        K = int(np.ceil(C / s))
        sizes = [s] * (K - 1) + [C - s * (K - 1)]
        # Edge: if last becomes 0 (can happen if C is exact multiple and ceil/round math), fix it
        if sizes[-1] == 0:
            sizes = [s] * (K - 1)
        return sizes

    @staticmethod
    def _zscore_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Z-score each row: (x - mean) / (std + eps)."""
        mu = A.mean(axis=1, keepdims=True)
        sd = A.std(axis=1, keepdims=True)
        return (A - mu) / (sd + eps)

    @staticmethod
    def _rank_along_axis(A: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute ranks along an axis (Spearman preparation), average ties."""
        # argsort twice trick to get integer ranks; for ties, we can approximate with mean rank via a stable argsort pass
        A = np.asarray(A)
        # Simple, fast approximation: rank by argsort twice (ties get consecutive ranks)
        order = np.argsort(A, axis=axis, kind="mergesort")
        ranks = np.argsort(order, axis=axis, kind="mergesort").astype(np.float64)
        return ranks

    @staticmethod
    def _corr_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Row-wise Pearson correlation of A (C x L) -> (C x C)."""
        A = np.asarray(A, dtype=np.float64)
        A = A - A.mean(axis=1, keepdims=True)
        ss = np.sqrt((A * A).sum(axis=1, keepdims=True)) + eps
        A = A / ss
        return np.clip(A @ A.T, -1.0, 1.0)


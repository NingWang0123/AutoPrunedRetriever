import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, replace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing various ANN libraries
FAISS_AVAILABLE = False
HNSWLIB_AVAILABLE = False
PYNNDESCENT_AVAILABLE = False
ANNOY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available for ANN search")
except ImportError:
    pass

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
    logger.info("HNSWlib available for ANN search")
except ImportError:
    pass

try:
    import pynndescent
    PYNNDESCENT_AVAILABLE = True
    logger.info("PyNNDescent available for ANN search")
except ImportError:
    pass

try:
    import annoy
    ANNOY_AVAILABLE = True
    logger.info("Annoy available for ANN search")
except ImportError:
    pass

from sklearn.neighbors import NearestNeighbors  # Fallback


class ANNBackend(Enum):
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    PYNNDESCENT = "pynndescent"
    ANNOY = "annoy"
    SKLEARN = "sklearn"


@dataclass
class ClusteringConfig:
    """Configuration for ANN k-NN clustering"""
    k_neighbors: int = 10          # Number of nearest neighbors
    similarity_threshold: float = 0.8  # Minimum similarity to connect
    min_cluster_size: int = 2      # Minimum size to keep cluster
    ann_backend: str = "auto"      # ANN backend to use
    metric: str = "cosine"         # Distance metric
    representative_method: str = "medoid"  # "medoid" or "density"
    use_gpu: bool = True           # Use GPU if available
    n_trees: int = 50              # For Annoy
    ef_construction: int = 200     # For HNSW
    ef_search: int = 100           # For HNSW search
    

class UnionFind:
    """Efficient Union-Find (Disjoint Set) data structure"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, x: int) -> int:
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank, returns True if merged"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def get_clusters(self) -> Dict[int, List[int]]:
        """Get all clusters as dict of root -> members"""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters
    
    def get_cluster_size(self, x: int) -> int:
        """Get size of cluster containing x"""
        return self.size[self.find(x)]


class ANNGraphBuilder:
    """Build approximate nearest neighbor graph using various backends"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.backend = self._select_backend()
        logger.info(f"Using ANN backend: {self.backend}")
    
    def _select_backend(self) -> ANNBackend:
        """Select best available backend"""
        backend_map = {
            "faiss": ANNBackend.FAISS,
            "hnswlib": ANNBackend.HNSWLIB,
            "pynndescent": ANNBackend.PYNNDESCENT,
            "annoy": ANNBackend.ANNOY,
            "sklearn": ANNBackend.SKLEARN
        }
        
        if self.config.ann_backend != "auto":
            return backend_map.get(self.config.ann_backend, ANNBackend.SKLEARN)
        
        # Auto-select best available
        if FAISS_AVAILABLE:
            return ANNBackend.FAISS
        elif HNSWLIB_AVAILABLE:
            return ANNBackend.HNSWLIB
        elif PYNNDESCENT_AVAILABLE:
            return ANNBackend.PYNNDESCENT
        elif ANNOY_AVAILABLE:
            return ANNBackend.ANNOY
        else:
            return ANNBackend.SKLEARN
    
    def build_graph(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph.
        Returns: (indices, distances) where each row i contains k neighbors of point i
        """
        if self.backend == ANNBackend.FAISS:
            return self._build_faiss(X)
        elif self.backend == ANNBackend.HNSWLIB:
            return self._build_hnswlib(X)
        elif self.backend == ANNBackend.PYNNDESCENT:
            return self._build_pynndescent(X)
        elif self.backend == ANNBackend.ANNOY:
            return self._build_annoy(X)
        else:
            return self._build_sklearn(X)
    
    def _build_faiss(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph using FAISS"""
        X = np.ascontiguousarray(X.astype(np.float32))
        d = X.shape[1]
        
        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            index = faiss.IndexFlatIP(d)  # Inner product = cosine for normalized
        else:
            index = faiss.IndexFlatL2(d)
        
        # Try to use GPU if available and requested
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using FAISS GPU")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Use HNSW for large datasets
        if len(X) > 10000:
            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
        
        index.add(X)
        
        # Search for k+1 neighbors (including self)
        distances, indices = index.search(X, self.config.k_neighbors + 1)
        
        # Remove self-loops (first neighbor is always self)
        return indices[:, 1:], distances[:, 1:]
    
    def _build_hnswlib(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph using HNSWlib"""
        import hnswlib
        
        X = np.ascontiguousarray(X.astype(np.float32))
        n, d = X.shape
        
        # Normalize for cosine
        if self.config.metric == "cosine":
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            space = "cosinesimil"
        else:
            space = "l2"
        
        # Build index
        index = hnswlib.Index(space=space, dim=d)
        index.init_index(max_elements=n, ef_construction=self.config.ef_construction, M=16)
        index.add_items(X, np.arange(n))
        index.set_ef(self.config.ef_search)
        
        # Query
        indices, distances = index.knn_query(X, k=self.config.k_neighbors + 1)
        
        # Remove self-loops
        return indices[:, 1:], distances[:, 1:]
    
    def _build_pynndescent(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph using PyNNDescent"""
        import pynndescent
        
        # Normalize for cosine
        if self.config.metric == "cosine":
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            metric = "cosine"
        else:
            metric = "euclidean"
        
        # Build index
        index = pynndescent.NNDescent(
            X,
            n_neighbors=self.config.k_neighbors + 1,
            metric=metric,
            random_state=42,
            n_jobs=-1
        )
        
        indices, distances = index.neighbor_graph
        
        # Remove self-loops
        return indices[:, 1:], distances[:, 1:]
    
    def _build_annoy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph using Annoy"""
        from annoy import AnnoyIndex
        
        n, d = X.shape
        metric = "angular" if self.config.metric == "cosine" else "euclidean"
        
        # Build index
        index = AnnoyIndex(d, metric)
        for i in range(n):
            index.add_item(i, X[i])
        index.build(self.config.n_trees)
        
        # Query
        indices = np.zeros((n, self.config.k_neighbors), dtype=np.int32)
        distances = np.zeros((n, self.config.k_neighbors), dtype=np.float32)
        
        for i in range(n):
            # Get k+1 to exclude self
            idx, dist = index.get_nns_by_item(i, self.config.k_neighbors + 1, include_distances=True)
            # Remove self
            idx = [j for j in idx if j != i][:self.config.k_neighbors]
            dist = dist[1:self.config.k_neighbors + 1]
            
            indices[i] = idx
            distances[i] = dist
        
        return indices, distances
    
    def _build_sklearn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to sklearn NearestNeighbors"""
        metric = self.config.metric if self.config.metric != "cosine" else "cosine"
        
        nbrs = NearestNeighbors(
            n_neighbors=self.config.k_neighbors + 1,
            metric=metric,
            n_jobs=-1
        )
        nbrs.fit(X)
        
        distances, indices = nbrs.kneighbors(X)
        
        # Remove self-loops
        return indices[:, 1:], distances[:, 1:]


# -----------------------
# Fixed RepresentativeSelector (metric-aware)
# -----------------------
class RepresentativeSelector:
    """Select representatives from clusters"""

    @staticmethod
    def select_medoid(X: np.ndarray, cluster_indices: List[int], metric: str = "euclidean") -> int:
        """Select point with minimum distance to all others (medoid)"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        from sklearn.metrics.pairwise import pairwise_distances
        dists = pairwise_distances(X[cluster_indices], metric=("cosine" if metric == "cosine" else "euclidean"))
        medoid_idx = np.argmin(dists.sum(axis=1))
        return cluster_indices[medoid_idx]

    @staticmethod
    def select_density_peak(X: np.ndarray, cluster_indices: List[int], k: int = 5, metric: str = "euclidean") -> int:
        """Select point with highest local density"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        Xc = X[cluster_indices]
        k = min(k, len(cluster_indices) - 1)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric=("cosine" if metric == "cosine" else "euclidean"))
        nn.fit(Xc)
        dists, _ = nn.kneighbors(Xc)
        densities = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-10)
        return cluster_indices[int(np.argmax(densities))]


# -----------------------
# Fixed ANNGraphBuilder (always returns SIMILARITIES for cosine)
# -----------------------
class ANNGraphBuilder:
    """Build approximate nearest neighbor graph using various backends"""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.backend = self._select_backend()
        logger.info(f"Using ANN backend: {self.backend}")

    def _select_backend(self) -> ANNBackend:
        backend_map = {
            "faiss": ANNBackend.FAISS,
            "hnswlib": ANNBackend.HNSWLIB,
            "pynndescent": ANNBackend.PYNNDESCENT,
            "annoy": ANNBackend.ANNOY,
            "sklearn": ANNBackend.SKLEARN
        }
        if self.config.ann_backend != "auto":
            return backend_map.get(self.config.ann_backend, ANNBackend.SKLEARN)

        if FAISS_AVAILABLE:
            return ANNBackend.FAISS
        elif HNSWLIB_AVAILABLE:
            return ANNBackend.HNSWLIB
        elif PYNNDESCENT_AVAILABLE:
            return ANNBackend.PYNNDESCENT
        elif ANNOY_AVAILABLE:
            return ANNBackend.ANNOY
        else:
            return ANNBackend.SKLEARN

    def build_graph(self, X_unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph.
        Returns: (indices, sims) where sims are COSINE SIMILARITIES if metric=='cosine'.
        """
        n = X_unit.shape[0]
        k = min(self.config.k_neighbors, max(1, n - 1))

        if self.backend == ANNBackend.FAISS:
            idx, sims = self._build_faiss_cosine(X_unit, k) if self.config.metric == "cosine" else self._build_faiss_l2(X_unit, k)
        elif self.backend == ANNBackend.HNSWLIB:
            idx, sims = self._build_hnswlib(X_unit, k)
        elif self.backend == ANNBackend.PYNNDESCENT:
            idx, sims = self._build_pynndescent(X_unit, k)
        elif self.backend == ANNBackend.ANNOY:
            idx, sims = self._build_annoy(X_unit, k)
        else:
            idx, sims = self._build_sklearn(X_unit, k)


        # Robustly remove self and pad/truncate to exactly k
        clean_idx = np.empty((n, k), dtype=np.int32)
        clean_sims = np.empty((n, k), dtype=np.float32)

        for i in range(n):
            row_idx = np.asarray(idx[i], dtype=np.int32)
            row_sims = np.asarray(sims[i], dtype=np.float32)

            # drop self wherever it appears
            keep = row_idx != i
            row_idx = row_idx[keep]
            row_sims = row_sims[keep]

            # if not enough neighbors remain, pad
            if row_idx.shape[0] < k:
                missing = k - row_idx.shape[0]
                if row_idx.shape[0] > 0:
                    pad_idx = np.full(missing, row_idx[-1], dtype=np.int32)
                    pad_sims = np.full(missing, row_sims[-1], dtype=np.float32)
                else:
                    # pathological case: no neighbors returned; pad with self and a very low sim
                    pad_idx = np.full(missing, i, dtype=np.int32)
                    pad_sims = np.full(missing, -1.0, dtype=np.float32)
                row_idx = np.concatenate([row_idx, pad_idx], axis=0)
                row_sims = np.concatenate([row_sims, pad_sims], axis=0)

            # enforce exact length k
            clean_idx[i, :] = row_idx[:k]
            clean_sims[i, :] = row_sims[:k]

        return clean_idx, clean_sims

    # ---------- FAISS helpers ----------
    def _build_faiss_cosine(self, X_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS (cosine) via inner product on unit vectors -> returns cosine SIMILARITIES."""
        import faiss
        Xf = np.ascontiguousarray(X_unit.astype(np.float32))
        d = Xf.shape[1]
        index = faiss.IndexFlatIP(d)  # IP on unit vectors = cosine similarity
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using FAISS GPU (IP)")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(Xf)
        sims, idx = index.search(Xf, k + 1)  # sims are inner products
        # Drop first col (self) assuming exact search; still self-cleaned again later
        return idx[:, 1:], sims[:, 1:]

    def _build_faiss_l2(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS (L2) -> convert to similarities via Gaussian kernel."""
        import faiss
        Xf = np.ascontiguousarray(X.astype(np.float32))
        d = Xf.shape[1]
        index = faiss.IndexFlatL2(d)
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using FAISS GPU (L2)")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(Xf)
        dists, idx = index.search(Xf, k + 1)
        d2 = dists[:, 1:]
        # Gaussian kernel similarity
        sigma = np.median(d2)
        sigma = float(sigma if sigma > 1e-12 else 1.0)
        sims = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        return idx[:, 1:], sims

    # ---------- HNSWlib ----------
    def _build_hnswlib(self, X_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        import hnswlib
        n, d = X_unit.shape
        if self.config.metric == "cosine":
            space = "cosine"  # correct name
        else:
            space = "l2"
        index = hnswlib.Index(space=space, dim=d)
        index.init_index(max_elements=n, ef_construction=self.config.ef_construction, M=16)
        index.add_items(X_unit, np.arange(n))
        index.set_ef(self.config.ef_search)
        idx, dists = index.knn_query(X_unit, k=k + 1)
        if self.config.metric == "cosine":
            sims = (1.0 - dists[:, 1:]).astype(np.float32)
            return idx[:, 1:], sims
        else:
            d2 = dists[:, 1:]
            sigma = np.median(d2)
            sigma = float(sigma if sigma > 1e-12 else 1.0)
            sims = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
            return idx[:, 1:], sims

    # ---------- PyNNDescent ----------
    def _build_pynndescent(self, X_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        import pynndescent
        if self.config.metric == "cosine":
            metric = "cosine"
        else:
            metric = "euclidean"
        index = pynndescent.NNDescent(
            X_unit if self.config.metric == "cosine" else X_unit,  # X_unit ok both ways
            n_neighbors=k + 1,
            metric=metric,
            random_state=42
        )
        idx, dists = index.neighbor_graph
        if self.config.metric == "cosine":
            sims = (1.0 - dists[:, 1:]).astype(np.float32)
        else:
            d2 = dists[:, 1:]
            sigma = np.median(d2)
            sigma = float(sigma if sigma > 1e-12 else 1.0)
            sims = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        return idx[:, 1:], sims

    # ---------- Annoy ----------
    def _build_annoy(self, X_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        from annoy import AnnoyIndex
        n, d = X_unit.shape
        metric = "angular" if self.config.metric == "cosine" else "euclidean"
        index = AnnoyIndex(d, metric)
        for i in range(n):
            index.add_item(i, X_unit[i])
        index.build(self.config.n_trees)
        idx = np.empty((n, k + 1), dtype=np.int32)
        sims = np.empty((n, k + 1), dtype=np.float32)
        for i in range(n):
            nn_idx, _ = index.get_nns_by_item(i, k + 1, include_distances=True)
            # compute cosine sim directly to avoid Annoy's distance conversions
            nn_idx = np.array(nn_idx, dtype=np.int32)
            sim_row = (X_unit[i] @ X_unit[nn_idx].T).astype(np.float32)
            idx[i] = nn_idx
            sims[i] = sim_row
        return idx[:, 1:], sims[:, 1:]

    # ---------- sklearn ----------
    def _build_sklearn(self, X_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors
        if self.config.metric == "cosine":
            nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
            nn.fit(X_unit)
            dists, idx = nn.kneighbors(X_unit)
            sims = (1.0 - dists[:, 1:]).astype(np.float32)
            return idx[:, 1:], sims
        else:
            nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
            nn.fit(X_unit)
            dists, idx = nn.kneighbors(X_unit)
            d2 = dists[:, 1:]
            sigma = np.median(d2)
            sigma = float(sigma if sigma > 1e-12 else 1.0)
            sims = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
            return idx[:, 1:], sims


# -----------------------
# Fixed combine_ents_ann_knn (cosine similarities used correctly)
# -----------------------
def _to_vec(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=np.float32)

def _avg_vec_from_decoded(decoded_q, dim: int) -> np.ndarray:
    """
    decoded_q: [[e_vec, r_vec, e_vec], ...] where each vec is list[float] or np.ndarray
    Returns one vector (float32) = mean over all component vectors across all edges.
    If no vectors found, returns a zero vector of length `dim`.
    """
    parts = []
    for triple in decoded_q:
        for v in triple:
            if v is not None:
                vv = _to_vec(v)
                if vv.size:
                    parts.append(vv.astype(np.float32, copy=False))
    if not parts:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(parts, axis=0), axis=0)

def decode_question(question, codebook_main, fmt='words'):
    """
    question: list[int] of edge indices
    codebook_main:
        {
            "e": [str, ...],
            "r": [str, ...],
            "edge_matrix": [[e_idx, r_idx, e_idx], ...],  # list or np.ndarray
            "questions": [[edges index,...],...]
            "e_embeddings": [vec, ...], 
            "r_embeddings": [vec, ...], 
        }
    fmt: 'words' -> [[e, r, e], ...]
         'embeddings' -> [[e_vec, r_vec, e_vec], ...]
         'edges' -> [[e index, r index, e index], ...]
    """
    edges = codebook_main["edge_matrix"]

    idxs = list(question)

    def get_edge(i):
        # works for both list and numpy array
        return edges[i]

    if fmt == 'words':
        E, R = codebook_main["e"], codebook_main["r"]
        decoded = [[E[h], R[r], E[t]] for (h, r, t) in (get_edge(i) for i in idxs)]
    elif fmt == 'embeddings':
        Ee = codebook_main.get("e_embeddings")
        Re = codebook_main.get("r_embeddings")
        if Ee is None or Re is None:
            raise KeyError("e_embeddings and r_embeddings are required for fmt='embeddings'.")
        decoded = [[Ee[h], Re[r], Ee[t]] for (h, r, t) in (get_edge(i) for i in idxs)]
    elif fmt == 'edges':
        decoded = [[h,r,t] for (h, r, t) in (get_edge(i) for i in idxs)]

    else:
        raise ValueError("fmt must be 'words', 'embeddings' or 'edges'.")

    return decoded

def decode_questions(questions, questions_source_codebook, fmt='words'):

    """
    questions_source_codebook must be the codebook that contain the questions
    Decode a list of questions using decode_question.
    
    questions: list of list[int]
        Each inner list is a sequence of edge indices.
    """
    return [decode_question(q, questions_source_codebook, fmt=fmt) for q in questions]

def combine_storage_ann_knn(
    codebook_main: Dict[str, Any],
    config: Optional[ClusteringConfig] = None,
    use_thinking: bool = True,
    use_facts: bool = True
) -> Dict[str, Any]:
    """
    Entity clustering using ANN k-NN graph + Union-Find with consistent cosine handling.
    """
    # Initialize configuration
    q_word_embeds = []
    dim = len(codebook_main["e_embeddings"][0])


    # combine questions first
    all_qs = codebook_main["questions_lst"]

    for q_edges in all_qs:
        decoded = decode_questions(q_edges, codebook_main, fmt='embeddings')
        q_word_embeds.append(_avg_vec_from_decoded(decoded, dim))

    if config is None:
        n = len(codebook_main.get('e', []))
        k = min(50, max(5, int(np.sqrt(max(1, n)))))  # Adaptive k
        config = ClusteringConfig(k_neighbors=k, similarity_threshold=0.8, min_cluster_size=2)

    # Get entities/embeddings
    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n, d = X.shape if X.ndim == 2 else (len(X), 1)

    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main

    # 1D + cosine is degenerate -> switch to euclidean just for this call
    metric = config.metric
    if metric == "cosine" and d == 1:
        logger.info("Detected d=1 with cosine metric; switching to 'euclidean' for this run to avoid ±1 collapse.")
        metric = "euclidean"

    logger.info(f"Clustering {n} entities (metric={metric}) using ANN k-NN + Union-Find")

    # Normalize for cosine
    if metric == "cosine":
        X_unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    else:
        X_unit = X  # for euclidean, use raw

    # Build ANN graph -> indices, SIMILARITIES (if cosine) or kernel sims (if euclidean)
    gb = ANNGraphBuilder(replace(config, metric=metric))
    indices, sims = gb.build_graph(X_unit)

    # Union-Find merge with similarity threshold
    uf = UnionFind(n)
    edges_created = 0
    if metric == "cosine":
        thr = float(config.similarity_threshold)
        for i in range(n):
            for j, s in zip(indices[i], sims[i]):
                if s >= thr:
                    if uf.union(i, int(j)):
                        edges_created += 1
    else:
        # euclidean path: sims are kernel similarities in (0,1]; interpret threshold in [0,1]
        thr = float(config.similarity_threshold)
        if not (0.0 <= thr <= 1.0):
            thr = 0.5  # safe default if user passed cosine-like value
        for i in range(n):
            for j, s in zip(indices[i], sims[i]):
                if s >= thr:
                    if uf.union(i, int(j)):
                        edges_created += 1

    logger.info(f"Created {edges_created} union edges at threshold {config.similarity_threshold}")

    # Build clusters
    clusters = uf.get_clusters()

    # Representative selection (metric-aware)
    rep_selector = RepresentativeSelector()
    representatives: Dict[int, int] = {}
    kept_indices: List[int] = []

    for _, members in clusters.items():
        if len(members) < config.min_cluster_size:
            for m in members:
                representatives[m] = m
                kept_indices.append(m)
        else:
            if config.representative_method == "medoid":
                rep = rep_selector.select_medoid(X_unit, members, metric=metric)
            else:
                rep = rep_selector.select_density_peak(X_unit, members, k=min(5, len(members)-1), metric=metric)
            kept_indices.append(rep)
            for m in members:
                representatives[m] = rep

    kept_indices = sorted(set(kept_indices))
    logger.info(f"Reduced from {n} to {len(kept_indices)} entities (clusters={len(clusters)})")

    # Create mappings
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[representatives[i]] for i in range(n)}

    # Rebuild entities/embeddings
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    # Remap edges
    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    tuple_to_new_edge_idx: Dict[Tuple[int, int, int], int] = {}
    new_edges: List[List[int]] = []
    old_edge_to_new_edge: Dict[int, int] = {}

    for old_idx, (e1, r, e2) in enumerate(old_edges):
        ne1 = old_ent_to_new.get(e1, e1)
        ne2 = old_ent_to_new.get(e2, e2)
        tup = (ne1, int(r), ne2)
        if tup not in tuple_to_new_edge_idx:
            tuple_to_new_edge_idx[tup] = len(new_edges)
            new_edges.append([ne1, int(r), ne2])
        old_edge_to_new_edge[old_idx] = tuple_to_new_edge_idx[tup]

    def remap_edge_indices(struct):
        if isinstance(struct, list):
            return [remap_edge_indices(x) for x in struct]
        try:
            return old_edge_to_new_edge.get(int(struct), int(struct))
        except (ValueError, TypeError):
            return struct

    if codebook_main.get('questions_lst') is not None:
        codebook_main['questions_lst'] = remap_edge_indices(codebook_main['questions_lst'])
    if codebook_main.get('answers_lst') is not None:
        codebook_main['answers_lst'] = remap_edge_indices(codebook_main['answers_lst'])
    if use_thinking and codebook_main.get('thinkings_lst') is not None:
        codebook_main['thinkings_lst'] = remap_edge_indices(codebook_main['thinkings_lst'])
    if use_facts and codebook_main.get('facts_lst') is not None:
        codebook_main['facts_lst'] = remap_edge_indices(codebook_main['facts_lst'])

    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]
    return codebook_main
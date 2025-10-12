import time, numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import warnings
import logging
import psutil

# -----------------------
#original
# -----------------------
def combine_ents_original(codebook_main, min_exp_num=2, max_exp_num=20, use_thinking=True, random_state=0):
    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n = X.shape[0]
    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    k_low  = max(2, int(np.ceil(n / max_exp_num)))
    k_high = max(2, min(n - 1, int(np.floor(n / min_exp_num))))
    if k_low > k_high:
        k_low, k_high = 2, max(2, min(n - 1, 5))
    cand_ks = list(range(k_low, k_high + 1))

    best_k, best_sil, inertia_by_k = None, -1.0, {}
    for k in cand_ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_norm)
        sil = silhouette_score(X_norm, labels, metric='euclidean')
        inertia_by_k[k] = km.inertia_
        if (sil > best_sil) or (np.isclose(sil, best_sil) and inertia_by_k[k] < inertia_by_k.get(best_k, np.inf)):
            best_sil, best_k = sil, k

    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_norm)
    centroids = km.cluster_centers_

    rep_set = set(); old_to_rep = {}
    for c in range(best_k):
        idxs = np.where(labels == c)[0]
        pts  = X_norm[idxs]
        d    = np.linalg.norm(pts - centroids[c], axis=1)
        rep  = idxs[int(np.argmin(d))]
        rep_set.add(rep)
        for i in idxs:
            old_to_rep[i] = rep

    kept_indices = sorted(rep_set)
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[old_to_rep[i]] for i in range(n)}

    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    tuple_to_new_edge_idx = {}
    new_edges = []
    old_edge_to_new_edge = {}

    for old_idx, (e1, r, e2) in enumerate(old_edges):
        ne1 = old_ent_to_new.get(e1, e1)
        ne2 = old_ent_to_new.get(e2, e2)
        tup = (ne1, int(r), ne2)
        if tup not in tuple_to_new_edge_idx:
            tuple_to_new_edge_idx[tup] = len(new_edges)
            new_edges.append([ne1, int(r), ne2])
        old_edge_to_new_edge[old_idx] = tuple_to_new_edge_idx[tup]

    def remap_edge_indices(struct):
        if isinstance(struct, list): return [remap_edge_indices(x) for x in struct]
        try: return old_edge_to_new_edge.get(int(struct), int(struct))
        except (ValueError, TypeError): return struct

    if codebook_main.get('questions_lst') is not None:
        codebook_main['questions_lst'] = remap_edge_indices(codebook_main['questions_lst'])
    if codebook_main.get('answers_lst') is not None:
        codebook_main['answers_lst'] = remap_edge_indices(codebook_main['answers_lst'])
    if use_thinking and codebook_main.get('thinkings_lst') is not None:
        codebook_main['thinkings_lst'] = remap_edge_indices(codebook_main['thinkings_lst'])

    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]
    # expose chosen k so we can print it
    codebook_main['_chosen_k'] = best_k
    return codebook_main


# -----------------------
# Optimized fast version
# -----------------------
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

def _normalize_rows_f32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order='C')
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    X /= norms
    return X

def _choose_k_candidates(n: int, min_exp_num: int, max_exp_num: int, num_ks: int = 8) -> List[int]:
    k_low  = max(2, int(np.ceil(n / max_exp_num)))
    k_high = max(2, min(n - 1, int(np.floor(n / min_exp_num))))
    if k_low > k_high:
        return sorted(set([2, min(5, n-1)]))
    target_size = int(np.sqrt(min_exp_num * max_exp_num))
    target_k = max(2, min(n - 1, int(round(n / max(target_size, 1)))))
    sizes = np.geomspace(max_exp_num, min_exp_num, num=num_ks)
    ks = set()
    for s in sizes:
        k = int(np.clip(int(np.ceil(n / s)), k_low, k_high))
        ks.add(k)
    ks.add(int(np.clip(target_k, k_low, k_high)))
    return sorted(ks)

def _score_k(X_sample_norm: np.ndarray, labels: np.ndarray, method: str = "db") -> float:
    if method == "silhouette":
        return silhouette_score(X_sample_norm, labels, metric='euclidean')  # higher is better
    else:
        return -davies_bouldin_score(X_sample_norm, labels)                 # negate → higher is better

def combine_ents_fast(codebook_main: Dict[str, Any],
                      min_exp_num: int = 2,
                      max_exp_num: int = 20,
                      use_thinking: bool = True,
                      random_state: int = 0,
                      sample_size: int = 3000,
                      k_grid_size: int = 8,
                      scoring: str = "db",          # "db" (fast) or "silhouette"
                      mbk_batch: int = 8192,
                      mbk_iters: int = 100) -> Dict[str, Any]:

    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n = X.shape[0]
    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main

    rng = np.random.default_rng(random_state)
    X_norm = _normalize_rows_f32(X)

    # ---- choose candidate ks
    cand_ks = _choose_k_candidates(n, min_exp_num, max_exp_num, num_ks=k_grid_size)
    max_k = max(cand_ks)

    # ---- ensure the scoring sample is large enough: n_samples >= n_clusters
    eff_sample_size = min(n, max(sample_size, int(1.2 * (max_k + 1))))  # 1.2x slack
    if n <= eff_sample_size:
        idx_sample = np.arange(n, dtype=np.int64)
        X_sample_norm = X_norm
    else:
        idx_sample = rng.choice(n, size=eff_sample_size, replace=False)
        X_sample_norm = X_norm[idx_sample]

    # helper so MiniBatchKMeans batches are always >= k
    batch_for = lambda k: max(k, min(mbk_batch, max(1000, k * 4)))

    # ---- score ks on the sample
    best_k, best_score = None, -np.inf
    for k in cand_ks:
        mbk = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_for(k),
            n_init=5,
            max_iter=mbk_iters,
            random_state=random_state,
            reassignment_ratio=0.01
        )
        labels = mbk.fit_predict(X_sample_norm)
        s = _score_k(X_sample_norm, labels, method=scoring)
        if s > best_score:
            best_score, best_k = s, k

    # ---- final clustering on full data
    mbk_full = MiniBatchKMeans(
        n_clusters=best_k,
        batch_size=batch_for(best_k),
        n_init=5,
        max_iter=mbk_iters,
        random_state=random_state,
        reassignment_ratio=0.01
    )
    labels_full = mbk_full.fit_predict(X_norm)
    centroids = mbk_full.cluster_centers_

    # ---- representative per cluster (skip empty clusters)
    rep_set: set[int] = set()
    old_to_rep: Dict[int, int] = {}

    for c in range(best_k):
        idxs = np.flatnonzero(labels_full == c)   # may be empty with MiniBatchKMeans
        if idxs.size == 0:
            continue  # skip empty cluster
        pts = X_norm[idxs]
        d = np.linalg.norm(pts - centroids[c], axis=1)
        rep = idxs[int(np.argmin(d))]
        rep_set.add(rep)
        for i in idxs:
            old_to_rep[i] = rep

    # Safety: in pathological cases, ensure we keep at least one entity
    if not rep_set:
        rep_set.add(0)
        old_to_rep = {i: 0 for i in range(n)}

    kept_indices = sorted(rep_set)
    rep_to_new: Dict[int, int] = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new: Dict[int, int] = {i: rep_to_new[old_to_rep[i]] for i in range(n)}

    # ---- rebuild entities & embeddings
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    # ---- remap & dedup edges
    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    tuple_to_new_edge_idx: Dict[Tuple[int,int,int], int] = {}
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
        if isinstance(struct, (int, np.integer)):
            return old_edge_to_new_edge.get(int(struct), int(struct))
        return struct

    if codebook_main.get('questions_lst') is not None:
        codebook_main['questions_lst'] = remap_edge_indices(codebook_main['questions_lst'])
    if codebook_main.get('answers_lst') is not None:
        codebook_main['answers_lst'] = remap_edge_indices(codebook_main['answers_lst'])
    if use_thinking and codebook_main.get('thinkings_lst') is not None:
        codebook_main['thinkings_lst'] = remap_edge_indices(codebook_main['thinkings_lst'])

    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]
    return codebook_main




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import FAISS and RAPIDS
FAISS_AVAILABLE = False
RAPIDS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available")
except ImportError:
    logger.info("FAISS not available, will use CPU clustering")

try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.metrics.cluster import davies_bouldin_score as cu_davies_bouldin_score
    from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
    import cupy as cp
    RAPIDS_AVAILABLE = True
    logger.info("RAPIDS cuML is available for GPU acceleration")
except ImportError:
    logger.info("RAPIDS cuML not available")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available")
except ImportError:
    logger.info("PyTorch not available")


class DeviceAwareClusterer:
    """Clustering that automatically uses GPU if available, otherwise CPU"""
    
    def __init__(self, backend='auto'):
        """
        Initialize clusterer with automatic device detection.
        
        Args:
            backend: 'auto', 'faiss', 'rapids', 'torch', or 'cpu'
        """
        self.backend = self._select_backend(backend)
        self.device_info = self._get_device_info()
        logger.info(f"Using backend: {self.backend}")
        logger.info(f"Device info: {self.device_info}")
    
    def _select_backend(self, backend):
        """Select the best available backend"""
        if backend == 'auto':
            if RAPIDS_AVAILABLE and self._check_cuda_available():
                return 'rapids'
            elif FAISS_AVAILABLE and self._check_cuda_available():
                return 'faiss'
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                return 'torch'
            else:
                return 'cpu'
        elif backend == 'faiss' and not FAISS_AVAILABLE:
            logger.warning("FAISS requested but not available, falling back to CPU")
            return 'cpu'
        elif backend == 'rapids' and not RAPIDS_AVAILABLE:
            logger.warning("RAPIDS requested but not available, falling back to CPU")
            return 'cpu'
        elif backend == 'torch' and not TORCH_AVAILABLE:
            logger.warning("PyTorch requested but not available, falling back to CPU")
            return 'cpu'
        return backend
    
    def _check_cuda_available(self):
        """Check if CUDA is available through various methods"""
        if FAISS_AVAILABLE:
            return faiss.get_num_gpus() > 0
        if TORCH_AVAILABLE:
            return torch.cuda.is_available()
        return False
    
    def _get_device_info(self):
        """Get information about available devices"""
        info = {'backend': self.backend}
        
        if self.backend == 'rapids':
            import cupy as cp
            info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
            info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            info['gpu_memory'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        elif self.backend == 'faiss':
            info['gpu_count'] = faiss.get_num_gpus()
        elif self.backend == 'torch':
            info['gpu_count'] = torch.cuda.device_count()
            if info['gpu_count'] > 0:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
        else:
            info['gpu_count'] = 0
            info['device'] = 'CPU'
        
        return info
    
    def cluster(self, X, n_clusters, n_init=5, max_iter=100, random_state=0):
        """
        Perform clustering using the best available backend.
        
        Returns:
            labels: cluster labels
            centroids: cluster centroids
            inertia: sum of squared distances to centroids
        """
        if self.backend == 'rapids':
            return self._cluster_rapids(X, n_clusters, n_init, max_iter, random_state)
        elif self.backend == 'faiss':
            return self._cluster_faiss(X, n_clusters, max_iter, random_state)
        elif self.backend == 'torch':
            return self._cluster_torch(X, n_clusters, n_init, max_iter, random_state)
        else:
            return self._cluster_cpu(X, n_clusters, n_init, max_iter, random_state)
    
    def _cluster_rapids(self, X, n_clusters, n_init, max_iter, random_state):
        """Cluster using RAPIDS cuML on GPU"""
        # Convert to CuPy array
        X_gpu = cp.asarray(X, dtype=cp.float32)
        
        # Use cuML KMeans
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            output_type='numpy'
        )
        
        labels = kmeans.fit_predict(X_gpu)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        # Convert back to numpy if needed
        if isinstance(labels, cp.ndarray):
            labels = cp.asnumpy(labels)
        if isinstance(centroids, cp.ndarray):
            centroids = cp.asnumpy(centroids)
            
        return labels, centroids, inertia
    
    def _cluster_faiss(self, X, n_clusters, max_iter, random_state):
        """Cluster using FAISS on GPU"""
        X = np.ascontiguousarray(X.astype(np.float32))
        d = X.shape[1]
        
        # Create clustering object
        kmeans = faiss.Kmeans(
            d, n_clusters,
            niter=max_iter,
            seed=random_state,
            gpu=self.device_info.get('gpu_count', 0) > 0
        )
        
        # Train
        kmeans.train(X)
        
        # Get labels
        _, labels = kmeans.index.search(X, 1)
        labels = labels.ravel()
        
        # Get centroids
        centroids = kmeans.centroids
        
        # Calculate inertia
        distances, _ = kmeans.index.search(X, 1)
        inertia = float(np.sum(distances))
        
        return labels, centroids, inertia
    
    def _cluster_torch(self, X, n_clusters, n_init, max_iter, random_state):
        """Cluster using PyTorch on GPU"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
        
        best_inertia = float('inf')
        best_labels = None
        best_centroids = None
        
        torch.manual_seed(random_state)
        
        for _ in range(n_init):
            # Initialize centroids using k-means++
            centroids = self._kmeans_plusplus_torch(X_torch, n_clusters)
            
            for _ in range(max_iter):
                # Assign points to nearest centroid
                distances = torch.cdist(X_torch, centroids)
                labels = torch.argmin(distances, dim=1)
                
                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.any():
                        new_centroids[k] = X_torch[mask].mean(dim=0)
                    else:
                        new_centroids[k] = centroids[k]
                
                # Check convergence
                if torch.allclose(centroids, new_centroids, rtol=1e-4):
                    break
                centroids = new_centroids
            
            # Calculate inertia
            distances = torch.cdist(X_torch, centroids)
            min_distances = torch.min(distances, dim=1)[0]
            inertia = torch.sum(min_distances ** 2).item()
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.cpu().numpy()
                best_centroids = centroids.cpu().numpy()
        
        return best_labels, best_centroids, best_inertia
    
    def _kmeans_plusplus_torch(self, X, n_clusters):
        n_samples = X.shape[0]
        centroids = torch.empty((n_clusters, X.shape[1]), device=X.device)
        idx0 = torch.randint(n_samples, ())
        centroids[0] = X[idx0]
        for c in range(1, n_clusters):
            distances = torch.cdist(X, centroids[:c])
            min_distances = torch.min(distances, dim=1)[0]
            probs = (min_distances ** 2)
            probs = probs / probs.sum()
            cumprobs = torch.cumsum(probs, dim=0)
            r = torch.rand((), device=X.device)
            idx = torch.searchsorted(cumprobs, r).item()
            centroids[c] = X[idx]
        return centroids
    
    def _cluster_cpu(self, X, n_clusters, n_init, max_iter, random_state):
        """Fallback to CPU clustering using sklearn"""
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            batch_size=min(8192, len(X))
        )
        
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        
        return labels, centroids, inertia
    
    def score(self, X, labels, method='davies_bouldin'):
        """Calculate clustering score using appropriate backend"""
        if self.backend == 'rapids' and method in ['davies_bouldin', 'silhouette']:
            return self._score_rapids(X, labels, method)
        else:
            return self._score_cpu(X, labels, method)
    
    def _score_rapids(self, X, labels, method):
        """Score using RAPIDS cuML on GPU"""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        labels_gpu = cp.asarray(labels, dtype=cp.int32)
        
        if method == 'davies_bouldin':
            score = cu_davies_bouldin_score(X_gpu, labels_gpu)
        else:  # silhouette
            score = cu_silhouette_score(X_gpu, labels_gpu)
        
        return float(score) if isinstance(score, cp.ndarray) else score
    
    def _score_cpu(self, X, labels, method):
        """Score using sklearn on CPU"""
        if method == 'davies_bouldin':
            return davies_bouldin_score(X, labels)
        else:  # silhouette
            return silhouette_score(X, labels)


def combine_ents_auto(
    codebook_main: Dict[str, Any],
    min_exp_num: int = 2,
    max_exp_num: int = 20,
    use_thinking: bool = True,
    use_facts: bool = True,
    random_state: int = 0,
    sample_size_prop: float = 0.2,  # fraction for k selection
    k_grid_size: int = 8,
    scoring: str = "silhouette",
    backend: str = 'auto',
    word_emb=None,  # <-- required for embedding new/changed edges
) -> Dict[str, Any]:
    """
    GPU-accelerated entity clustering with automatic device selection.
    Incrementally maintains codebook_main['edge_matrix_embedding'].
    """

    # ---------- helpers ----------
    def _edge_texts(edge_matrix: List[List[int]], E: List[str], R: List[str]) -> List[str]:
        return [f"{E[h]} {R[r]} {E[t]}" for h, r, t in edge_matrix]

    # ---------- init ----------
    E = list(codebook_main.get('e', []))
    R = list(codebook_main.get('r', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n = X.shape[0]

    # trivial case: keep edge embeddings as-is
    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        # leave codebook_main['edge_matrix_embedding'] untouched
        return codebook_main

    # ---------- backend clusterer ----------
    clusterer = DeviceAwareClusterer(backend=backend)
    logger.info(f"Clustering {n} entities using {clusterer.backend} backend")

    # cosine-normalize
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # candidate k’s from expected cluster sizes
    k_low  = max(2, int(np.ceil(n / max_exp_num)))
    k_high = max(2, min(n - 1, int(np.floor(n / min_exp_num))))
    if k_low > k_high:
        k_low, k_high = 2, max(2, min(n - 1, 5))

    sizes   = np.geomspace(max_exp_num, min_exp_num, num=k_grid_size)
    cand_ks = sorted(set(int(np.clip(int(np.ceil(n / s)), k_low, k_high)) for s in sizes))

    # sample for k-selection
    rng = np.random.default_rng(random_state)
    max_k = max(cand_ks)
    eff_sample = min(n, max(int(sample_size_prop * n), int(1.2 * (max_k + 1))))
    if n <= eff_sample:
        X_sample_norm = X_norm
    else:
        idx_sample = rng.choice(n, size=eff_sample, replace=False)
        X_sample_norm = X_norm[idx_sample]

    # pick best k
    best_k, best_score = None, -np.inf
    score_method = 'davies_bouldin' if scoring == 'db' else 'silhouette'
    for k in cand_ks:
        labels, _, _ = clusterer.cluster(X_sample_norm, k, n_init=3, max_iter=100, random_state=random_state)
        try:
            score = clusterer.score(X_sample_norm, labels, method=score_method)
            if score_method == 'davies_bouldin':
                score = -score
        except Exception as e:
            logger.debug(f"Score failed for k={k}: {e}")
            continue
        logger.debug(f"k={k}, score={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    logger.info(f"Selected k={best_k} with score={best_score:.4f}")

    # final clustering
    labels_full, centroids, _ = clusterer.cluster(
        X_norm,
        n_clusters=int(best_k),
        n_init=5,
        max_iter=200,
        random_state=random_state
    )

    # representatives = nearest to centroid per cluster
    rep_set: Set[int] = set()
    old_to_rep: Dict[int, int] = {}
    for c in range(best_k):
        idxs = np.where(labels_full == c)[0]
        if len(idxs) == 0:
            continue
        pts = X_norm[idxs]
        d = np.linalg.norm(pts - centroids[c], axis=1)
        rep = int(idxs[int(np.argmin(d))])
        rep_set.add(rep)
        for i in idxs:
            old_to_rep[int(i)] = rep

    if not rep_set:
        rep_set.add(0)
        old_to_rep = {i: 0 for i in range(n)}

    kept_indices = sorted(rep_set)
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[old_to_rep[i]] for i in range(n)}

    # rebuild E / embeddings
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    # ---------- remap edges ----------
    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    tuple_to_new_edge_idx: Dict[Tuple[int, int, int], int] = {}
    new_edges: List[List[int]] = []
    old_edge_to_new_edge: Dict[int, int] = {}

    for old_idx, (h, r, t) in enumerate(old_edges):
        nh = old_ent_to_new.get(h, h)
        nt = old_ent_to_new.get(t, t)
        tup = (nh, int(r), nt)
        if tup not in tuple_to_new_edge_idx:
            tuple_to_new_edge_idx[tup] = len(new_edges)
            new_edges.append([nh, int(r), nt])
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

    # ---------- incremental edge_matrix_embedding ----------
    old_edge_embeds = codebook_main.get('edge_matrix_embedding', None)
    have_old_edge_embeds = isinstance(old_edge_embeds, list) and len(old_edge_embeds) == len(old_edges)

    # Build old/new edge texts
    old_edge_texts = _edge_texts(old_edges, E, R) if have_old_edge_embeds else None
    new_edge_texts = _edge_texts(new_edges, new_e, R)

    new_edge_embeds_list: List[Optional[np.ndarray]] = [None] * len(new_edges)

    # Reuse embeddings when surface text matches
    if have_old_edge_embeds and old_edge_texts is not None:
        for old_idx, new_idx in old_edge_to_new_edge.items():
            if new_edge_embeds_list[new_idx] is None and old_edge_texts[old_idx] == new_edge_texts[new_idx]:
                new_edge_embeds_list[new_idx] = np.asarray(old_edge_embeds[old_idx], dtype=np.float32)

    # Embed the rest
    to_embed = [i for i, v in enumerate(new_edge_embeds_list) if v is None]
    if to_embed:
        if word_emb is None:
            raise ValueError("word_emb is required to embed new/changed edges.")
        embeds = get_word_embeddings([new_edge_texts[i] for i in to_embed], word_emb)
        for i, emb in zip(to_embed, embeds):
            new_edge_embeds_list[i] = np.asarray(emb, dtype=np.float32)

    edge_matrix_embedding = [v for v in new_edge_embeds_list]  # materialize

    # ---------- commit ----------
    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]
    codebook_main['edge_matrix_embedding'] = edge_matrix_embedding

    return codebook_main




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


# helpers 

def get_word_embeddings(list_of_text,word_emb):
    """
    list_of_text: ['str1 str2 ...',]
    word_emb: embedding model

    list_of_text_embeddings:  [embedding_vals,...]
    """
    # Check if it's HuggingFaceEmbeddings or Word2VecEmbeddings
    if hasattr(word_emb, '_embed_text'):
        # Word2VecEmbeddings or WordAvgEmbeddings
        list_of_text_embeddings = [word_emb._embed_text(text) for text in list_of_text]
    elif hasattr(word_emb, 'embed_documents'):
        # HuggingFaceEmbeddings
        list_of_text_embeddings = word_emb.embed_documents(list_of_text)
    else:
        raise AttributeError(f"Unsupported embedding model type: {type(word_emb)}")

    # Ensure all embeddings are numpy arrays with consistent shape
    list_of_text_embeddings = [np.asarray(emb, dtype=np.float32) for emb in list_of_text_embeddings]
    
    return list_of_text_embeddings

# -----------------------
# Fixed combine_ents_ann_knn (cosine similarities used correctly)
# -----------------------
def combine_ents_ann_knn(
    codebook_main: Dict[str, Any],
    config: Optional[ClusteringConfig] = None,
    use_thinking: bool = True,
    use_facts: bool = True,
    sim_threshold: float = 0.9,
    word_emb=None,  # <-- embedder for incremental edge embeddings
) -> Dict[str, Any]:
    """
    Entity clustering using ANN k-NN graph + Union-Find with consistent cosine handling.
    Incrementally maintains codebook_main['edge_matrix_embedding'] by reusing old embeddings
    whenever the new edge's triple text equals the old one; only embeds the missing ones.
    """
    # ---- helpers
    def _edge_texts(edge_matrix, E, R):
        # edge_matrix: List[List[int]] with [h, r, t]
        return [f"{E[h]} {R[r]} {E[t]}" for h, r, t in edge_matrix]

    # Initialize configuration
    if config is None:
        n = len(codebook_main.get('e', []))
        k = min(50, max(5, int(np.sqrt(max(1, n)))))  # Adaptive k
        config = ClusteringConfig(k_neighbors=k, similarity_threshold=sim_threshold, min_cluster_size=2)

    # Get entities/embeddings
    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n, d = X.shape if X.ndim == 2 else (len(X), 1)

    if n <= 2:
        # Nothing to cluster; still ensure edge_matrix is int lists and keep existing edge embeddings as-is
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        # edge_matrix_embedding unchanged
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

    # Build ANN graph -> indices, SIMILARITIES
    gb = ANNGraphBuilder(replace(config, metric=metric))
    indices, sims = gb.build_graph(X_unit)

    # Union-Find merge with similarity threshold
    uf = UnionFind(n)
    edges_created = 0
    if metric == "cosine":
        thr = float(config.similarity_threshold)
        for i in range(n):
            for j, s in zip(indices[i], sims[i]):
                if s >= thr and uf.union(i, int(j)):
                    edges_created += 1
    else:
        thr = float(config.similarity_threshold)
        if not (0.0 <= thr <= 1.0):
            thr = 0.5
        for i in range(n):
            for j, s in zip(indices[i], sims[i]):
                if s >= thr and uf.union(i, int(j)):
                    edges_created += 1

    logger.info(f"Created {edges_created} union edges at threshold {config.similarity_threshold}")

    if edges_created == 0:
        logger.info("No merges formed; returning original codebook unchanged.")
        return codebook_main

    # Build clusters & choose representatives
    clusters = uf.get_clusters()
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

    # Maps from old entity -> new entity index
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[representatives[i]] for i in range(n)}

    # Rebuild E / embeddings
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    # --- Prepare for incremental edge remap & embedding reuse
    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    old_edge_embeds = codebook_main.get('edge_matrix_embedding', None)
    have_old_edge_embeds = isinstance(old_edge_embeds, list) and len(old_edge_embeds) == len(old_edges)

    # If we can reuse, build old edge texts once (based on *old* E/R)
    if have_old_edge_embeds:
        old_edge_texts = _edge_texts(old_edges, E, codebook_main['r'])
    else:
        old_edge_texts = None

    # Remap edges and build mapping old_edge_idx -> new_edge_idx
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

    # Remap indices inside questions/answers/thinkings/facts
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

    # ---- Incremental edge_matrix_embedding maintenance
    # Build desired new edge texts using *new* E and existing R
    R = codebook_main.get('r', [])
    new_edge_texts = _edge_texts(new_edges, new_e, R)

    new_edge_embeds_list: List[Optional[np.ndarray]] = [None] * len(new_edges)

    # Try to reuse when text is identical
    if have_old_edge_embeds and old_edge_texts is not None:
        # For each old edge, see which new edge it maps to; if texts match, copy embedding
        for old_idx, new_idx in old_edge_to_new_edge.items():
            if new_edge_embeds_list[new_idx] is None and old_edge_texts[old_idx] == new_edge_texts[new_idx]:
                new_edge_embeds_list[new_idx] = np.asarray(old_edge_embeds[old_idx], dtype=np.float32)

    # Collect the ones that still need embedding
    to_embed_idx = [i for i, v in enumerate(new_edge_embeds_list) if v is None]

    if to_embed_idx:
        if word_emb is None:
            raise ValueError("word_emb is required to embed new/changed edges.")
        texts_to_embed = [new_edge_texts[i] for i in to_embed_idx]
        embeds = get_word_embeddings(texts_to_embed, word_emb)
        # Place them back
        for i, emb in zip(to_embed_idx, embeds):
            new_edge_embeds_list[i] = np.asarray(emb, dtype=np.float32)

    # Safety: no Nones remain
    edge_matrix_embedding = [v for v in new_edge_embeds_list]

    # ---- Commit results
    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]
    codebook_main['edge_matrix_embedding'] = edge_matrix_embedding

    return codebook_main



## new coarse combine will only do the aggressive merging if ram takes more then a threshold
def coarse_combine(codebook_main: Dict[str, Any],
                   min_exp_num: int = 2,
                   max_exp_num: int = 20,
                   use_thinking: bool = True,
                   use_facts: bool = True,
                   random_state: int = 0,
                   sample_size_prop: float = 0.2,
                   k_grid_size: int = 8,
                   scoring: str = "silhouette",
                   backend: str = 'auto',
                   config: Optional["ClusteringConfig"] = None,
                   ram_threshold: float = 70.0,  # percentage,
                   sim_threshold: float = 0.9
                   ):
    """
    Combines entities in two stages:
    1. KNN-based combine (always applied).
    2. Auto combine (only if RAM usage < ram_threshold).

    Parameters
    ----------
    ram_threshold : float
        Percentage of system RAM usage above which the second
        (aggressive) compression will be skipped.
    """

    # Stage 1: always do the lightweight combine
    filtered_codebook_main = combine_ents_ann_knn(
        codebook_main,
        config,
        use_thinking,
        use_facts,
        sim_threshold
    )

    # Check RAM usage before doing the aggressive combine
    ram_used_percent = psutil.virtual_memory().percent
    print(f"[INFO] Current RAM usage: {ram_used_percent:.2f}%")

    if ram_used_percent < ram_threshold:
        # Stage 2: apply powerful combine
        final_codebook_main = combine_ents_auto(
            filtered_codebook_main,
            min_exp_num,
            max_exp_num,
            use_thinking,
            use_facts,
            random_state,
            sample_size_prop,
            k_grid_size,
            scoring,
            backend
        )
    else:
        print(f"[WARN] RAM usage {ram_used_percent:.2f}% exceeds threshold {ram_threshold}%. "
              f"Skipping aggressive combine.")
        final_codebook_main = filtered_codebook_main

    return final_codebook_main
    

    
# -----------------------
# Synthetic data + timing
# -----------------------
def make_codebook(n=10_000, d=128, m_edges=50_000, seed=0):
    rng = np.random.default_rng(seed)
    # make a mixture so clustering isn't degenerate
    n_clusters = max(8, int(np.sqrt(n/50)))
    centers = rng.normal(size=(n_clusters, d)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    X = centers[assign] + 0.1 * rng.normal(size=(n, d)).astype(np.float32)
    e = list(range(n))
    # random edges (some duplicates likely)
    edge_matrix = np.column_stack([
        rng.integers(0, n, size=m_edges),
        rng.integers(0, 50, size=m_edges),          # relation ids
        rng.integers(0, n, size=m_edges)
    ]).tolist()
    return {
        'e': e,
        'e_embeddings': X,
        'edge_matrix': edge_matrix,
        'questions_lst': [list(rng.integers(0, m_edges, size=10)) for _ in range(50)],
        'answers_lst':   [list(rng.integers(0, m_edges, size=10)) for _ in range(50)],
        'thinkings_lst': [list(rng.integers(0, m_edges, size=10)) for _ in range(50)],
    }

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0
# if __name__ == "__main__":
#     # Small demo (should finish quickly)
#     N_SMALL, D = 10_000, 1
#     cb_small = make_codebook(N_SMALL, D, m_edges=50_000, seed=42)

#     print(f"\n=== SMALL DATASET: n={N_SMALL}, d={D} ===")
#     # fast
#     # cb_fast_s, t_fast_s = timed(combine_ents_fast, deepcopy(cb_small),
#     #                             min_exp_num=2, max_exp_num=20,
#     #                             random_state=0, sample_size=2000, k_grid_size=8, scoring="db")
#     # print(f"[fast]   time={t_fast_s:.2f}s, k={cb_fast_s['_chosen_k']}, |E'|={len(cb_fast_s['e'])}")

#     cb_fast_s, t_fast_s = timed(coarse_combine, deepcopy(cb_small),
#                                 min_exp_num=2, max_exp_num=20,
#                                 backend='auto',scoring = 'silhouette')
#     print(f"[fast new]   time={t_fast_s:.2f}s, |E'|={len(cb_fast_s['e'])}")


#     # config1 = ClusteringConfig(k_neighbors=10, similarity_threshold=0.05, ann_backend="auto")

#     # cb_ann, t_ann = timed(combine_ents_ann_knn, deepcopy(cb_small),
#     #                             config=config1)
#     # print(f"[fast new]   time={t_ann:.2f}s, |E'|={len(cb_ann['e'])}") 
#     N_SMALL, D = 10_000, 1
#     cb_small = make_codebook(N_SMALL, D, m_edges=50_000, seed=42)
#     config1 = ClusteringConfig(k_neighbors=10, similarity_threshold=0.5, ann_backend="auto", metric="cosine")
#     cb_ann, t_ann = timed(combine_ents_ann_knn, deepcopy(cb_small), config=config1)
#     print(f"[1D] time={t_ann:.2f}s, |E'|={len(cb_ann['e'])}")

#     # 2) Higher-D cosine (should merge a lot)
#     N2, D2 = 10_000, 80
#     cb2 = make_codebook(N2, D2, m_edges=50_000, seed=0)
#     config2 = ClusteringConfig(k_neighbors=5, similarity_threshold=0.95, ann_backend="auto", metric="cosine")
#     cb2_ann, t2 = timed(combine_ents_ann_knn, deepcopy(cb2), config=config2)
#     print(f"[64D] time={t2:.2f}s, |E'|={len(cb2_ann['e'])}")




# python py_files/combine_ent_cached.py
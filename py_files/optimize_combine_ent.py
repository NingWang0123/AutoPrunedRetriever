# ===== Benchmark: Original vs Fast combine_ents =====
import time, numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import warnings
import logging

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


def combine_ents_auto(codebook_main: Dict[str, Any],
                     min_exp_num: int = 2,
                     max_exp_num: int = 20,
                     use_thinking: bool = True,
                     random_state: int = 0,
                     sample_size_prop: float = 0.2, #
                     k_grid_size: int = 8,
                     scoring: str = "silhouette",
                     backend: str = 'auto') -> Dict[str, Any]:
    """
    GPU-accelerated entity clustering with automatic device selection.
    
    Args:
        codebook_main: Dictionary containing entities and embeddings
        min_exp_num: Minimum expected number of entities per cluster
        max_exp_num: Maximum expected number of entities per cluster
        use_thinking: Whether to remap thinking indices
        random_state: Random seed for reproducibility
        sample_size: Sample size for k selection
        k_grid_size: Number of k values to test
        scoring: 'db' for Davies-Bouldin or 'silhouette'
        backend: 'auto', 'faiss', 'rapids', 'torch', or 'cpu'
    
    Returns:
        Updated codebook with clustered entities
    """
    
    # Initialize entities and embeddings
    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n = X.shape[0]
    
    # Handle trivial case
    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main
    
    # Initialize clusterer with automatic device selection
    clusterer = DeviceAwareClusterer(backend=backend)
    logger.info(f"Clustering {n} entities using {clusterer.backend} backend")
    
    # Normalize embeddings
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    # Generate candidate k values
    k_low = max(2, int(np.ceil(n / max_exp_num)))
    k_high = max(2, min(n - 1, int(np.floor(n / min_exp_num))))
    if k_low > k_high:
        k_low, k_high = 2, max(2, min(n - 1, 5))
    
    # Use geometric spacing for k values
    sizes = np.geomspace(max_exp_num, min_exp_num, num=k_grid_size)
    cand_ks = sorted(set([ int(np.clip(int(np.ceil(n / s)), k_low, k_high)) for s in sizes ]))
        
    # Sample data for k selection if needed
    rng = np.random.default_rng(random_state)
    max_k = max(cand_ks)
    eff_sample_size = min(n, max(int(sample_size_prop*n), int(1.2 * (max_k + 1))))
    
    if n <= eff_sample_size:
        X_sample_norm = X_norm
    else:
        idx_sample = rng.choice(n, size=eff_sample_size, replace=False)
        X_sample_norm = X_norm[idx_sample]
    
    # Find best k
    best_k, best_score = None, -np.inf
    score_method = 'davies_bouldin' if scoring == 'db' else 'silhouette'

    for k in cand_ks:
        labels, _, _ = clusterer.cluster(X_sample_norm, k, n_init=3, max_iter=100, random_state=random_state)
        try:
            score = clusterer.score(X_sample_norm, labels, method=score_method)
            if score_method == 'davies_bouldin':
                score = -score
            print(f'k={k}, score={score:.4f}')
        except Exception as e:
            logger.debug(f"Score failed for k={k}: {e}")
            continue
        logger.debug(f"k={k}, score={score:.4f}")

        if score > best_score:
            best_score, best_k = score, k
    
    
    logger.info(f"Selected k={best_k} with score={best_score:.4f}")
    
    # Final clustering on full data
    labels_full, centroids, _ = clusterer.cluster(
        X_norm,
        n_clusters=int(best_k),
        n_init=5,
        max_iter=200,
        random_state=random_state
    )
    
    # Find representatives for each cluster
    rep_set = set()
    old_to_rep = {}
    
    for c in range(best_k):
        idxs = np.where(labels_full == c)[0]
        if len(idxs) == 0:
            continue
        
        pts = X_norm[idxs]
        # Calculate distances to centroid
        d = np.linalg.norm(pts - centroids[c], axis=1)
        rep = idxs[int(np.argmin(d))]
        rep_set.add(int(rep))
        
        for i in idxs:
            old_to_rep[int(i)] = int(rep)
    
    # Safety check
    if not rep_set:
        rep_set.add(0)
        old_to_rep = {i: 0 for i in range(n)}
    
    # Create mapping
    kept_indices = sorted(rep_set)
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[old_to_rep[i]] for i in range(n)}
    
    # Rebuild entities and embeddings
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) 
                 for i in kept_indices]
    
    # Remap edges
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
    
    # Remap function
    def remap_edge_indices(struct):
        if isinstance(struct, list):
            return [remap_edge_indices(x) for x in struct]
        try:
            return old_edge_to_new_edge.get(int(struct), int(struct))
        except (ValueError, TypeError):
            return struct
    
    # Update codebook
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



import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
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


class RepresentativeSelector:
    """Select representatives from clusters"""
    
    @staticmethod
    def select_medoid(X: np.ndarray, cluster_indices: List[int]) -> int:
        """Select point with minimum distance to all others (medoid)"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        
        X_cluster = X[cluster_indices]
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import pairwise_distances
        dists = pairwise_distances(X_cluster)
        
        # Find medoid (minimum sum of distances)
        medoid_idx = np.argmin(dists.sum(axis=1))
        return cluster_indices[medoid_idx]
    
    @staticmethod
    def select_density_peak(X: np.ndarray, cluster_indices: List[int], k: int = 5) -> int:
        """Select point with highest local density"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        
        X_cluster = X[cluster_indices]
        
        # Compute k-NN distances for density estimation
        k = min(k, len(cluster_indices) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(X_cluster)
        distances, _ = nbrs.kneighbors(X_cluster)
        
        # Density = 1 / average k-NN distance
        densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
        
        # Select highest density point
        peak_idx = np.argmax(densities)
        return cluster_indices[peak_idx]


def combine_ents_ann_knn(
    codebook_main: Dict[str, Any],
    config: Optional[ClusteringConfig] = None,
    min_exp_num: int = 2,
    max_exp_num: int = 20,
    use_thinking: bool = True
) -> Dict[str, Any]:
    """
    Entity clustering using ANN k-NN graph + Union-Find.
    
    This is faster and more flexible than k-means clustering.
    
    Args:
        codebook_main: Dictionary containing entities and embeddings
        config: ClusteringConfig object with parameters
        min_exp_num: Minimum expected entities per cluster (used to set k)
        max_exp_num: Maximum expected entities per cluster (used to set k)
        use_thinking: Whether to remap thinking indices
    
    Returns:
        Updated codebook with merged entities
    """
    
    # Initialize configuration
    if config is None:
        # Auto-configure based on data size
        n = len(codebook_main.get('e', []))
        k = min(50, max(5, int(np.sqrt(n))))  # Adaptive k
        config = ClusteringConfig(
            k_neighbors=k,
            similarity_threshold=0.75,  # Adjust based on your data
            min_cluster_size=2
        )
    
    # Get entities and embeddings
    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)
    n = X.shape[0]
    
    # Handle trivial cases
    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main
    
    logger.info(f"Clustering {n} entities using ANN k-NN + Union-Find")
    
    # Normalize embeddings for cosine similarity
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    # Build ANN graph
    graph_builder = ANNGraphBuilder(config)
    indices, distances = graph_builder.build_graph(X_norm)
    
    # Convert distances to similarities (for cosine)
    if config.metric == "cosine":
        similarities = 1 - distances
    else:
        # For Euclidean, use Gaussian kernel
        sigma = np.median(distances)
        similarities = np.exp(-distances**2 / (2 * sigma**2))
    
    # Initialize Union-Find
    uf = UnionFind(n)
    
    # Connect nodes with similarity >= threshold
    edges_created = 0
    for i in range(n):
        for j, sim in zip(indices[i], similarities[i]):
            if sim >= config.similarity_threshold:
                if uf.union(i, j):
                    edges_created += 1
    
    logger.info(f"Created {edges_created} edges with threshold {config.similarity_threshold}")
    
    # Get clusters
    clusters = uf.get_clusters()
    
    # Filter small clusters and select representatives
    rep_selector = RepresentativeSelector()
    representatives = {}  # old_idx -> representative_idx
    kept_indices = []
    
    for root, members in clusters.items():
        if len(members) < config.min_cluster_size:
            # Keep singleton clusters as-is
            for m in members:
                representatives[m] = m
                kept_indices.append(m)
        else:
            # Select representative
            if config.representative_method == "medoid":
                rep = rep_selector.select_medoid(X_norm, members)
            else:  # density
                rep = rep_selector.select_density_peak(X_norm, members)
            
            kept_indices.append(rep)
            for m in members:
                representatives[m] = rep
    
    # Remove duplicates and sort
    kept_indices = sorted(set(kept_indices))
    
    logger.info(f"Reduced from {n} to {len(kept_indices)} entities")
    logger.info(f"Number of clusters: {len(clusters)}")
    
    # Create mappings
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[representatives[i]] for i in range(n)}
    
    # Rebuild entities and embeddings
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) 
                 for i in kept_indices]
    
    # Remap edges (same as before)
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
    
    # Remap function
    def remap_edge_indices(struct):
        if isinstance(struct, list):
            return [remap_edge_indices(x) for x in struct]
        try:
            return old_edge_to_new_edge.get(int(struct), int(struct))
        except (ValueError, TypeError):
            return struct
    
    # Update codebook
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
#     N_SMALL, D = 10000, 96
#     cb_small = make_codebook(N_SMALL, D, m_edges=50000, seed=42)

#     print(f"\n=== SMALL DATASET: n={N_SMALL}, d={D} ===")
#     # fast
#     # cb_fast_s, t_fast_s = timed(combine_ents_fast, deepcopy(cb_small),
#     #                             min_exp_num=2, max_exp_num=20,
#     #                             random_state=0, sample_size=2000, k_grid_size=8, scoring="db")
#     # print(f"[fast]   time={t_fast_s:.2f}s, k={cb_fast_s['_chosen_k']}, |E'|={len(cb_fast_s['e'])}")

#     cb_fast_s, t_fast_s = timed(combine_ents_auto, deepcopy(cb_small),
#                                 min_exp_num=2, max_exp_num=20,
#                                 backend='auto',scoring = 'silhouette')
#     print(f"[fast new]   time={t_fast_s:.2f}s, |E'|={len(cb_fast_s['e'])}")


#     config1 = ClusteringConfig(k_neighbors=10, similarity_threshold=0.2, ann_backend="auto")

#     cb_ann, t_ann = timed(combine_ents_ann_knn, deepcopy(cb_small),
#                                 config=config1)
#     print(f"[fast new]   time={t_ann:.2f}s, |E'|={len(cb_ann['e'])}") 




# python py_files/optimize_combine_ent.py
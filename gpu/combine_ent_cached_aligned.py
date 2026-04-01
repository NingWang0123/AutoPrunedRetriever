import gc, time, numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Any, Dict, List, Tuple, Optional, Set, Literal
import warnings
import logging
import psutil
from dataclasses import dataclass, replace

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


# ============================================================
# Embedding helpers
# ============================================================

def get_word_embeddings(list_of_text, word_emb):
    """
    list_of_text: ['str1 str2 ...',]
    word_emb: embedding model
    list_of_text_embeddings:  [embedding_vals,...]
    """
    if hasattr(word_emb, '_embed_text'):
        list_of_text_embeddings = [word_emb._embed_text(text) for text in list_of_text]
    elif hasattr(word_emb, 'embed_documents'):
        list_of_text_embeddings = word_emb.embed_documents(list_of_text)
    else:
        raise AttributeError(f"Unsupported embedding model type: {type(word_emb)}")
    list_of_text_embeddings = [np.asarray(emb, dtype=np.float32) for emb in list_of_text_embeddings]
    return list_of_text_embeddings


# ============================================================
# Incremental embedding rebuild helpers
# ============================================================

def _edge_text(h: int, r: int, t: int, E: List[str], R: List[str]) -> str:
    h_s = E[h] if h < len(E) else f"ent_{h}"
    r_s = R[r] if r < len(R) else f"rel_{r}"
    t_s = E[t] if t < len(E) else f"ent_{t}"
    return f"{h_s} {r_s} {t_s}"


def _chunk_full_text(edge_indices: List[int], codebook: Dict[str, Any]) -> str:
    E = codebook["e"]
    R = codebook["r"]
    EM = codebook["edge_matrix"]
    parts = []
    for ei in edge_indices:
        h, r, t = EM[ei]
        parts.append(f"[H]{E[h]} [R]{R[r]} [T]{E[t]}")
    return " <SEP> ".join(parts)


def _rebuild_edge_matrix_embedding_incremental(
    codebook_main: Dict[str, Any],
    old_edges: List[List[int]],
    old_edge_emb,
    old_E: List[str],
    old_R: List[str],
    old_edge_to_new_edge: Dict[int, int],
    word_emb,
) -> np.ndarray:
    new_edges = codebook_main['edge_matrix']
    new_E = codebook_main['e']
    new_R = codebook_main['r']
    n_new = len(new_edges)

    if old_edge_emb is None or (hasattr(old_edge_emb, '__len__') and len(old_edge_emb) == 0):
        if n_new == 0:
            _d = _infer_embedding_dim(codebook_main, None)
            return np.zeros((0, max(_d, 1)), dtype=np.float32)
        texts = [_edge_text(h, r, t, new_E, new_R) for h, r, t in new_edges]
        return np.asarray(get_word_embeddings(texts, word_emb), dtype=np.float32)

    old_emb = np.asarray(old_edge_emb, dtype=np.float32)
    if old_emb.ndim != 2:
        old_emb = old_emb.reshape(len(old_edges), -1)
    d = old_emb.shape[1]

    if n_new == 0:
        return np.zeros((0, d), dtype=np.float32)

    new_to_old: Dict[int, List[int]] = {}
    for old_idx, new_idx in old_edge_to_new_edge.items():
        new_to_old.setdefault(new_idx, []).append(old_idx)

    new_texts = [_edge_text(h, r, t, new_E, new_R) for h, r, t in new_edges]

    # Free caches to reduce memory pressure before large allocation
    codebook_main.pop('_base_emb_np_cache', None)
    codebook_main.pop('_norm_r_matrix', None)
    import gc; gc.collect()

    result = np.zeros((n_new, d), dtype=np.float32)
    needs_compute: List[int] = []

    for new_idx in range(n_new):
        reused = False
        for old_idx in new_to_old.get(new_idx, []):
            if old_idx >= len(old_edges) or old_idx >= old_emb.shape[0]:
                continue
            oh, or_, ot = old_edges[old_idx]
            if _edge_text(oh, or_, ot, old_E, old_R) == new_texts[new_idx]:
                result[new_idx] = old_emb[old_idx]
                reused = True
                break
        if not reused:
            needs_compute.append(new_idx)

    if needs_compute:
        print(f"[combine] Embedding {len(needs_compute)} changed edges …")
        texts_to_embed = [new_texts[i] for i in needs_compute]
        new_vecs = np.asarray(get_word_embeddings(texts_to_embed, word_emb), dtype=np.float32)
        for j, ni in enumerate(needs_compute):
            result[ni] = new_vecs[j]

    logger.info(f"edge_matrix_embedding: reused {n_new - len(needs_compute)}/{n_new}, "
                f"recomputed {len(needs_compute)}")
    # Store which edges were recomputed for fast GPU patching in remap
    codebook_main['_recomputed_edge_indices'] = needs_compute
    return result


def _infer_embedding_dim(codebook_main: Dict[str, Any], old_embs) -> int:
    """Infer embedding dimension from edge_matrix_embedding or existing group embeddings."""
    # Try edge_matrix_embedding first (always a reliable source)
    eme = codebook_main.get('edge_matrix_embedding')
    if eme is not None:
        a = np.asarray(eme)
        if a.ndim == 2 and a.shape[1] > 0:
            return int(a.shape[1])
    # Try existing group embeddings
    if old_embs:
        for g in old_embs:
            a = np.asarray(g)
            if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] > 0:
                return int(a.shape[1])
            if a.ndim == 1 and a.shape[0] > 0:
                return int(a.shape[0])
    # Try e_embeddings
    ee = codebook_main.get('e_embeddings', [])
    if ee and len(ee) > 0:
        a = np.asarray(ee[0])
        if a.shape[0] > 0:
            return int(a.reshape(-1).shape[0])
    return 0


def _rebuild_lst_embeddings_incremental(
    codebook_main: Dict[str, Any],
    lst_key: str,
    emb_key: str,
    dirty_chunks: Set[Tuple[int, int]],
    word_emb,
) -> list:
    """Rebuild list embeddings using pre-computed dirty chunk set for O(1) lookup."""
    lst = codebook_main.get(lst_key)
    old_embs = codebook_main.get(emb_key)

    if lst is None:
        return []

    if old_embs is None or len(old_embs) == 0:
        return _compute_lst_embeddings_fresh(codebook_main, lst_key, word_emb)

    # Infer embedding dim from edge_matrix_embedding or existing group embeddings
    _dim = _infer_embedding_dim(codebook_main, old_embs)

    new_embs = []
    reused_count = 0
    recomputed_count = 0

    for gi, group in enumerate(lst):
        old_group_emb = old_embs[gi] if gi < len(old_embs) else None

        group_rows = []
        for ci, chunk_edges in enumerate(group):
            # O(1) set lookup instead of O(edges_per_chunk) scan
            chunk_changed = (gi, ci) in dirty_chunks

            if (not chunk_changed
                    and old_group_emb is not None
                    and ci < (len(old_group_emb) if hasattr(old_group_emb, '__len__') else old_group_emb.shape[0])):
                row = np.asarray(old_group_emb[ci], dtype=np.float32).reshape(-1)
                if row.shape[0] == 0 and _dim > 0:
                    row = np.zeros(_dim, dtype=np.float32)
                group_rows.append(row)
                reused_count += 1
            else:
                txt = _chunk_full_text(chunk_edges, codebook_main)
                vec = np.asarray(get_word_embeddings([txt], word_emb)[0], dtype=np.float32).reshape(-1)
                if vec.shape[0] > 0 and _dim == 0:
                    _dim = vec.shape[0]  # learn dim from first real embedding
                group_rows.append(vec)
                recomputed_count += 1

        if group_rows:
            new_embs.append(np.stack(group_rows, axis=0))
        else:
            new_embs.append(np.zeros((0, max(_dim, 1)), dtype=np.float32))

    logger.info(f"{emb_key}: reused {reused_count}, recomputed {recomputed_count}")
    return new_embs


def _compute_lst_embeddings_fresh(
    codebook_main: Dict[str, Any],
    lst_key: str,
    word_emb,
) -> list:
    lst = codebook_main.get(lst_key, [])
    _dim = _infer_embedding_dim(codebook_main, None)
    result = []
    for group in lst:
        rows = []
        for chunk_edges in group:
            txt = _chunk_full_text(chunk_edges, codebook_main)
            vec = np.asarray(get_word_embeddings([txt], word_emb)[0], dtype=np.float32).reshape(-1)
            if vec.shape[0] > 0 and _dim == 0:
                _dim = vec.shape[0]
            rows.append(vec)
        if rows:
            result.append(np.stack(rows, axis=0))
        else:
            result.append(np.zeros((0, max(_dim, 1)), dtype=np.float32))
    return result


def _find_changed_new_edges(
    old_edges: List[List[int]],
    old_E: List[str],
    old_R: List[str],
    new_EM: List[List[int]],
    new_E: List[str],
    new_R: List[str],
    old_edge_to_new_edge: Dict[int, int],
) -> Set[int]:
    new_to_old: Dict[int, List[int]] = {}
    for old_idx, new_idx in old_edge_to_new_edge.items():
        new_to_old.setdefault(new_idx, []).append(old_idx)

    changed: Set[int] = set()
    for new_idx in range(len(new_EM)):
        nh, nr, nt = new_EM[new_idx]
        new_text = _edge_text(nh, nr, nt, new_E, new_R)

        old_candidates = new_to_old.get(new_idx, [])
        found_match = False
        for old_idx in old_candidates:
            if old_idx < len(old_edges):
                oh, or_, ot = old_edges[old_idx]
                if _edge_text(oh, or_, ot, old_E, old_R) == new_text:
                    found_match = True
                    break
        if not found_match:
            changed.add(new_idx)

    return changed


def _build_dirty_chunks(
    codebook_main: Dict[str, Any],
    lst_key: str,
    changed_new_edges: Set[int],
) -> Set[Tuple[int, int]]:
    """
    Pre-compute which (group_idx, chunk_idx) contain changed edges.
    Returns a set of (gi, ci) tuples for O(1) lookup during rebuild.
    """
    dirty: Set[Tuple[int, int]] = set()
    lst = codebook_main.get(lst_key, [])
    for gi, group in enumerate(lst):
        for ci, chunk_edges in enumerate(group):
            if any(int(eidx) in changed_new_edges for eidx in chunk_edges):
                dirty.add((gi, ci))
    return dirty


def _update_all_cached_embeddings(
    codebook_main: Dict[str, Any],
    old_edges: List[List[int]],
    old_edge_emb,
    old_E: List[str],
    old_R: List[str],
    old_edge_to_new_edge: Dict[int, int],
    word_emb,
    use_thinking: bool = True,
    use_facts: bool = True,
):
    if word_emb is None:
        logger.warning("word_emb is None — skipping cached embedding updates after combine_ents.")
        return

    if codebook_main.get('edge_matrix_embedding') is not None or old_edge_emb is not None:
        codebook_main['edge_matrix_embedding'] = _rebuild_edge_matrix_embedding_incremental(
            codebook_main, old_edges, old_edge_emb, old_E, old_R,
            old_edge_to_new_edge, word_emb,
        )

    changed_new_edges = _find_changed_new_edges(
        old_edges, old_E, old_R,
        codebook_main['edge_matrix'], codebook_main['e'], codebook_main['r'],
        old_edge_to_new_edge,
    )

    # --- dirty-tracking: build dirty chunk sets ONCE, reuse across all lst rebuilds ---
    lst_emb_pairs = [
        ('questions_lst', 'questions_lst_embedding'),
        ('answers_lst', 'answers_lst_embedding'),
    ]
    if use_thinking:
        lst_emb_pairs.append(('thinkings_lst', 'thinkings_lst_embedding'))
    if use_facts:
        lst_emb_pairs.append(('facts_lst', 'facts_lst_embedding'))

    for lst_key, emb_key in lst_emb_pairs:
        if codebook_main.get(emb_key) is not None:
            dirty_chunks = _build_dirty_chunks(codebook_main, lst_key, changed_new_edges)
            codebook_main[emb_key] = _rebuild_lst_embeddings_incremental(
                codebook_main, lst_key, emb_key, dirty_chunks, word_emb,
            )

    if changed_new_edges:
        logger.info(f"Incremental embedding update: {len(changed_new_edges)} edges had text changes")
    else:
        logger.info("Incremental embedding update: no text changes detected, all embeddings reused")


# ============================================================
# DeviceAwareClusterer
# ============================================================

class DeviceAwareClusterer:
    """Clustering that automatically uses GPU if available, otherwise CPU"""

    def __init__(self, backend='auto'):
        self.backend = self._select_backend(backend)
        self.device_info = self._get_device_info()
        logger.info(f"Using backend: {self.backend}")
        logger.info(f"Device info: {self.device_info}")

    def _select_backend(self, backend):
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
        if FAISS_AVAILABLE:
            return faiss.get_num_gpus() > 0
        if TORCH_AVAILABLE:
            return torch.cuda.is_available()
        return False

    def _get_device_info(self):
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
        if self.backend == 'rapids':
            return self._cluster_rapids(X, n_clusters, n_init, max_iter, random_state)
        elif self.backend == 'faiss':
            return self._cluster_faiss(X, n_clusters, max_iter, random_state)
        elif self.backend == 'torch':
            return self._cluster_torch(X, n_clusters, n_init, max_iter, random_state)
        else:
            return self._cluster_cpu(X, n_clusters, n_init, max_iter, random_state)

    def _cluster_rapids(self, X, n_clusters, n_init, max_iter, random_state):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        kmeans = cuKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                          random_state=random_state, output_type='numpy')
        labels = kmeans.fit_predict(X_gpu)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        if isinstance(labels, cp.ndarray): labels = cp.asnumpy(labels)
        if isinstance(centroids, cp.ndarray): centroids = cp.asnumpy(centroids)
        return labels, centroids, inertia

    def _cluster_faiss(self, X, n_clusters, max_iter, random_state):
        X = np.ascontiguousarray(X.astype(np.float32))
        d = X.shape[1]
        kmeans = faiss.Kmeans(d, n_clusters, niter=max_iter, seed=random_state,
                              gpu=self.device_info.get('gpu_count', 0) > 0)
        kmeans.train(X)
        _, labels = kmeans.index.search(X, 1)
        labels = labels.ravel()
        distances, _ = kmeans.index.search(X, 1)
        inertia = float(np.sum(distances))
        return labels, kmeans.centroids, inertia

    def _cluster_torch(self, X, n_clusters, n_init, max_iter, random_state):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
        best_inertia = float('inf')
        best_labels = best_centroids = None
        torch.manual_seed(random_state)
        for _ in range(n_init):
            centroids = self._kmeans_plusplus_torch(X_torch, n_clusters)
            for _ in range(max_iter):
                distances = torch.cdist(X_torch, centroids)
                labels = torch.argmin(distances, dim=1)
                new_centroids = torch.zeros_like(centroids)
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.any(): new_centroids[k] = X_torch[mask].mean(dim=0)
                    else: new_centroids[k] = centroids[k]
                if torch.allclose(centroids, new_centroids, rtol=1e-4): break
                centroids = new_centroids
            distances = torch.cdist(X_torch, centroids)
            inertia = torch.sum(torch.min(distances, dim=1)[0] ** 2).item()
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.cpu().numpy()
                best_centroids = centroids.cpu().numpy()
        return best_labels, best_centroids, best_inertia

    def _kmeans_plusplus_torch(self, X, n_clusters):
        n_samples = X.shape[0]
        centroids = torch.empty((n_clusters, X.shape[1]), device=X.device)
        centroids[0] = X[torch.randint(n_samples, ())]
        for c in range(1, n_clusters):
            distances = torch.cdist(X, centroids[:c])
            min_distances = torch.min(distances, dim=1)[0]
            probs = min_distances ** 2
            probs = probs / probs.sum()
            idx = torch.searchsorted(torch.cumsum(probs, dim=0), torch.rand((), device=X.device)).item()
            centroids[c] = X[idx]
        return centroids

    def _cluster_cpu(self, X, n_clusters, n_init, max_iter, random_state):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                                 random_state=random_state, batch_size=min(8192, len(X)))
        labels = kmeans.fit_predict(X)
        return labels, kmeans.cluster_centers_, kmeans.inertia_

    def score(self, X, labels, method='davies_bouldin'):
        if self.backend == 'rapids' and method in ['davies_bouldin', 'silhouette']:
            return self._score_rapids(X, labels, method)
        else:
            return self._score_cpu(X, labels, method)

    def _score_rapids(self, X, labels, method):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        labels_gpu = cp.asarray(labels, dtype=cp.int32)
        if method == 'davies_bouldin':
            score = cu_davies_bouldin_score(X_gpu, labels_gpu)
        else:
            score = cu_silhouette_score(X_gpu, labels_gpu)
        return float(score) if isinstance(score, cp.ndarray) else score

    def _score_cpu(self, X, labels, method):
        if method == 'davies_bouldin':
            return davies_bouldin_score(X, labels)
        else:
            return silhouette_score(X, labels)


# ============================================================
# combine_ents_auto  — with incremental embedding update
# ============================================================

def combine_ents_auto(
    codebook_main: Dict[str, Any],
    min_exp_num: int = 2,
    max_exp_num: int = 20,
    use_thinking: bool = True,
    use_facts: bool = True,
    random_state: int = 0,
    sample_size_prop: float = 0.2,
    k_grid_size: int = 8,
    scoring: str = "silhouette",
    backend: str = 'auto',
    word_emb=None,
) -> Dict[str, Any]:

    # Fast exit: nothing changed since last combine
    dirty_set = codebook_main.get('_dirty_entities')
    if dirty_set is not None and len(dirty_set) == 0:
        logger.info("No dirty entities; skipping combine_ents_auto entirely.")
        return codebook_main

    E = list(codebook_main.get('e', []))
    R = list(codebook_main.get('r', []))

    # --- memory-safe matrix build ---
    gc.collect()
    emb_list = codebook_main.get('e_embeddings', [])
    n = len(emb_list)
    if n > 0:
        d = int(np.asarray(emb_list[0]).shape[0])
        X = np.empty((n, d), dtype=np.float32)
        for _i, _v in enumerate(emb_list):
            X[_i] = _v
        del emb_list
        gc.collect()
    else:
        X = np.empty((0, 0), dtype=np.float32)

    if n <= 2:
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [X[i].copy() for i in range(n)] if X.ndim == 2 and n > 0 else list(codebook_main.get('e_embeddings', []))
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main

    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    old_edge_emb = codebook_main.get('edge_matrix_embedding')
    old_E = list(E)
    old_R = list(R)

    clusterer = DeviceAwareClusterer(backend=backend)
    logger.info(f"Clustering {n} entities using {clusterer.backend} backend")

    # normalise in-place to save memory
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    X /= norms
    del norms
    X_norm = X  # alias (no copy)

    k_low  = max(2, int(np.ceil(n / max_exp_num)))
    k_high = max(2, min(n - 1, int(np.floor(n / min_exp_num))))
    if k_low > k_high:
        k_low, k_high = 2, max(2, min(n - 1, 5))

    sizes   = np.geomspace(max_exp_num, min_exp_num, num=k_grid_size)
    cand_ks = sorted(set(int(np.clip(int(np.ceil(n / s)), k_low, k_high)) for s in sizes))

    rng = np.random.default_rng(random_state)
    max_k = max(cand_ks)
    eff_sample = min(n, max(int(sample_size_prop * n), int(1.2 * (max_k + 1))))
    if n <= eff_sample:
        X_sample_norm = X_norm
    else:
        idx_sample = rng.choice(n, size=eff_sample, replace=False)
        X_sample_norm = X_norm[idx_sample]

    best_k, best_score = None, -np.inf
    score_method = 'davies_bouldin' if scoring == 'db' else 'silhouette'
    from tqdm import tqdm as _tqdm
    for k in _tqdm(cand_ks, desc="[combine] k-grid search", leave=False):
        labels, _, _ = clusterer.cluster(X_sample_norm, k, n_init=3, max_iter=100, random_state=random_state)
        try:
            score = clusterer.score(X_sample_norm, labels, method=score_method)
            if score_method == 'davies_bouldin':
                score = -score
        except Exception as e:
            logger.debug(f"Score failed for k={k}: {e}")
            continue
        if score > best_score:
            best_score, best_k = score, k

    logger.info(f"Selected k={best_k} with score={best_score:.4f}")

    print(f"[combine] Running final clustering (k={best_k}, n={n}) …")
    labels_full, centroids, _ = clusterer.cluster(
        X_norm, n_clusters=int(best_k), n_init=5, max_iter=200, random_state=random_state)
    print(f"[combine] Clustering done.")

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

    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

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

    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]

    _update_all_cached_embeddings(
        codebook_main, old_edges, old_edge_emb, old_E, old_R,
        old_edge_to_new_edge, word_emb,
        use_thinking=use_thinking, use_facts=use_facts,
    )

    # Store remap for fast TorchDBCache update (avoid full rebuild)
    codebook_main['_combine_ents_remap'] = {
        'old_edge_to_new': old_edge_to_new_edge,
        'old_ent_to_new': old_ent_to_new,
        'n_new_edges': len(new_edges),
        'n_new_ents': len(new_e),
    }

    return codebook_main


# ============================================================
# ANN / Union-Find helpers
# ============================================================

class ANNBackend:
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    PYNNDESCENT = "pynndescent"
    ANNOY = "annoy"
    SKLEARN = "sklearn"


@dataclass
class ClusteringConfig:
    k_neighbors: int = 10
    similarity_threshold: float = 0.8
    min_cluster_size: int = 2
    ann_backend: str = "auto"
    metric: str = "cosine"
    representative_method: str = "medoid"
    use_gpu: bool = True
    n_trees: int = 50
    ef_construction: int = 200
    ef_search: int = 100


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True

    def get_clusters(self) -> Dict[int, List[int]]:
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            clusters.setdefault(root, []).append(i)
        return clusters


class RepresentativeSelector:
    @staticmethod
    def select_medoid(X, cluster_indices, metric="euclidean"):
        if len(cluster_indices) == 1: return cluster_indices[0]
        from sklearn.metrics.pairwise import pairwise_distances
        dists = pairwise_distances(X[cluster_indices],
                                   metric=("cosine" if metric == "cosine" else "euclidean"))
        return cluster_indices[np.argmin(dists.sum(axis=1))]

    @staticmethod
    def select_density_peak(X, cluster_indices, k=5, metric="euclidean"):
        if len(cluster_indices) == 1: return cluster_indices[0]
        Xc = X[cluster_indices]
        k = min(k, len(cluster_indices) - 1)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1,
                              metric=("cosine" if metric == "cosine" else "euclidean"))
        nn.fit(Xc)
        dists, _ = nn.kneighbors(Xc)
        densities = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-10)
        return cluster_indices[int(np.argmax(densities))]


class ANNGraphBuilder:
    def __init__(self, config):
        self.config = config
        self.backend = self._select_backend()

    def _select_backend(self):
        backend_map = {
            "faiss": ANNBackend.FAISS, "hnswlib": ANNBackend.HNSWLIB,
            "pynndescent": ANNBackend.PYNNDESCENT, "annoy": ANNBackend.ANNOY,
            "sklearn": ANNBackend.SKLEARN
        }
        if self.config.ann_backend != "auto":
            return backend_map.get(self.config.ann_backend, ANNBackend.SKLEARN)
        if FAISS_AVAILABLE: return ANNBackend.FAISS
        else: return ANNBackend.SKLEARN

    def build_graph(self, X_unit):
        n = X_unit.shape[0]
        k = min(self.config.k_neighbors, max(1, n - 1))

        if self.backend == ANNBackend.FAISS:
            try:
                if self.config.metric == "cosine":
                    idx, sims = self._build_faiss_cosine(X_unit, k)
                else:
                    idx, sims = self._build_faiss_l2(X_unit, k)
            except (MemoryError, RuntimeError) as e:
                logger.warning(f"FAISS OOM ({e}), falling back to batched torch GPU k-NN")
                try:
                    if TORCH_AVAILABLE and self.config.metric == "cosine":
                        idx, sims = self._build_torch_cosine(X_unit, k)
                    else:
                        raise RuntimeError("torch not available or metric != cosine")
                except (MemoryError, RuntimeError) as e2:
                    logger.warning(f"Torch fallback failed ({e2}), falling back to sklearn CPU")
                    idx, sims = self._build_sklearn(X_unit, k)
        elif self.backend == ANNBackend.SKLEARN:
            idx, sims = self._build_sklearn(X_unit, k)
        else:
            idx, sims = self._build_sklearn(X_unit, k)

        # Clean up: remove self-references, pad if needed
        clean_idx = np.empty((n, k), dtype=np.int32)
        clean_sims = np.empty((n, k), dtype=np.float32)
        for i in range(n):
            row_idx = np.asarray(idx[i], dtype=np.int32)
            row_sims = np.asarray(sims[i], dtype=np.float32)
            keep = row_idx != i
            row_idx, row_sims = row_idx[keep], row_sims[keep]
            if row_idx.shape[0] < k:
                missing = k - row_idx.shape[0]
                pad_idx = np.full(missing, row_idx[-1] if row_idx.shape[0] > 0 else i, dtype=np.int32)
                pad_sims = np.full(missing, row_sims[-1] if row_sims.shape[0] > 0 else -1.0, dtype=np.float32)
                row_idx = np.concatenate([row_idx, pad_idx])
                row_sims = np.concatenate([row_sims, pad_sims])
            clean_idx[i] = row_idx[:k]
            clean_sims[i] = row_sims[:k]
        return clean_idx, clean_sims

    # ----------------------------------------------------------
    # BATCHED torch cosine k-NN — never loads full DB on GPU
    # ----------------------------------------------------------
    def _build_torch_cosine(self, X_unit, k):
        """GPU-based k-NN — batches both queries and database to avoid OOM."""
        dev = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        X = torch.from_numpy(X_unit.astype(np.float32))
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        n, d = X.shape

        all_idx = np.empty((n, k), dtype=np.int64)
        all_sims = np.empty((n, k), dtype=np.float32)

        # Budget ~256MB per shard on GPU
        db_shard_size = max(1, int(256e6 / (d * 4)))
        q_batch_size = max(1, min(2048, int(256e6 / (d * 4))))

        logger.info(f"torch k-NN: n={n}, k={k}, q_batch={q_batch_size}, "
                     f"db_shard={db_shard_size}, device={dev}")

        for q_start in range(0, n, q_batch_size):
            q_end = min(q_start + q_batch_size, n)
            q = X[q_start:q_end].to(dev)
            bsz = q.shape[0]

            topk_sims = torch.full((bsz, k), -float('inf'), device=dev)
            topk_idx = torch.zeros((bsz, k), dtype=torch.long, device=dev)

            for db_start in range(0, n, db_shard_size):
                db_end = min(db_start + db_shard_size, n)
                db = X[db_start:db_end].to(dev)
                sim_block = q @ db.T  # (bsz, shard_size)

                # Mask self-similarity
                if q_start < db_end and q_end > db_start:
                    overlap_q_lo = max(q_start, db_start) - q_start
                    overlap_q_hi = min(q_end, db_end) - q_start
                    for i in range(overlap_q_lo, overlap_q_hi):
                        sim_block[i, (q_start + i) - db_start] = -float('inf')

                # Merge running top-k with this shard
                shard_k = min(k, sim_block.shape[1])
                shard_sims, shard_pos = sim_block.topk(shard_k, dim=1)
                shard_idx = shard_pos + db_start  # global indices

                combined_sims = torch.cat([topk_sims, shard_sims], dim=1)
                combined_idx = torch.cat([topk_idx, shard_idx], dim=1)
                top_vals, top_pos = combined_sims.topk(k, dim=1)
                topk_sims = top_vals
                topk_idx = combined_idx.gather(1, top_pos)

                del db, sim_block, shard_sims, shard_pos, shard_idx

            all_idx[q_start:q_end] = topk_idx.cpu().numpy()
            all_sims[q_start:q_end] = topk_sims.cpu().numpy()
            del q, topk_sims, topk_idx
            if dev == "cuda":
                torch.cuda.empty_cache()

        return all_idx, all_sims

    def _build_faiss_cosine(self, X_unit, k):
        Xf = np.ascontiguousarray(X_unit.astype(np.float32))
        d = Xf.shape[1]
        index = faiss.IndexFlatIP(d)
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(Xf)
        sims, idx = index.search(Xf, k + 1)
        return idx[:, 1:], sims[:, 1:]

    def _build_faiss_l2(self, X, k):
        Xf = np.ascontiguousarray(X.astype(np.float32))
        d = Xf.shape[1]
        index = faiss.IndexFlatL2(d)
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(Xf)
        dists, idx = index.search(Xf, k + 1)
        d2 = dists[:, 1:]
        sigma = float(np.median(d2))
        sigma = sigma if sigma > 1e-12 else 1.0
        sims = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        return idx[:, 1:], sims

    def _build_sklearn(self, X_unit, k):
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
            sigma = float(np.median(d2))
            sigma = sigma if sigma > 1e-12 else 1.0
            sims = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
            return idx[:, 1:], sims


# ============================================================
# combine_ents_ann_knn — with incremental embedding update
# ============================================================

def combine_ents_ann_knn(
    codebook_main: Dict[str, Any],
    config: Optional[ClusteringConfig] = None,
    use_thinking: bool = True,
    use_facts: bool = True,
    sim_threshold: float = 0.9,
    word_emb=None,
) -> Dict[str, Any]:

    if config is None:
        n0 = len(codebook_main.get('e', []))
        k = min(50, max(5, int(np.sqrt(max(1, n0)))))
        config = ClusteringConfig(k_neighbors=k, similarity_threshold=sim_threshold, min_cluster_size=2)

    # --- early exit: nothing changed since last combine ---
    dirty_set: Optional[Set[int]] = codebook_main.get('_dirty_entities')
    if dirty_set is not None and len(dirty_set) == 0:
        logger.info("No dirty entities; skipping combine_ents_ann_knn entirely.")
        return codebook_main

    E_old = list(codebook_main.get('e', []))
    R_old = list(codebook_main.get('r', []))

    # --- memory-safe matrix build: gc first, pre-allocate, fill row-wise ---
    gc.collect()
    emb_list = codebook_main.get('e_embeddings', [])
    n = len(emb_list)
    if n > 0:
        d = int(np.asarray(emb_list[0]).shape[0])
        X = np.empty((n, d), dtype=np.float32)
        for _i, _v in enumerate(emb_list):
            X[_i] = _v
        del emb_list
        gc.collect()
    else:
        d = 1
        X = np.empty((0, 0), dtype=np.float32)

    if n <= 2:
        codebook_main['e'] = list(E_old)
        codebook_main['e_embeddings'] = [X[i].copy() for i in range(n)] if X.ndim == 2 and n > 0 else list(codebook_main.get('e_embeddings', []))
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main

    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    old_edge_emb = codebook_main.get('edge_matrix_embedding')
    old_E = list(E_old)
    old_R = list(R_old)

    metric = config.metric
    if metric == "cosine" and d == 1:
        logger.info("Detected d=1 with cosine metric; switching to 'euclidean'")
        metric = "euclidean"

    # normalise in-place to avoid a second 123 MiB allocation
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        X /= norms          # in-place
        del norms
    X_unit = X              # alias (no copy)

    # ---------- decide incremental vs full ----------
    # Round 0 (first call) always does full; subsequent calls do incremental
    _round = codebook_main.get('_combine_round', 0)
    k_cfg = config.k_neighbors
    thr = float(config.similarity_threshold)

    use_incremental = (
        _round > 0                              # not the first call
        and dirty_set is not None
        and 0 < len(dirty_set) < n * 0.5
        and n > 50
    )

    uf = UnionFind(n)
    edges_created = 0

    if use_incremental:
        dirty_arr = sorted(dirty_set)
        n_dirty = len(dirty_arr)
        k_search = min(k_cfg, n - 1)
        logger.info(f"Incremental mode (round {_round}): searching {n_dirty}/{n} dirty entities (k={k_search})")

        X_query = np.ascontiguousarray(X_unit[dirty_arr], dtype=np.float32)

        if metric == "cosine":
            if FAISS_AVAILABLE:
                Xf = np.ascontiguousarray(X_unit, dtype=np.float32)
                index = faiss.IndexFlatIP(d)
                index.add(Xf)
                raw_sims, raw_idx = index.search(X_query, k_search + 1)
                del Xf
            else:
                from sklearn.neighbors import NearestNeighbors as _NN
                _nn = _NN(n_neighbors=k_search + 1, metric="cosine", algorithm="brute")
                _nn.fit(X_unit)
                _dists, raw_idx = _nn.kneighbors(X_query)
                raw_sims = (1.0 - _dists).astype(np.float32)
        else:  # euclidean
            if FAISS_AVAILABLE:
                Xf = np.ascontiguousarray(X_unit, dtype=np.float32)
                index = faiss.IndexFlatL2(d)
                index.add(Xf)
                raw_d2, raw_idx = index.search(X_query, k_search + 1)
                del Xf
                sigma = float(np.median(raw_d2[:, 1:]))
                sigma = sigma if sigma > 1e-12 else 1.0
                raw_sims = np.exp(-raw_d2 / (2 * sigma ** 2)).astype(np.float32)
            else:
                from sklearn.neighbors import NearestNeighbors as _NN
                _nn = _NN(n_neighbors=k_search + 1, metric="euclidean", algorithm="auto")
                _nn.fit(X_unit)
                _dists, raw_idx = _nn.kneighbors(X_query)
                sigma = float(np.median(_dists[:, 1:]))
                sigma = sigma if sigma > 1e-12 else 1.0
                raw_sims = np.exp(-_dists ** 2 / (2 * sigma ** 2)).astype(np.float32)

        for qi in range(n_dirty):
            gi = dirty_arr[qi]
            for col in range(raw_idx.shape[1]):
                j = int(raw_idx[qi, col])
                if j == gi:
                    continue
                s = float(raw_sims[qi, col])
                if s >= thr and uf.union(gi, j):
                    edges_created += 1

        del X_query, raw_idx, raw_sims

    else:
        # ---------- FULL path ----------
        logger.info(f"Full mode (round {_round}): searching all {n} entities")
        gb = ANNGraphBuilder(replace(config, metric=metric))
        indices, sims = gb.build_graph(X_unit)

        for i in range(n):
            for j, s in zip(indices[i], sims[i]):
                if float(s) >= thr and uf.union(i, int(j)):
                    edges_created += 1
        del indices, sims

    logger.info(f"Created {edges_created} union edges at threshold {config.similarity_threshold}")

    if edges_created == 0:
        logger.info("No merges formed; returning original codebook unchanged.")
        return codebook_main

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
                rep = rep_selector.select_density_peak(X_unit, members, k=min(5, len(members) - 1), metric=metric)
            kept_indices.append(rep)
            for m in members:
                representatives[m] = rep

    kept_indices = sorted(set(kept_indices))
    logger.info(f"Reduced from {n} to {len(kept_indices)} entities (clusters={len(clusters)})")

    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    old_ent_to_new = {i: rep_to_new[representatives[i]] for i in range(n)}

    new_e = [E_old[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    tuple_to_new_edge_idx: Dict[Tuple[int, int, int], int] = {}
    new_edges: List[List[int]] = []
    old_edge_to_new_edge: Dict[int, int] = {}

    for old_idx, (e1, r, e2) in enumerate(old_edges):
        ne1 = old_ent_to_new.get(int(e1), int(e1))
        ne2 = old_ent_to_new.get(int(e2), int(e2))
        tup = (int(ne1), int(r), int(ne2))
        if tup not in tuple_to_new_edge_idx:
            tuple_to_new_edge_idx[tup] = len(new_edges)
            new_edges.append([int(ne1), int(r), int(ne2)])
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

    _update_all_cached_embeddings(
        codebook_main, old_edges, old_edge_emb, old_E, old_R,
        old_edge_to_new_edge, word_emb,
        use_thinking=use_thinking, use_facts=use_facts,
    )

    # Store remap for fast TorchDBCache update (avoid full rebuild)
    codebook_main['_combine_ents_remap'] = {
        'old_edge_to_new': old_edge_to_new_edge,
        'old_ent_to_new': old_ent_to_new,
        'n_new_edges': len(new_edges),
        'n_new_ents': len(new_e),
    }

    return codebook_main


# ============================================================
# coarse_combine
# ============================================================

def coarse_combine(
    codebook_main: Dict[str, Any],
    min_exp_num: int = 2,
    max_exp_num: int = 20,
    use_thinking: bool = True,
    use_facts: bool = True,
    random_state: int = 0,
    sample_size_prop: float = 0.2,
    k_grid_size: int = 8,
    scoring: str = "silhouette",
    backend: str = 'auto',
    config: Optional[ClusteringConfig] = None,
    ram_threshold: float = 70.0,
    sim_threshold: float = 0.9,
    word_emb=None,
):
    filtered_codebook_main = combine_ents_ann_knn(
        codebook_main, config, use_thinking, use_facts, sim_threshold,
        word_emb=word_emb,
    )

    ram_used_percent = psutil.virtual_memory().percent
    print(f"[INFO] Current RAM usage: {ram_used_percent:.2f}%")

    if ram_used_percent < ram_threshold:
        final_codebook_main = combine_ents_auto(
            filtered_codebook_main, min_exp_num, max_exp_num,
            use_thinking, use_facts, random_state, sample_size_prop,
            k_grid_size, scoring, backend,
            word_emb=word_emb,
        )
    else:
        print(f"[WARN] RAM usage {ram_used_percent:.2f}% exceeds threshold {ram_threshold}%. "
              f"Skipping aggressive combine.")
        final_codebook_main = filtered_codebook_main

    return final_codebook_main

# python py_files/combine_ent_cached_aligned.py
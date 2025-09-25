import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, replace
from enum import Enum

# ann_merge_questions_answer_gated.py
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, replace
import logging
import numpy as np

# ===================== Logging =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== Optional ANN backends =====================
FAISS_AVAILABLE = False
HNSWLIB_AVAILABLE = False
PYNNDESCENT_AVAILABLE = False
ANNOY_AVAILABLE = False

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
    logger.info("FAISS available")
except Exception:
    pass

try:
    import hnswlib  # type: ignore
    HNSWLIB_AVAILABLE = True
    logger.info("HNSWlib available")
except Exception:
    pass

try:
    import pynndescent  # type: ignore
    PYNNDESCENT_AVAILABLE = True
    logger.info("PyNNDescent available")
except Exception:
    pass

try:
    import annoy  # type: ignore
    ANNOY_AVAILABLE = True
    logger.info("Annoy available")
except Exception:
    pass

from sklearn.neighbors import NearestNeighbors  # fallback

# ===================== Your decoders (unchanged) =====================
def decode_question(question, codebook_main, fmt='words'):
    """
    question: list[int] of edge indices
    fmt: 'words' | 'embeddings' | 'edges'
    """
    edges = codebook_main["edge_matrix"]
    idxs = list(question)

    def get_edge(i):
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
        decoded = [[h, r, t] for (h, r, t) in (get_edge(i) for i in idxs)]
    else:
        raise ValueError("fmt must be 'words', 'embeddings' or 'edges'.")
    return decoded

def decode_questions(questions, questions_source_codebook, fmt='words'):
    return [decode_question(q, questions_source_codebook, fmt=fmt) for q in questions]

# ===================== Configs & Structures =====================
class ANNBackend:
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    PYNNDESCENT = "pynndescent"
    ANNOY = "annoy"
    SKLEARN = "sklearn"

@dataclass
class ANNConfig:
    k_neighbors: int = 10
    metric: str = "cosine"             # we’ll use cosine
    representative_method: str = "medoid"
    use_gpu: bool = True
    n_trees: int = 50                  # Annoy
    ef_construction: int = 200         # HNSW
    ef_search: int = 100               # HNSW

@dataclass
class QMergeGate:
    q_sim_threshold: float = 0.85
    a_sim_threshold: float = 0.80
    combine_strategy: str = "representative"  # or "union"

# ===================== Union-Find =====================
class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True

# ===================== ANN Graph Builder (cosine sims) =====================
class ANNGraphBuilder:
    def __init__(self, config: ANNConfig):
        self.config = config
        self.backend = self._select_backend()
        logger.info(f"Using ANN backend: {self.backend}")

    def _select_backend(self) -> str:
        if FAISS_AVAILABLE:      return ANNBackend.FAISS
        if HNSWLIB_AVAILABLE:    return ANNBackend.HNSWLIB
        if PYNNDESCENT_AVAILABLE:return ANNBackend.PYNNDESCENT
        if ANNOY_AVAILABLE:      return ANNBackend.ANNOY
        return ANNBackend.SKLEARN

    def build_graph(self, X_unit: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, similarities) with cosine similarities in [−1,1] (we’ll map to [0,1] later)."""
        n, d = X_unit.shape
        k = min(k, max(1, n-1))
        if self.backend == ANNBackend.FAISS:
            return self._faiss_ip(X_unit, k)
        elif self.backend == ANNBackend.HNSWLIB:
            return self._hnsw_cos(X_unit, k)
        elif self.backend == ANNBackend.PYNNDESCENT:
            return self._pynnd_cos(X_unit, k)
        elif self.backend == ANNBackend.ANNOY:
            return self._annoy_cos(X_unit, k)
        else:
            return self._sklearn_cos(X_unit, k)

    def _faiss_ip(self, X_unit: np.ndarray, k: int):
        import faiss
        Xf = np.ascontiguousarray(X_unit.astype(np.float32))
        index = faiss.IndexFlatIP(Xf.shape[1])
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(Xf)
        sims, idx = index.search(Xf, k+1)  # inner products = cosine because unit
        return idx[:, 1:], sims[:, 1:]

    def _hnsw_cos(self, X_unit: np.ndarray, k: int):
        import hnswlib
        n, d = X_unit.shape
        index = hnswlib.Index(space="cosine", dim=d)
        index.init_index(max_elements=n, ef_construction=self.config.ef_construction, M=16)
        index.add_items(X_unit, np.arange(n))
        index.set_ef(self.config.ef_search)
        idx, dists = index.knn_query(X_unit, k=k+1)
        sims = 1.0 - dists[:, 1:]
        return idx[:, 1:], sims

    def _pynnd_cos(self, X_unit: np.ndarray, k: int):
        import pynndescent
        index = pynndescent.NNDescent(X_unit, n_neighbors=k+1, metric="cosine", random_state=42)
        idx, dists = index.neighbor_graph
        sims = 1.0 - dists[:, 1:]
        return idx[:, 1:], sims

    def _annoy_cos(self, X_unit: np.ndarray, k: int):
        from annoy import AnnoyIndex
        n, d = X_unit.shape
        index = AnnoyIndex(d, "angular")
        for i in range(n):
            index.add_item(i, X_unit[i])
        index.build(self.config.n_trees)
        idx = np.empty((n, k+1), dtype=np.int32)
        sims = np.empty((n, k+1), dtype=np.float32)
        for i in range(n):
            nn_idx, _ = index.get_nns_by_item(i, k+1, include_distances=True)
            nn_idx = np.array(nn_idx, dtype=np.int32)
            sim_row = (X_unit[i] @ X_unit[nn_idx].T).astype(np.float32)
            idx[i] = nn_idx
            sims[i] = sim_row
        return idx[:, 1:], sims[:, 1:]

    def _sklearn_cos(self, X_unit: np.ndarray, k: int):
        nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute")
        nn.fit(X_unit)
        dists, idx = nn.kneighbors(X_unit)
        sims = (1.0 - dists[:, 1:]).astype(np.float32)
        return idx[:, 1:], sims

# ===================== Embedding builders (from decoded edges) =====================
def _flatten_edge_triplet_to_vec(edge_triplet) -> np.ndarray:
    """edge_triplet: [e_vec, r_vec, e_vec] → concat vector"""
    e1, r, e2 = edge_triplet
    return np.concatenate([np.asarray(e1, np.float32),
                           np.asarray(r,  np.float32),
                           np.asarray(e2, np.float32)], axis=0)

def _embed_small_item(edge_indices: List[int], codebook_main: Dict[str, Any]) -> Optional[np.ndarray]:
    """One small question/answer (list[int] edges) → average of concatenated edge vectors."""
    decoded = decode_question(edge_indices, codebook_main, fmt='embeddings')
    if not decoded:
        return None
    mats = [_flatten_edge_triplet_to_vec(t) for t in decoded]
    return np.mean(np.stack(mats, axis=0), axis=0)

def _embed_nested(items_nested: List[List[int]], codebook_main: Dict[str, Any]) -> Optional[np.ndarray]:
    """Question/Answer as list of small items → average of their vectors."""
    vecs = []
    for sub in (items_nested or []):
        v = _embed_small_item(sub, codebook_main)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))

# ===================== Representative selection (medoid/density) =====================
from sklearn.neighbors import NearestNeighbors
def _select_rep_medoid(X_unit: np.ndarray, members: List[int]) -> int:
    if len(members) == 1:
        return members[0]
    sub = X_unit[members]
    sims = sub @ sub.T
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    cos = sims / ((norms @ norms.T) + 1e-12)
    dmat = 1.0 - cos
    return members[int(np.argmin(dmat.sum(axis=1)))]

def _select_rep_density(X_unit: np.ndarray, members: List[int], k: int = 5) -> int:
    if len(members) == 1:
        return members[0]
    sub = X_unit[members]
    k = min(k, len(members)-1)
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute")
    nn.fit(sub)
    dists, _ = nn.kneighbors(sub)
    densities = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-10)
    return members[int(np.argmax(densities))]

# ===================== MAIN: ANN + answer-gated merging =====================
@dataclass
class QAnnMergeConfig:
    ann: ANNConfig = ANNConfig()
    gate: QMergeGate = QMergeGate()

def ann_merge_questions_answer_gated(
    codebook_main: Dict[str, Any],
    questions_lst: List[List[List[int]]],
    answers_lst:   List[List[List[int]]],
    cfg: Optional[QAnnMergeConfig] = None,
) -> Tuple[List[List[List[int]]], List[List[List[int]]], List[int], List[List[int]], List[int]]:
    """
    Returns:
      new_questions_lst, new_answers_lst, q_old_to_new, q_clusters, kept_indices
    Note: codebook_main is NOT modified.
    """
    assert len(questions_lst) == len(answers_lst), "questions_lst and answers_lst must align in length"
    n = len(questions_lst)
    if cfg is None:
        k = min(50, max(5, int(np.sqrt(max(1, n)))))
        cfg = QAnnMergeConfig(ann=ANNConfig(k_neighbors=k), gate=QMergeGate())

    # 1) Build question & answer vectors from decoded embeddings
    Q_vecs, A_vecs = [], []
    for i in range(n):
        qv = _embed_nested(questions_lst[i], codebook_main)
        av = _embed_nested(answers_lst[i],   codebook_main)
        if qv is None or av is None:
            # robust fallback (keeps them from merging due to low sims)
            if qv is None and A_vecs:  # reuse known dimension
                qv = np.zeros_like(A_vecs[0])
            if av is None and Q_vecs:
                av = np.zeros_like(Q_vecs[0])
        Q_vecs.append(qv.astype(np.float32))
        A_vecs.append(av.astype(np.float32))
    Q = np.stack(Q_vecs, axis=0)  # (n, d)
    A = np.stack(A_vecs, axis=0)

    # Normalize Q for cosine ANN
    Q_unit = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    # 2) ANN kNN graph on questions
    gb = ANNGraphBuilder(cfg.ann)
    idx, sims_cos = gb.build_graph(Q_unit, k=cfg.ann.k_neighbors)  # cosine sims in [-1,1] (or [0,1] depending backend)
    # Clamp to [-1,1] then map to [0,1] for thresholding
    q_sims01 = np.clip(sims_cos, -1.0, 1.0) * 0.5 + 0.5

    # 3) Union-Find with double gate (question & answer sim)
    uf = UnionFind(n)
    for i in range(n):
        for j, qsim01 in zip(idx[i], q_sims01[i]):
            j = int(j)
            if j == i:
                continue
            if qsim01 >= cfg.gate.q_sim_threshold:
                asim = _cos_sim(A[i], A[j])  # in [-1,1]
                asim01 = 0.5 * (asim + 1.0)
                if asim01 >= cfg.gate.a_sim_threshold:
                    uf.union(i, j)

    # 4) Build clusters
    roots: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        roots.setdefault(r, []).append(i)
    clusters = list(roots.values())

    # 5) Representatives
    kept_indices: List[int] = []
    rep_of: Dict[int, int] = {}
    for members in clusters:
        if cfg.ann.representative_method == "density":
            rep = _select_rep_density(Q_unit, members, k=min(5, len(members)-1))
        else:  # medoid
            rep = _select_rep_medoid(Q_unit, members)
        kept_indices.append(rep)
        for m in members:
            rep_of[m] = rep
    kept_indices = sorted(set(kept_indices))
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    q_old_to_new = [rep_to_new[rep_of[i]] for i in range(n)]
    q_clusters = [sorted([i for i in range(n) if rep_of[i] == rep]) for rep in kept_indices]

    # 6) Rebuild outputs (no codebook_main mutation)
    def members_of(rep_old: int) -> List[int]:
        return [i for i in range(n) if rep_of[i] == rep_old]

    if cfg.gate.combine_strategy == "representative":
        new_questions_lst = [questions_lst[i] for i in kept_indices]
        new_answers_lst   = [answers_lst[i]   for i in kept_indices]
    else:  # "union"
        new_questions_lst, new_answers_lst = [], []
        for rep in kept_indices:
            mids = members_of(rep)
            qset: Set[Tuple[int, ...]] = set()
            aset: Set[Tuple[int, ...]] = set()
            for i in mids:
                for subq in (questions_lst[i] or []):
                    qset.add(tuple(int(v) for v in subq))
                for suba in (answers_lst[i] or []):
                    aset.add(tuple(int(v) for v in suba))
            new_questions_lst.append([list(t) for t in sorted(qset)])
            new_answers_lst.append([list(t) for t in sorted(aset)])

    return new_questions_lst, new_answers_lst, q_old_to_new, q_clusters, kept_indices

# ===== The facts-only ANN merger =====
@dataclass
class FactsAnnConfig:
    ann: ANNConfig = ANNConfig()
    sim_threshold: float = 0.85          # threshold on cosine(sim) in [0,1]
    combine_strategy: str = "representative"  # or "union"

# ===== Representative selection =====
def _select_rep_medoid_cos(X_unit: np.ndarray, members: List[int]) -> int:
    if len(members) == 1:
        return members[0]
    sub = X_unit[members]
    sims = sub @ sub.T
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    cos = sims / ((norms @ norms.T) + 1e-12)
    dmat = 1.0 - cos
    return members[int(np.argmin(dmat.sum(axis=1)))]

def _select_rep_density_cos(X_unit: np.ndarray, members: List[int], k: int = 5) -> int:
    if len(members) == 1:
        return members[0]
    sub = X_unit[members]
    k = min(k, len(members)-1)
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute")
    nn.fit(sub)
    dists, _ = nn.kneighbors(sub)
    densities = 1.0 / (dists[:,1:].mean(axis=1) + 1e-10)
    return members[int(np.argmax(densities))]

def ann_feat_combine(
    codebook_main: Dict[str, Any],
    feat_lst: List[List[List[int]]],
    cfg: Optional[FactsAnnConfig] = None,
) -> Tuple[List[List[List[int]]], List[int], List[List[int]], List[int]]:
    """
    Merge near-duplicate fact bundles using ANN over embeddings decoded from edge ids.

    Inputs:
      - codebook_main: read-only; NOT modified
      - feat_lst: list of facts items; each item is a list of "small facts"; each small fact is a list[int] edge ids

    Returns:
      - new_feat_lst           (merged, order = kept representatives)
      - facts_old_to_new        (list[int], mapping each original index -> new index)
      - facts_clusters          (list[list[int]], original indices per kept representative)
      - kept_indices            (list[int], original indices that were kept)
    """
    n = len(feat_lst)
    if n == 0:
        return [], [], [], []

    if cfg is None:
        k = min(50, max(5, int(np.sqrt(max(1, n)))))
        cfg = FactsAnnConfig(ann=ANNConfig(k_neighbors=k))

    # 1) Build vectors for each facts item
    F_vecs = []
    for i in range(n):
        fv = _embed_nested(feat_lst[i], codebook_main)
        if fv is None:
            # Put a zero vec of the same size as first available
            if F_vecs:
                fv = np.zeros_like(F_vecs[0])
            else:
                # Try to synthesize a dimension by decoding one subfact from i (if exists)
                probe = None
                for sub in (feat_lst[i] or []):
                    probe = _embed_small_item(sub, codebook_main)
                    if probe is not None:
                        break
                fv = probe if probe is not None else np.zeros(32, dtype=np.float32)
        F_vecs.append(fv.astype(np.float32))
    F = np.stack(F_vecs, axis=0)

    # 2) Normalize & ANN kNN
    F_unit = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    gb = ANNGraphBuilder(cfg.ann)
    idx, sims = gb.build_graph(F_unit, k=cfg.ann.k_neighbors)
    # sims are cosine IPs; map to [0,1] if needed (ensure bounds)
    sims01 = np.clip(sims, -1.0, 1.0) * 0.5 + 0.5

    # 3) Union-Find on similarity threshold
    uf = UnionFind(n)
    thr = float(cfg.sim_threshold)
    for i in range(n):
        for j, s01 in zip(idx[i], sims01[i]):
            j = int(j)
            if j == i: 
                continue
            if s01 >= thr:
                uf.union(i, j)

    # 4) Clusters
    roots: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        roots.setdefault(r, []).append(i)
    clusters = list(roots.values())

    # 5) Representatives
    kept_indices: List[int] = []
    rep_of: Dict[int, int] = {}
    for members in clusters:
        if cfg.ann.representative_method == "density":
            rep = _select_rep_density_cos(F_unit, members, k=min(5, len(members)-1))
        else:
            rep = _select_rep_medoid_cos(F_unit, members)
        kept_indices.append(rep)
        for m in members:
            rep_of[m] = rep
    kept_indices = sorted(set(kept_indices))
    rep_to_new = {old: new for new, old in enumerate(kept_indices)}
    facts_old_to_new = [rep_to_new[rep_of[i]] for i in range(n)]
    facts_clusters = [sorted([i for i in range(n) if rep_of[i] == rep]) for rep in kept_indices]

    # 6) Rebuild facts list (no mutation to codebook_main)
    def members_of(rep_old: int) -> List[int]:
        return [i for i in range(n) if rep_of[i] == rep_old]

    if cfg.combine_strategy == "representative":
        new_feat_lst = [feat_lst[i] for i in kept_indices]
    else:
        # union of subfacts (dedup as tuples)
        new_feat_lst = []
        for rep in kept_indices:
            mids = members_of(rep)
            aset: Set[Tuple[int, ...]] = set()
            for i in mids:
                for sub in (feat_lst[i] or []):
                    aset.add(tuple(int(v) for v in sub))
            new_feat_lst.append([list(t) for t in sorted(aset)])

    return new_feat_lst, facts_old_to_new, facts_clusters, kept_indices

# # ===================== Minimal Example =====================
# if __name__ == "__main__":
#     rng = np.random.default_rng(42)

#     # Tiny codebook with embeddings for entities/relations
#     E = ["fair skin", "BCC", "UV", "melanin", "dermis"]
#     R = ["has effect", "causes", "modulates"]
#     e_dim, r_dim = 8, 6
#     e_embeddings = rng.normal(size=(len(E), e_dim)).astype(np.float32).tolist()
#     r_embeddings = rng.normal(size=(len(R), r_dim)).astype(np.float32).tolist()

#     # Edges [h, r, t] with small ids
#     edge_matrix = [
#         [0, 1, 1],  # 0  fair skin causes BCC
#         [2, 1, 1],  # 1  UV causes BCC
#         [3, 2, 2],  # 2  melanin modulates UV
#         [1, 0, 4],  # 3  BCC has effect dermis
#         [0, 0, 3],  # 4  fair skin has effect melanin
#         [2, 0, 0],  # 5  UV has effect fair skin
#         [3, 0, 0],  # 6  melanin has effect fair skin
#         [2, 1, 0],  # 7  UV causes fair skin (nonsense, demo)
#         [4, 2, 1],  # 8  dermis modulates BCC
#     ]
#     codebook_main = {
#         "e": E, "r": R,
#         "edge_matrix": edge_matrix,
#         "e_embeddings": e_embeddings,
#         "r_embeddings": r_embeddings,
#     }

#     # Questions: list of small-questions (each is list[int] of edge ids)
#     questions_lst = [
#         [[0]],                     # q0
#         [[0]],                     # q1  ~ dup of q0
#         [[1, 2], [2, 1]],          # q2
#         [[1, 2], [2, 1]],          # q3  ~ dup of q2
#         [[4]],                     # q4 unique
#         [[7]],                     # q5 unique
#     ]

#     # Answers: same sub-count per question, but different edge ids
#     answers_lst = [
#         [[1]],                     # a0 matches a1
#         [[1]],                     # a1 matches a0
#         [[8, 3], [3, 8]],          # a2 matches a3
#         [[8, 3], [3, 8],[0]],          # a3 matches a2
#         [[6]],                     # a4
#         [[5]],                     # a5
#     ]

#     print("=== BEFORE ===")
#     print("Q count:", len(questions_lst))
#     print("questions_lst:", questions_lst)
#     print("answers_lst:",   answers_lst)

#     # Adaptive k; strict-ish thresholds
#     new_q, new_a, q_old_to_new, q_clusters, kept = ann_merge_questions_answer_gated(
#         codebook_main,
#         questions_lst,
#         answers_lst,
#         cfg=QAnnMergeConfig(
#             ann=ANNConfig(k_neighbors=5, representative_method="medoid"),
#             gate=QMergeGate(q_sim_threshold=0.90, a_sim_threshold=0.80, combine_strategy="representative"),
#         )
#     )

#     print("\n=== AFTER ===")
#     print("Q kept:", len(new_q))
#     print("q_old_to_new:", q_old_to_new)
#     print("q_clusters:", q_clusters)
#     print("questions_lst (merged):", new_q)
#     print("answers_lst   (merged):", new_a)

# python optimize_combine_storage.py
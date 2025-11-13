from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch

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


from typing import Dict, List, Tuple
import numpy as np
import torch

def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.as_tensor(b)
    if a.ndim == 1: a = a.unsqueeze(0)
    if b.ndim == 1: b = b.unsqueeze(0)
    if a.numel() == 0 or b.numel() == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a @ b.T).detach().cpu().numpy().astype(np.float32)

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def get_overped_or_unique_edge_lists_sentence_emebed(
    codebook_main: Dict,
    edge_lists: List[List[int]],
    sent_emb,                      # must have .embed_documents(List[str]) -> List[List[float]]
    sim_threshold: float = 0.98,
    unique: bool = False,          # if True, consolidate survivors into list 0
    mode: str = "global_one",        # "per_list" or "global_one"
) -> List[List[int]]:
    """
    Deduplicate near-duplicate triples across lists using sentence-embedding cosine similarity.

    - Build similarity ONLY across different lists (never within a list).
    - DSU groups edges into components via (cos >= sim_threshold).
    - Selection:
        * mode="per_list": for each component, keep ONE representative per participating list
                           (longest decoded text in that list; tie -> larger eid).
        * mode="global_one": for each component, keep ONE representative globally (longest; tie -> larger eid).
    - If unique=True: after selection, consolidate all survivors into list 0 (others emptied).
      If unique=False: survivors remain in their original lists, preserving order.
    """
    # ---- 1) Collect unique edge IDs
    uniq_edge_ids = sorted({eid for lst in edge_lists for eid in lst})
    if not uniq_edge_ids:
        return [[] for _ in edge_lists]

    E = codebook_main["e"]
    R = codebook_main["r"]
    edge_matrix = codebook_main["edge_matrix"]

    # ---- 2) Build text per edge id and batch-embed once
    sentences, lengths = [], []
    for eid in uniq_edge_ids:
        h, ridx, t = edge_matrix[eid]
        s = f"{E[h]} {R[ridx]} {E[t]}".strip()
        sentences.append(s)
        lengths.append(len(s))
    embs = torch.tensor(sent_emb.embed_documents(sentences), dtype=torch.float32)

    row_of_edge = {eid: i for i, eid in enumerate(uniq_edge_ids)}
    len_of_edge = {eid: lengths[i] for i, eid in enumerate(uniq_edge_ids)}

    # ---- 3) Per-list embeddings aligned to edge_lists
    per_list_embs: List[torch.Tensor] = []
    for lst in edge_lists:
        if lst:
            per_list_embs.append(embs[[row_of_edge[eid] for eid in lst]])
        else:
            per_list_embs.append(torch.empty((0, embs.shape[1]), dtype=torch.float32))

    # ---- 4) Global indexing over all items
    offsets, total = [], 0
    for lst in edge_lists:
        offsets.append(total)
        total += len(lst)
    if total == 0:
        return [[] for _ in edge_lists]

    def gid(li: int, i_local: int) -> int:
        return offsets[li] + i_local

    # reverse map: global -> (list_id, local_idx)
    rev: List[Tuple[int,int]] = []
    for li, lst in enumerate(edge_lists):
        for i_local in range(len(lst)):
            rev.append((li, i_local))

    # ---- 5) Union overlaps ACROSS DIFFERENT LISTS ONLY
    dsu = DSU(total)  # your DSU

    L = len(edge_lists)
    for a in range(L):
        Ei = per_list_embs[a]
        if Ei.numel() == 0:
            continue
        for b in range(a + 1, L):
            Ej = per_list_embs[b]
            if Ej.numel() == 0:
                continue
            S = _cosine_matrix(Ei, Ej)  # your cosine helper
            ai, bj = np.where(S >= sim_threshold)
            for i_local, j_local in zip(ai.tolist(), bj.tolist()):
                dsu.union(gid(a, i_local), gid(b, j_local))

    # ---- 6) Components: root -> [global indices]
    comps: Dict[int, List[int]] = {}
    for g in range(total):
        comps.setdefault(dsu.find(g), []).append(g)

    # ---- 7) Select representatives
    chosen_globals: set = set()

    if mode == "global_one":
        # one representative per component (global)
        def member_key(gidx: int):
            li, i_local = rev[gidx]
            eid = edge_lists[li][i_local]
            return (len_of_edge[eid], eid)  # prefer longer; tie-break by larger eid
        for members in comps.values():
            best_g = max(members, key=member_key)
            chosen_globals.add(best_g)

    elif mode == "per_list":
        # one representative per component *per list*
        for members in comps.values():
            # group members by list id
            by_list: Dict[int, List[int]] = {}
            for gidx in members:
                li, _ = rev[gidx]
                by_list.setdefault(li, []).append(gidx)

            # choose best within each list for this component
            for li, gidxs in by_list.items():
                def member_key_in_list(gidx: int):
                    _li, i_local = rev[gidx]
                    eid = edge_lists[_li][i_local]
                    return (len_of_edge[eid], eid)
                best_g = max(gidxs, key=member_key_in_list)
                chosen_globals.add(best_g)
    else:
        raise ValueError("mode must be 'per_list' or 'global_one'")

    # ---- 8) Rebuild outputs
    if unique:
        # consolidate survivors into list 0, preserving first-seen order across lists
        merged: List[int] = []
        seen = set()
        for li, lst in enumerate(edge_lists):
            for i_local, eid in enumerate(lst):
                if gid(li, i_local) in chosen_globals and eid not in seen:
                    merged.append(eid); seen.add(eid)
        kept_edge_lists = [[] for _ in edge_lists]
        kept_edge_lists[0] = merged
        filtered = [x for x in kept_edge_lists if x] 
        return filtered
    else:
        # keep winners in their original lists (stable order)
        kept_edge_lists: List[List[int]] = [[] for _ in edge_lists]
        for li, lst in enumerate(edge_lists):
            for i_local, eid in enumerate(lst):
                if gid(li, i_local) in chosen_globals:
                    kept_edge_lists[li].append(eid)
        filtered = [x for x in kept_edge_lists if x]  
        return filtered

############# optimized ver
def _to_vec(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=np.float32)

def _avg_vec_from_decoded(decoded_q, dim: int) -> np.ndarray:
    """decoded_q: [[e_vec, r_vec, e_vec], ...] -> mean vector over all parts."""
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

def _cosine_sim_word(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine sim for (n,d) vs (m,d); robust to 1D by reshaping."""
    A = np.atleast_2d(A.astype(np.float32, copy=False))
    B = np.atleast_2d(B.astype(np.float32, copy=False))
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T  # (n,m)

def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    # (kept for parity; not used in this optimized path)
    if not isinstance(a, torch.Tensor): a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.as_tensor(b)
    if a.ndim == 1: a = a.unsqueeze(0)
    if b.ndim == 1: b = b.unsqueeze(0)
    if a.numel() == 0 or b.numel() == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a @ b.T).detach().cpu().numpy().astype(np.float32)

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n)); self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: self.p[ra] = rb
        elif self.r[ra] > self.r[rb]: self.p[rb] = ra
        else: self.p[rb] = ra; self.r[ra] += 1

def get_overped_or_unique_edge_lists_sentence_emebed_optimized(
    codebook_main: Dict,
    edge_lists: List[List[int]],
    sent_emb,                         # must have .embed_documents(List[str]) -> List[List[float]]
    sim_threshold: float = 0.90,      # used for both prefilter and final check
    unique: bool = False,             # if True, merge survivors into list 0
    mode: str = "global_one",           # "per_list" or "global_one"
) -> List[List[int]]:
    """
    Optimized dedup with prefilter + final sentence-embedding similarity.

    - Prefilter: cosine on avg decoded embeddings (cheap).
    - Final: cosine on sentence embeddings (accurate).
    - Unions are ONLY across different lists (avoid within-list merges).
    - Selection:
        * mode="per_list": keep ONE best per participating list in each DSU component.
        * mode="global_one": keep ONE best globally per component.
    - unique=True -> unique survivors into list 0; else keep in original lists.
    """
    # ---- 0) unique edge ids
    uniq_edge_ids = sorted({eid for lst in edge_lists for eid in lst})
    if not uniq_edge_ids:
        return [[] for _ in edge_lists]

    E = codebook_main["e"]; R = codebook_main["r"]; edge_matrix = codebook_main["edge_matrix"]
    dim = len(codebook_main["e_embeddings"][0])

    # ---- Map eid -> set of lists that contain it (for cross-list gating)
    eid_in_lists: Dict[int, set] = {eid: set() for eid in uniq_edge_ids}
    for li, lst in enumerate(edge_lists):
        for eid in lst:
            eid_in_lists[eid].add(li)

    # ---- 1) Prefilter embeddings via decoded triples
    avg_vecs = []
    for eid in uniq_edge_ids:
        decoded = decode_question([eid], codebook_main, fmt='embeddings')  # NOTE: [eid], not eid
        avg_vecs.append(_avg_vec_from_decoded(decoded, dim))
    avg_vecs = np.asarray(avg_vecs, dtype=np.float32)            # (N,d)
    pre_sim = _cosine_sim_word(avg_vecs, avg_vecs)               # (N,N)
    np.fill_diagonal(pre_sim, -1.0)

    # ---- 2) Sentences + sentence embeddings (once)
    sentences, lengths = [], []
    for eid in uniq_edge_ids:
        h, ridx, t = edge_matrix[eid]
        s = f"{E[h]} {R[ridx]} {E[t]}".strip()
        sentences.append(s)
        lengths.append(len(s))
    sen_embs = np.asarray(sent_emb.embed_documents(sentences), dtype=np.float32)  # (N,d2)
    # L2-normalize for cosine via dot
    sen_embs /= (np.linalg.norm(sen_embs, axis=1, keepdims=True) + 1e-12)

    # ---- 3) DSU over edges (indexing 0..N-1), but only cross-list unions
    N = len(uniq_edge_ids)
    dsu = DSU(N)

    # helper: check if two edge indices appear in different lists
    def cross_list(i_idx: int, j_idx: int) -> bool:
        ei, ej = uniq_edge_ids[i_idx], uniq_edge_ids[j_idx]
        Li, Lj = eid_in_lists[ei], eid_in_lists[ej]
        # at least one pair li != lj
        return any(li != lj for li in Li for lj in Lj)

    cand_i, cand_j = np.where(pre_sim >= sim_threshold)
    for i, j in zip(cand_i.tolist(), cand_j.tolist()):
        if not cross_list(i, j):
            continue  # skip within-list-only similarities
        cos_ij = float(np.dot(sen_embs[i], sen_embs[j]))  # both normalized
        if cos_ij >= sim_threshold:
            dsu.union(i, j)

    # ---- 4) components of edge indices
    comps: Dict[int, List[int]] = {}
    for i in range(N):
        comps.setdefault(dsu.find(i), []).append(i)

    idx_to_eid = {i: eid for i, eid in enumerate(uniq_edge_ids)}
    len_of_eid = {idx_to_eid[i]: lengths[i] for i in range(N)}

    # ---- 5) selection (per component)
    chosen_eids: set = set()

    if mode == "global_one":
        # pick one best (longest; tie->larger eid)
        for members in comps.values():
            best = max(members, key=lambda i: (lengths[i], idx_to_eid[i]))
            chosen_eids.add(idx_to_eid[best])

    elif mode == "per_list":
        # pick one best per participating list
        # Build a quick reverse map from eid to all (list_id, local_idx) occurrences
        # so we can later keep per-list ordering.
        for members in comps.values():
            # group member indices by which lists they occur in
            # We’ll keep one best eid PER list.
            lists_to_candidates: Dict[int, List[int]] = {}
            for i in members:
                eid = idx_to_eid[i]
                for li in eid_in_lists[eid]:
                    lists_to_candidates.setdefault(li, []).append(i)

            for li, i_list in lists_to_candidates.items():
                best_i = max(i_list, key=lambda i: (lengths[i], idx_to_eid[i]))
                chosen_eids.add(idx_to_eid[best_i])

    else:
        raise ValueError("mode must be 'per_list' or 'global_one'")

    # ---- 6) rebuild outputs
    if unique:
        # Merge survivors into list 0 in first-seen order across lists
        merged: List[int] = []
        seen = set()
        for li, lst in enumerate(edge_lists):
            for eid in lst:
                if eid in chosen_eids and eid not in seen:
                    merged.append(eid); seen.add(eid)
        kept = [[] for _ in edge_lists]
        kept[0] = merged
        filtered = [x for x in kept if x]  
        
        return filtered
    else:
        # Keep winners in their original lists (preserve per-list order)
        kept: List[List[int]] = [[] for _ in edge_lists]
        for li, lst in enumerate(edge_lists):
            for eid in lst:
                if eid in chosen_eids:
                    kept[li].append(eid)

        filtered = [x for x in kept if x]  
        return filtered



def get_unique_or_overlap_by_sentence_embedded(codebook_main,edge_lists,sent_emb,sim_threshold=0.9,unique=False,optimized=True):
    
    if optimized:
        kept_edge_lists = get_overped_or_unique_edge_lists_sentence_emebed_optimized(codebook_main,edge_lists,sent_emb,sim_threshold,unique)

    else:
        kept_edge_lists = get_overped_or_unique_edge_lists_sentence_emebed(codebook_main,edge_lists,sent_emb,sim_threshold,unique)

    return kept_edge_lists
       
       

# ---------------------------
# Example usage
# ---------------------------
# if __name__ == "__main__":
#     sent_emb = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en"
#     )


#     final_merged_json_unsliced = {'e': ['most common', 'skin cancer', 'Treatment by risk and recurrence', 'cancer', 'symptoms', 'Quality of life'], 'r': ['facet of'], 'rule': 'Answer questions', 'questions(edges[i])': [[0]], 'edge_matrix': [[0, 0, 1], [2, 0, 3], [4, 0, 5]], 'facts(edges[i])': [[1], [2]]}

#     edge_lists = [[0,1],[0,1,2]]

#     kept_edge_lists =get_overped_or_unique_edge_lists_sentence_emebed(final_merged_json_unsliced, edge_lists,sent_emb,unique = True)

#     print(kept_edge_lists)




# python sentence_embed_overlap.py
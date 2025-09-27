from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

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


def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    # Accept numpy → torch; ensure 2D
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)

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
        if ra == rb:
            return
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
    sim_threshold: float = 0.9,
    unique: bool = False,
) -> List[List[int]]:
    # ---- 1) Collect unique edge IDs
    uniq_edge_ids = sorted({eid for lst in edge_lists for eid in lst})
    if not uniq_edge_ids:
        return [[] for _ in edge_lists]

    e = codebook_main["e"]
    r = codebook_main["r"]
    edge_matrix = codebook_main["edge_matrix"]

    # ---- 2) Build text and batch-embed once
    sentences = []
    lengths = []
    for eid in uniq_edge_ids:
        h, ridx, t = edge_matrix[eid]
        s = f"{e[h]} {r[ridx]} {e[t]}".strip()
        sentences.append(s)
        lengths.append(len(s))
    embs_list = sent_emb.embed_documents(sentences)              # List[List[float]]
    embs = torch.tensor(embs_list, dtype=torch.float32)          # (N, d)
    row_of_edge = {eid: i for i, eid in enumerate(uniq_edge_ids)}
    len_of_edge = {eid: lengths[i] for i, eid in enumerate(uniq_edge_ids)}

    # ---- 3) Per-list embedding tensors aligned to edge_lists
    per_list_embs: List[torch.Tensor] = []
    for lst in edge_lists:
        if lst:
            rows = [row_of_edge[eid] for eid in lst]
            per_list_embs.append(embs[rows])                     # (len(lst), d)
        else:
            per_list_embs.append(torch.empty((0, embs.shape[1]), dtype=torch.float32))

    # ---- 4) Global indexing over list items
    offsets = []
    total = 0
    for lst in edge_lists:
        offsets.append(total)
        total += len(lst)
    if total == 0:
        return [[] for _ in edge_lists]

    def gid(li: int, local_i: int) -> int:
        return offsets[li] + local_i

    # Reverse map: global -> (list_id, local_idx)
    rev: List[Tuple[int, int]] = []
    for li, lst in enumerate(edge_lists):
        for i_local in range(len(lst)):
            rev.append((li, i_local))

    # ---- 5) Union overlaps **across different lists only**
    dsu = DSU(total)
    L = len(edge_lists)
    for a in range(L):
        Ei = per_list_embs[a]
        if Ei.numel() == 0:
            continue
        for b in range(a + 1, L):
            Ej = per_list_embs[b]
            if Ej.numel() == 0:
                continue
            S = _cosine_matrix(Ei, Ej)                            # (Na, Nb)
            ai, bj = np.where(S >= sim_threshold)
            for i_local, j_local in zip(ai.tolist(), bj.tolist()):
                dsu.union(gid(a, i_local), gid(b, j_local))

    # ---- 6) Components
    comps: Dict[int, List[int]] = {}
    for g in range(total):
        comps.setdefault(dsu.find(g), []).append(g)

    # ---- 7) Decide which globals to keep
    chosen_globals: set = set()

    if unique:
        # keep ONLY singletons
        for members in comps.values():
            if len(members) == 1:
                chosen_globals.add(members[0])
    else:
        # keep all singletons; for overlaps keep exactly one global representative (longest text)
        for members in comps.values():
            if len(members) == 1:
                chosen_globals.add(members[0])
            else:
                def member_key(gidx: int):
                    li, i_local = rev[gidx]
                    eid = edge_lists[li][i_local]
                    return (len_of_edge[eid], -eid)               # length first; deterministic tie-break
                best_g = max(members, key=member_key)
                chosen_globals.add(best_g)

    # ---- 8) Rebuild kept lists in original order
    kept_edge_lists: List[List[int]] = [[] for _ in edge_lists]
    for li, lst in enumerate(edge_lists):
        for i_local, eid in enumerate(lst):
            if gid(li, i_local) in chosen_globals:
                kept_edge_lists[li].append(eid)

    # ---- 9) Safety fallback: avoid all-empties if inputs were non-empty
    if all(len(lst) == 0 for lst in kept_edge_lists) and any(len(lst) > 0 for lst in edge_lists):
        # Too aggressive threshold or everything clustered → return originals
        return [list(lst) for lst in edge_lists]

    return kept_edge_lists

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

def get_overped_or_unique_edge_lists_sentence_emebed_optimized(
    codebook_main: Dict,
    edge_lists: List[List[int]],
    sent_emb,                   # must have .embed_documents(List[str]) -> List[List[float]]
    sim_threshold: float = 0.8, # prefilter threshold on avg-emb; final threshold applies on sentence emb
    unique: bool = False,
) -> List[List[int]]:
    # ---- 0) collect uniq ids
    uniq_edge_ids = sorted({eid for lst in edge_lists for eid in lst})
    if not uniq_edge_ids:
        return [[] for _ in edge_lists]

    e = codebook_main["e"]; r = codebook_main["r"]; edge_matrix = codebook_main["edge_matrix"]
    dim = len(codebook_main["e_embeddings"][0])

    # ---- 1) cheap prefilter embeddings via decoded triples (vectorized)
    # build decoded avg vec per edge (use [eid] so decode_question returns a single triple)
    avg_vecs = []
    for eid in uniq_edge_ids:
        decoded = decode_question([eid], codebook_main, fmt='embeddings')
        avg_vecs.append(_avg_vec_from_decoded(decoded, dim))
    avg_vecs = np.asarray(avg_vecs, dtype=np.float32)            # (N,d)
    pre_sim = _cosine_sim_word(avg_vecs, avg_vecs)               # (N,N)
    np.fill_diagonal(pre_sim, -1.0)                              # ignore self

    # ---- 2) sentences + sentence embeddings (once!)
    sentences = []
    lengths = []
    for eid in uniq_edge_ids:
        h, ridx, t = edge_matrix[eid]
        s = f"{e[h]} {r[ridx]} {e[t]}".strip()
        sentences.append(s)
        lengths.append(len(s))
    sen_embs = np.asarray(sent_emb.embed_documents(sentences), dtype=np.float32)  # (N,d2)

    # normalize for cosine
    sen_embs /= (np.linalg.norm(sen_embs, axis=1, keepdims=True) + 1e-12)

    # ---- 3) DSU over edges (not per-list yet): connect if similarity ≥ final threshold.
    # Use prefilter to avoid all pairs; only check pairs where pre_sim >= sim_threshold.
    N = len(uniq_edge_ids)
    dsu = DSU(N)

    # Get candidate pairs above prefilter
    cand_i, cand_j = np.where(pre_sim >= sim_threshold)
    # Compute precise cosine for only those pairs
    for i, j in zip(cand_i.tolist(), cand_j.tolist()):
        # precise cos (single dot since normalized)
        cos_ij = float(np.dot(sen_embs[i], sen_embs[j]))
        if cos_ij >= sim_threshold:
            dsu.union(i, j)

    # ---- 4) build components of edge ids
    comps: Dict[int, List[int]] = {}
    for i in range(N):
        comps.setdefault(dsu.find(i), []).append(i)  # store indices into uniq_edge_ids

    # ---- 5) choose keep set by semantics
    idx_to_eid = {i: eid for i, eid in enumerate(uniq_edge_ids)}
    len_of_eid = {idx_to_eid[i]: lengths[i] for i in range(N)}

    keep_eids: set = set()
    if unique:
        # keep only singletons
        for members in comps.values():
            if len(members) == 1:
                keep_eids.add(idx_to_eid[members[0]])
    else:
        # keep all singletons; for overlaps keep one longest globally
        for members in comps.values():
            if len(members) == 1:
                keep_eids.add(idx_to_eid[members[0]])
            else:
                # pick longest text; deterministic tie-break on eid
                best = max(members, key=lambda i: (lengths[i], -idx_to_eid[i]))
                keep_eids.add(idx_to_eid[best])

    # ---- 6) rebuild per-list in original order
    kept_edge_lists: List[List[int]] = []
    for lst in edge_lists:
        if unique:
            # keep only singletons present in this list
            kept_edge_lists.append([eid for eid in lst if eid in keep_eids])
        else:
            # keep representative if it happens to be in this list, plus all singletons
            kept_edge_lists.append([eid for eid in lst if eid in keep_eids])

    return kept_edge_lists




def get_unique_or_overlap_by_sentence_embedded(codebook_main,edge_lists,sent_emb,sim_threshold=0.8,unique=False,optimized=False):
    
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
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )


#     final_merged_json_unsliced = {'e': ['most common', 'skin cancer', 'Treatment by risk and recurrence', 'cancer', 'symptoms', 'Quality of life'], 'r': ['facet of'], 'rule': 'Answer questions', 'questions(edges[i])': [[0]], 'edge_matrix': [[0, 0, 1], [2, 0, 3], [4, 0, 5]], 'facts(edges[i])': [[1], [2]]}

#     edge_lists = [[0,1,2],[0,2]]

#     kept_edge_lists =get_overped_or_unique_edge_lists_sentence_emebed(final_merged_json_unsliced, edge_lists,sent_emb)

#     print(kept_edge_lists)




# python sentence_embed_overlap.py
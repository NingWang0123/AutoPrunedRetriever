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
    sent_emb,  # must have .embed_documents(List[str]) -> List[List[float]]
    sim_threshold: float = 0.8,
    unique = False,
) -> Tuple[List[List[int]], Dict[int, List[Tuple[int, int, int]]]]:
    """
    Keep exactly one overlapping triple (by longest text) across lists based on sentence-embedding similarity.

    Returns:
      kept_edge_lists: same shape as edge_lists, but with overlaps collapsed (only the chosen representative kept).
      groups: {rep_edge_id: [(list_id, local_idx, edge_id), ...]} membership per connected component.
    """

    # ---- 1) Gather all unique edge ids we need
    uniq_edge_ids = sorted({eid for lst in edge_lists for eid in lst})
    if not uniq_edge_ids:
        return [[] for _ in edge_lists], {}

    # ---- 2) Build text for each edge id and batch-embed
    e = codebook_main["e"]
    r = codebook_main["r"]
    edge_matrix = codebook_main["edge_matrix"]

    sentences = []
    lengths = []
    for eid in uniq_edge_ids:
        eh, ridx, et = edge_matrix[eid]
        sent = f"{e[eh]} {r[ridx]} {e[et]}".strip()
        sentences.append(sent)
        lengths.append(len(sent))

    # sent_emb.embed_documents expects a list[str]
    embs_list = sent_emb.embed_documents(sentences)  # List[List[float]]
    embs = torch.tensor(embs_list, dtype=torch.float32)

    # Map edge_id -> (embedding row idx, length)
    row_of_edge = {eid: i for i, eid in enumerate(uniq_edge_ids)}
    len_of_edge = {eid: lengths[i] for i, eid in enumerate(uniq_edge_ids)}

    # ---- 3) Build per-list embedding tensors aligned to edge_lists
    per_list_embs: List[torch.Tensor] = []
    for lst in edge_lists:
        if lst:
            rows = [row_of_edge[eid] for eid in lst]
            per_list_embs.append(embs[rows])         # [len(lst), dim]
        else:
            per_list_embs.append(torch.empty((0, embs.shape[1]), dtype=torch.float32))

    # ---- 4) Global indexing over all elements for DSU
    offsets = []
    total = 0
    for lst in edge_lists:
        offsets.append(total)
        total += len(lst)

    def gid(li: int, local_i: int) -> int:
        # Global id of the li-th list's local index
        return offsets[li] + local_i

    if total == 0:
        return [[] for _ in edge_lists], {}

    dsu = DSU(total)

    # ---- 5) Union overlaps across different lists (a < b)
    L = len(edge_lists)
    for a in range(L):
        Ei = per_list_embs[a]
        if Ei.numel() == 0:
            continue
        for b in range(a + 1, L):
            Ej = per_list_embs[b]
            if Ej.numel() == 0:
                continue
            S = _cosine_matrix(Ei, Ej)  # [Na, Nb]
            ai, bj = np.where(S >= sim_threshold)
            # Merge matched pairs
            for i_local, j_local in zip(ai.tolist(), bj.tolist()):
                dsu.union(gid(a, i_local), gid(b, j_local))

    # ---- 6) Collect components
    comps: Dict[int, List[int]] = {}
    for g in range(total):
        root = dsu.find(g)
        comps.setdefault(root, []).append(g)

    # Reverse mapping global -> (list_id, local_idx)
    rev: List[Tuple[int, int]] = []
    for li, lst in enumerate(edge_lists):
        for i_local in range(len(lst)):
            rev.append((li, i_local))

    # ---- 7) Pick representative by longest text; record groups
    kept_edge_lists: List[List[int]] = [[] for _ in edge_lists]
    print(kept_edge_lists)

    if unique:
        min_len_m = 1
    else:
        min_len_m = 2

    for _, members in comps.items():
        # choose best by longest text
        # compute length for each member
        if len(members)>=min_len_m:
            def member_len(gidx: int) -> int:
                li, i_local = rev[gidx]
                eid = edge_lists[li][i_local]
                print(f'edge: {len_of_edge[eid]} {eid}')
                return len_of_edge[eid]

            best_g = max(members, key=member_len)
            rep_li, rep_i = rev[best_g]
            rep_eid = edge_lists[rep_li][rep_i]


            # Keep only the representative in its original list
            print(rep_eid)
            print(members)
            kept_edge_lists[rep_li].append(rep_eid)

    return kept_edge_lists


############# optimized ver
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

def _cosine_sim_word(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: (n, d), B: (m, d) -> (n, m) cosine similarity matrix.
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def get_overped_or_unique_edge_lists_sentence_emebed_optimized(
    codebook_main: Dict,
    edge_lists: List[List[int]],
    sent_emb,  # must have .embed_documents(List[str]) -> List[List[float]]
    sim_threshold: float = 0.8,
    unique = False,
) -> Tuple[List[List[int]], Dict[int, List[Tuple[int, int, int]]]]:
    """
    Keep exactly one overlapping triple (by longest text) across lists based on sentence-embedding similarity.

    Returns:
      kept_edge_lists: same shape as edge_lists, but with overlaps collapsed (only the chosen representative kept).
      groups: {rep_edge_id: [(list_id, local_idx, edge_id), ...]} membership per connected component.
    """

    # ---- 1) Gather all unique edge ids we need
    uniq_edge_ids = sorted({eid for lst in edge_lists for eid in lst})
    if not uniq_edge_ids:
        return [[] for _ in edge_lists], {}


    # instead of using sent_emb only, use word embedding sim first to sort the matrix
    word_embeds = []
    dim = len(codebook_main["e_embeddings"][0])

    for q_edges in uniq_edge_ids:
        decoded = decode_question(q_edges, codebook_main, fmt='embeddings')
        word_embeds.append(_avg_vec_from_decoded(decoded, dim))

    # get word embedd matrix
    word_embedd_len = len(word_embeds)
    word_sim_matrix = np.zeros((word_embedd_len, word_embedd_len)) 

    for i,word_embed_i in enumerate(word_embeds):
      for j,word_embed_j in enumerate(word_embeds):

        if i != j:
          word_sim =_cosine_sim_word(word_embed_i,word_embed_j)
          word_sim_matrix[i][j] = word_sim
        else:
          word_sim_matrix[i][j] = -1


    # get all possible pairs and sort them from high to low

    e = codebook_main["e"]
    r = codebook_main["r"]
    edge_matrix = codebook_main["edge_matrix"]

    # flatten and get sorted indices (descending)
    flat = word_sim_matrix.ravel()
    order = np.argsort(-flat)  # indices of elements sorted high → low

    # convert back to row, col indices
    rows, cols = np.unravel_index(order, word_sim_matrix.shape)
    edge_sentemb_dict = {}
    removed_val = []
    kept_val = []

    # iterate and pruning
    for rank, (r, c) in enumerate(zip(rows, cols), start=1):

      edge_r = uniq_edge_ids[r]

      edge_c = uniq_edge_ids[c]

      if edge_r in removed_val or edge_c in removed_val:
        continue

      else:
        # get edge_r info
        if edge_r in edge_sentemb_dict:
          sent_r_emb,len_sent_r = edge_sentemb_dict[edge_r]
        else:
          eh_r, ridx_r, et_r = edge_matrix[edge_r]
          sent_r = f"{e[eh_r]} {r[ridx_r]} {e[et_r]}".strip()
          len_sent_r = len(sent_r)
          sent_r_emb = sent_emb.embed_documents([sent_r])
          edge_sentemb_dict[edge_r] = (sent_r_emb,len_sent_r)

        # get edge_c info
        if edge_c in edge_sentemb_dict:
          sent_c_emb,len_sent_c = edge_sentemb_dict[edge_c]
        else:
          eh_c, ridx_c, et_c = edge_matrix[edge_c]
          sent_c = f"{e[eh_c]} {r[ridx_c]} {e[et_c]}".strip()
          len_sent_c = len(sent_c)
          sent_c_emb = sent_emb.embed_documents([sent_c])
          edge_sentemb_dict[edge_c] = (sent_c_emb,len_sent_c)

        S = _cosine_matrix(sent_r_emb, sent_c_emb)

        if S >= sim_threshold:
          if len_sent_r>= len_sent_c:
            removed_val.append(edge_c)
            kept_val.append(edge_r)
            if edge_c in kept_val:
              kept_val.remove(edge_c)
          else:
            removed_val.append(edge_r)
            if edge_r in kept_val:
              kept_val.remove(edge_r)

    kept_edge_lists = edge_lists.copy()

    if unique:
      for i,edge_list in enumerate(kept_edge_lists):
        kept_edge_lists[i] = [edge for edge in edge_list if edge not in removed_val]
    else:
      for i,edge_list in enumerate(kept_edge_lists):
        kept_edge_lists[i] = [edge for edge in edge_list if edge in kept_val]

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
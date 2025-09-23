from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings


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


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    sent_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    final_merged_json_unsliced = {'e': ['most common', 'skin cancer', 'Treatment by risk and recurrence', 'cancer', 'symptoms', 'Quality of life'], 'r': ['facet of'], 'rule': 'Answer questions', 'questions(edges[i])': [[0]], 'edge_matrix': [[0, 0, 1], [2, 0, 3], [4, 0, 5]], 'facts(edges[i])': [[1], [2]]}

    edge_lists = [[0,1,2],[0,2]]

    kept_edge_lists =get_overped_edge_lists_sentence_emebed(final_merged_json_unsliced, edge_lists,sent_emb)

    print(kept_edge_lists)



# python sentence_embed_overlap.py
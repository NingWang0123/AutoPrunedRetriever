from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

Triple = Tuple[str, str, str]  # (head, relation, tail)

def triple_to_text(triple: Triple) -> str:
    """Format a triple as a sentence-like string."""
    h, r, t = triple
    return f"{h} {r} {t}"

def embed_triples_with_langchain(
    sent_emb: HuggingFaceEmbeddings,
    lists_of_triples: List[List[Triple]],
    mode: str = "concat"
):
    """
    Encode triples using a LangChain HuggingFaceEmbeddings object.
    mode = "concat" → whole triple as text
    mode = "component" → average of head, relation, tail
    """
    embs = []
    for triples in lists_of_triples:
        if not triples:
            embs.append(torch.empty((0, sent_emb.client.get_sentence_embedding_dimension())))
            continue

        if mode == "concat":
            texts = [triple_to_text(t) for t in triples]
            arr = sent_emb.embed_documents(texts)  # returns list of lists
            embs.append(torch.tensor(arr, dtype=torch.float32))
        elif mode == "component":
            heads = sent_emb.embed_documents([t[0] for t in triples])
            rels  = sent_emb.embed_documents([t[1] for t in triples])
            tails = sent_emb.embed_documents([t[2] for t in triples])
            arr = (np.array(heads) + np.array(rels) + np.array(tails)) / 3.0
            embs.append(torch.tensor(arr, dtype=torch.float32))
        else:
            raise ValueError("mode must be 'concat' or 'component'")
    return embs

def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Compute cosine similarity matrix between two embedding sets."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a @ b.T).cpu().numpy()

def compute_overlap_with_langchain(
    sent_emb: HuggingFaceEmbeddings,
    lists_of_triples: List[List[Triple]],
    mode: str = "concat"
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute semantic overlap matrices across all list pairs.
    Returns dict {(i,j): similarity_matrix}
    """
    embs = embed_triples_with_langchain(sent_emb, lists_of_triples, mode=mode)
    pair_mats = {}
    for i in range(len(lists_of_triples)):
        for j in range(i+1, len(lists_of_triples)):
            pair_mats[(i,j)] = cosine_matrix(embs[i], embs[j])
    return pair_mats

def top_matches(
    list_a: List[Triple],
    list_b: List[Triple],
    sim_matrix: np.ndarray,
    topk: int = 5
) -> pd.DataFrame:
    """Get top-k matches across two triple lists."""
    rows = []
    for i in range(sim_matrix.shape[0]):
        best_idx = np.argsort(-sim_matrix[i])[:topk]
        for j in best_idx:
            rows.append({
                "a_idx": i,
                "b_idx": j,
                "a_triple": list_a[i],
                "b_triple": list_b[j],
                "similarity": sim_matrix[i,j]
            })
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)

### naive method

def _embed_lists(
    sent_emb: HuggingFaceEmbeddings,
    lists_of_triples: List[List[Triple]],
    mode: str = "concat",
):
    """Return per-list embedding tensors (Ni x D), CPU float32, not normalized."""
    embs = []
    for triples in lists_of_triples:
        if not triples:
            embs.append(torch.empty((0, 0), dtype=torch.float32))
            continue
        if mode == "concat":
            texts = [triple_to_text(t) for t in triples]
            arr = sent_emb.embed_documents(texts)  # list[list[float]]
            embs.append(torch.tensor(arr, dtype=torch.float32))
        elif mode == "component":
            H = np.array(sent_emb.embed_documents([t[0] for t in triples]))
            R = np.array(sent_emb.embed_documents([t[1] for t in triples]))
            T = np.array(sent_emb.embed_documents([t[2] for t in triples]))
            arr = (H + R + T) / 3.0
            embs.append(torch.tensor(arr, dtype=torch.float32))
        else:
            raise ValueError("mode must be 'concat' or 'component'")
    return embs

def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    if a.numel() == 0 or b.numel() == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a @ b.T).cpu().numpy().astype(np.float32)

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def select_informative_overlaps(
    sent_emb: HuggingFaceEmbeddings,
    lists_of_triples: List[List[Triple]],
    mode: str = "concat",
    sim_threshold: float = 0.6,
):
    """
    Build an overlap graph across all lists (edges if cosine >= sim_threshold).
    For each connected component, keep the triple with the largest text length.
    Returns:
      kept: List[dict] with representative info
      groups: Dict[rep_global_idx, List[dict]] of full component membership
    """
    # Flatten index mapping
    offsets = []
    all_triples: List[Triple] = []
    for li, lst in enumerate(lists_of_triples):
        offsets.append(len(all_triples))
        all_triples.extend(lst)
    N = len(all_triples)
    if N == 0:
        return [], {}

    # Embed per list, then stitch pairwise to avoid O(N^2) memory blowups
    per_list_embs = _embed_lists(sent_emb, lists_of_triples, mode=mode)

    # Union-Find over global indices; add edge if sim >= threshold
    dsu = DSU(N)

    # Helper to convert local (li, i) -> global index
    def gid(li, i): return offsets[li] + i

    # Pairwise list sims (cross-list only)
    L = len(lists_of_triples)
    for a in range(L):
        for b in range(a+1, L):
            Ei, Ej = per_list_embs[a], per_list_embs[b]
            if Ei.numel() == 0 or Ej.numel() == 0: 
                continue
            S = _cosine_matrix(Ei, Ej)  # [Na, Nb]
            # Find all pairs meeting the threshold (vectorized)
            ai, bj = np.where(S >= sim_threshold)
            for i_local, j_local in zip(ai.tolist(), bj.tolist()):
                dsu.union(gid(a, i_local), gid(b, j_local))

    # Build components
    comps: Dict[int, List[int]] = {}
    for g in range(N):
        r = dsu.find(g)
        comps.setdefault(r, []).append(g)

    # Pick representative = longest triple text
    def text_len(t: Triple) -> int:
        return len(triple_to_text(t))

    kept = []
    groups = {}
    # For reverse lookup: global -> (list_id, local_id)
    rev = []
    for li, lst in enumerate(lists_of_triples):
        for i in range(len(lst)):
            rev.append((li, i))  # aligns with global order

    for root, members in comps.items():
        # choose by longest text
        best_g = max(members, key=lambda g: text_len(all_triples[g]))
        rep_li, rep_i = rev[best_g]
        rep_triple = lists_of_triples[rep_li][rep_i]

        # record group members (with where they came from)
        group_detail = []
        for g in members:
            li, i = rev[g]
            t = lists_of_triples[li][i]
            group_detail.append({
                "list_id": li,
                "local_idx": i,
                "triple": t,
                "text_len": text_len(t),
            })

        groups[best_g] = group_detail
        kept.append({
            "rep_global_idx": best_g,
            "list_id": rep_li,
            "local_idx": rep_i,
            "triple": rep_triple,
            "text_len": text_len(rep_triple),
            "group_size": len(members),
        })

    # Optional: sort kept by group size (desc) then text_len (desc)
    kept.sort(key=lambda d: (d["group_size"], d["text_len"]), reverse=True)
    return kept, groups


# 

def _cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    if a.numel() == 0 or b.numel() == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a @ b.T).cpu().numpy().astype(np.float32)

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def get_overped_edge_lists_sentence_emebed(codebook_main, edge_lists,sent_emb,sim_threshold = 0.8):
    """
    Input:
      codebook_main:
          {
              "e": [str, ...],
              "r": [str, ...],
              "edge_matrix": [[e_idx, r_idx, e_idx], ...],  # list or np.ndarray
              "questions": [[edges index,...],...]
              "e_embeddings": [vec, ...], 
              "r_embeddings": [vec, ...], }
      edge_lists: list[List[edge_matrix index,...],...] edge_matrix index here is the  

    Output:
      kept_edge_lists: list[List[edge_matrix index,...],...] edge_matrix index here is the 

    """

    # get the edge matrix belongs to edge_lists

    unique_edge_lists_flat = list(set([x for sublist in edge_lists for x in sublist]))

    # get corresponds sentence embeddings
    all_ere_sent = {}
    all_ere_len = {}

    for i in unique_edge_lists_flat:
      edge_mat_i = codebook_main['edge_matrix'][i]
      e_index_h,r_index,e_index_t = edge_mat_i
      e_h = codebook_main['e'][e_index_h]
      r = codebook_main['r'][r_index]
      e_t = codebook_main['e'][e_index_t]
      ere_sent = f"{e_h} {r} {e_t}"
      all_ere_sent[i] = sent_emb.embed_documents(ere_sent) 
      all_ere_len[i] = len(ere_sent)


    # calculate cos sim
    # Union-Find over global indices; add edge if sim >= threshold
    dsu = DSU(N)

    # Helper to convert local (li, i) -> global index
    def gid(li, i): return offsets[li] + i

    # Pairwise list sims (cross-list only)
    # Flatten index mapping
    offsets = []
    all_triples = []
    for li, lst in enumerate(edge_lists):
        offsets.append(len(all_triples))
        all_triples.extend(lst)
    N = len(all_triples)
    if N == 0:
        return [], {}

    L = len(edge_lists)
    for a in range(L):
        for b in range(a+1, L):
            Ei, Ej = all_ere_sent[a], all_ere_sent[b]
            if Ei.numel() == 0 or Ej.numel() == 0: 
                continue
            S = _cosine_matrix(Ei, Ej)  # [Na, Nb]
            # Find all pairs meeting the threshold (vectorized)
            ai, bj = np.where(S >= sim_threshold)
            for i_local, j_local in zip(ai.tolist(), bj.tolist()):
                dsu.union(gid(a, i_local), gid(b, j_local))

    # Build components
    comps: Dict[int, List[int]] = {}
    for g in range(N):
        r = dsu.find(g)
        comps.setdefault(r, []).append(g)

    kept = []
    groups = {}
    # For reverse lookup: global -> (list_id, local_id)
    rev = []
    for li, lst in enumerate(edge_lists):
        for i in range(len(lst)):
            rev.append((li, i))  # aligns with global order

    for root, members in comps.items():
        # choose by longest text
        best_g = max(members, key=lambda i: all_ere_len[i])
        rep_li, rep_i = rev[best_g]
        rep_triple = edge_lists[rep_li][rep_i]

        # record group members (with where they came from)
        group_detail = []
        for g in members:
            li, i = rev[g]
            t = edge_lists[li][i]

            print(t)

            print((li, i))

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    sent_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    final_merged_json_unsliced = {'e': ['most common', 'skin cancer', 'Treatment by risk and recurrence', 'cancer', 'symptoms', 'Quality of life'], 'r': ['facet of'], 'rule': 'Answer questions', 'questions(edges[i])': [[0]], 'edge_matrix': [[0, 0, 1], [2, 0, 3], [4, 0, 5]], 'facts(edges[i])': [[1], [2]]}

    edge_lists = [[0,1],[0,2]]

    get_overped_edge_lists_sentence_emebed(final_merged_json_unsliced, edge_lists,sent_emb)



# python sentence_embed_overlap.py
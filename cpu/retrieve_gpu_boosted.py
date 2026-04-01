import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext
from langchain_community.embeddings import HuggingFaceEmbeddings

# ======================================================
# Global defaults / knobs
# ======================================================
def _resolve_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Prefer MPS on Apple Silicon if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

_DEFAULT_DEVICE = _resolve_default_device()
_DEFAULT_DTYPE  = torch.float16 if _DEFAULT_DEVICE.type == "cuda" else torch.float32

# Better matmul perf on Ampere+ (safe no-op on older)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ======================================================
# Your decoder (kept as-is; adjust to your codebook keys)
# ======================================================
def decode_question(question, codebook_main, fmt='words'):
    """
    question: list[int] of edge indices
    codebook_main:
        {
            "e": [str, ...],
            "r": [str, ...],
            "edge_matrix": [[e_idx, r_idx, e_idx], ...],
            "questions_lst": [[edges index,...],...],
            "answers_lst":   [[edges index,...],...],
            "facts_lst":     [[edges index,...],...],
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
        return edges[i]

    if fmt == 'words':
        E, R = codebook_main["e"], codebook_main["r"]
        return [[E[h], R[r], E[t]] for (h, r, t) in (get_edge(i) for i in idxs)]
    elif fmt == 'embeddings':
        Ee = codebook_main.get("e_embeddings")
        Re = codebook_main.get("r_embeddings")
        if Ee is None or Re is None:
            raise KeyError("e_embeddings and r_embeddings are required for fmt='embeddings'.")
        return [[Ee[h], Re[r], Ee[t]] for (h, r, t) in (get_edge(i) for i in idxs)]
    elif fmt == 'edges':
        return [[h, r, t] for (h, r, t) in (get_edge(i) for i in idxs)]
    else:
        raise ValueError("fmt must be 'words', 'embeddings' or 'edges'.")


# ======================================================
# Unified autocast (fixes your error + deprecation)
# ======================================================
def _autocast_ctx(device: torch.device, use_amp: bool, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype)
    # MPS autocast can be fragile across PyTorch versions; keep it off by default
    if device.type == "mps":
        return nullcontext()
    if device.type == "cpu":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    return nullcontext()


# ======================================================
# Low-level tensor helpers
# ======================================================
def _to_tensor(x, device=_DEFAULT_DEVICE, dtype=_DEFAULT_DTYPE) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    arr = np.asarray(x, dtype=np.float32, order="C")
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

def _to_tensor_or_none(x: Optional[np.ndarray],
                       device=_DEFAULT_DEVICE,
                       dtype=_DEFAULT_DTYPE) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device=device, dtype=dtype)

def _l2norm_rows_torch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return F.normalize(X, p=2, dim=1, eps=eps)

def _cosine_sim_matrix_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.numel() == 0 or B.numel() == 0:
        return torch.zeros((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    An = _l2norm_rows_torch(A)
    Bn = _l2norm_rows_torch(B)
    return An @ Bn.T  # (na, nb)


# ======================================================
# Entity/Relation extraction (torch)
# ======================================================
def _extract_entities_relations_from_run_torch(
    edge_run: List[int],
    codebook_main: Dict[str, Any],
    device=_DEFAULT_DEVICE,
    dtype=_DEFAULT_DTYPE,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
      E: (n_e, de) torch or None    # heads+tails
      R: (n_r, dr) torch or None    # relations
    """
    decoded = decode_question(edge_run, codebook_main, fmt='embeddings')
    Es, Rs = [], []
    for h_vec, r_vec, t_vec in decoded:
        if h_vec is not None:
            Es.append(np.asarray(h_vec, dtype=np.float32))
        if t_vec is not None:
            Es.append(np.asarray(t_vec, dtype=np.float32))
        if r_vec is not None:
            Rs.append(np.asarray(r_vec, dtype=np.float32))
    E = _to_tensor_or_none(np.vstack(Es).astype(np.float32) if Es else None, device, dtype)
    R = _to_tensor_or_none(np.vstack(Rs).astype(np.float32) if Rs else None, device, dtype)
    return E, R


# ======================================================
# Max-pair cosine (torch)
# ======================================================
@torch.no_grad()
def _pairwise_max_cos_torch(A: Optional[torch.Tensor],
                            B: Optional[torch.Tensor]) -> torch.Tensor:
    if (A is None) or (B is None):
        return torch.zeros((), device=_DEFAULT_DEVICE, dtype=_DEFAULT_DTYPE)
    if A.numel() == 0 or B.numel() == 0:
        return torch.zeros((), device=A.device, dtype=A.dtype)
    An = _l2norm_rows_torch(A); Bn = _l2norm_rows_torch(B)
    return torch.max(An @ Bn.T)

@torch.no_grad()
def entrel_maxpair_similarity_torch(Eq: Optional[torch.Tensor],
                                    Rq: Optional[torch.Tensor],
                                    Ef: Optional[torch.Tensor],
                                    Rf: Optional[torch.Tensor],
                                    w_ent: float = 1.0,
                                    w_rel: float = 0.5) -> torch.Tensor:
    ent_score = _pairwise_max_cos_torch(Eq, Ef)
    rel_score = _pairwise_max_cos_torch(Rq, Rf)
    return ent_score * float(w_ent) + rel_score * float(w_rel)


# ======================================================
# Top-k merge (torch)
# ======================================================
def _merge_topk_torch(prev_scores: torch.Tensor,
                      prev_idx: torch.Tensor,
                      batch_scores: torch.Tensor,
                      batch_idx: torch.Tensor,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if prev_scores.numel() == 0:
        if batch_scores.numel() <= k:
            return batch_scores, batch_idx
        s, i = torch.topk(batch_scores, k, largest=True, sorted=True)
        return s, batch_idx[i]
    all_scores = torch.cat([prev_scores, batch_scores], dim=0)
    all_idx    = torch.cat([prev_idx,  batch_idx],  dim=0)
    k_eff = min(k, all_scores.numel())
    s, i = torch.topk(all_scores, k_eff, largest=True, sorted=True)
    return s, all_idx[i]


# ======================================================
# VRAM-aware auto-tuning of batch sizes
# ======================================================
def _get_vram_free_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device.index or 0)
            return int(free_bytes)
        except Exception:
            pass
    return 0

def _avg_rows_and_dims(pairs: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
    ne_sum = 0; nr_sum = 0; ne_cnt = 0; nr_cnt = 0
    d_e = 0; d_r = 0
    for (E, R) in pairs:
        if E is not None and E.numel() > 0:
            ne_sum += E.shape[0]; ne_cnt += 1; d_e = max(d_e, E.shape[1])
        if R is not None and R.numel() > 0:
            nr_sum += R.shape[0]; nr_cnt += 1; d_r = max(d_r, R.shape[1])
    avg_ne = max(1, ne_sum // max(1, ne_cnt))
    avg_nr = max(0, nr_sum // max(1, nr_cnt))
    if d_e == 0 and d_r == 0:
        d_e = d_r = 1
    if d_e == 0: d_e = d_r
    if d_r == 0: d_r = d_e
    return avg_ne, avg_nr, d_e, d_r

def _estimate_flops_per_pair(avg_ne:int, avg_nr:int, d_e:int, d_r:int) -> float:
    flops_ent = 2.0 * avg_ne * avg_ne * d_e if d_e > 0 else 0.0
    flops_rel = 2.0 * avg_nr * avg_nr * d_r if d_r > 0 else 0.0
    return max(1.0, flops_ent + flops_rel)

def _autotune_batches(
    N_total:int,
    M_total:int,
    avg_ne:int,
    avg_nr:int,
    d_e:int,
    d_r:int,
    device: torch.device,
    use_amp: bool,
    max_vram_utilization: float = 0.6,
    target_min_flops: float = 5e7,  # ↑ default: encourage larger slices
) -> Tuple[int, int]:
    bytes_per_score = 4  # keep row_scores in fp32 for stability
    free_bytes = _get_vram_free_bytes(device)

    if device.type == "cuda" and free_bytes > 0:
        temp_budget = int(free_bytes * max_vram_utilization * 0.25)  # conservative slice of VRAM
        max_q = min(N_total, 64)
        max_db = min(M_total, 8192)  # ↑ allow bigger DB slices
        q_candidates  = [1, 2, 4, 8, 16, 32, 64]
        db_candidates = [128, 256, 512, 1024, 2048, 4096, 8192]
    else:
        temp_budget = 0
        max_q = min(N_total, 32)
        max_db = min(M_total, 1024)
        q_candidates  = [1, 2, 4, 8, 16, 32]
        db_candidates = [128, 256, 512, 1024]

    flops_per_pair = _estimate_flops_per_pair(avg_ne, avg_nr, d_e, d_r)

    best = None
    for qb in q_candidates:
        if qb > max_q: break
        for dbb in db_candidates:
            if dbb > max_db: break

            flops_launch = flops_per_pair * qb * dbb
            if flops_launch < target_min_flops:
                continue

            temp_bytes = qb * dbb * bytes_per_score
            if temp_budget and temp_bytes > temp_budget:
                continue

            score = flops_launch
            if best is None or score > best[0]:
                best = (score, qb, dbb)

    if best is None:
        qb = min(max_q, 16)
        dbb = min(max_db, 512 if device.type == "cpu" else 256)
        return qb, dbb
    return best[1], best[2]


# ======================================================
# Vectorized slice scoring (no per-candidate Python loop)
# ======================================================
def _segment_amax_1d(values: torch.Tensor, seg_ids: torch.Tensor, nseg: int) -> torch.Tensor:
    """
    Per-segment amax over a 1D tensor using scatter_reduce_.
    values:  (N,) float tensor
    seg_ids: (N,) int64 in [0, nseg-1]
    Returns: (nseg,) float tensor with per-segment maxima.
    """
    # ensure shapes/dtypes
    if values.ndim != 1 or seg_ids.ndim != 1:
        raise ValueError("values and seg_ids must be 1D")
    if values.shape[0] != seg_ids.shape[0]:
        raise ValueError("values and seg_ids must have the same length")
    if seg_ids.dtype != torch.int64:
        seg_ids = seg_ids.to(torch.int64)

    # allocate with -inf in the same dtype/device as values
    neg_inf = -float("inf")
    out = torch.full((nseg,), neg_inf, device=values.device, dtype=values.dtype)

    # Preferred fast path (PyTorch ≥1.13): scatter_reduce_
    if hasattr(out, "scatter_reduce_"):
        try:
            # include_self=True so that segments with no elements keep -inf
            out.scatter_reduce_(dim=0, index=seg_ids, src=values, reduce="amax", include_self=True)
            return out
        except Exception:
            # fall through to the portable path
            pass

    # Fallback (older PyTorch): do a stable sort + segmented max on CPU/GPU
    # NOTE: This path is slower but correct; only used on very old versions.
    order = torch.argsort(seg_ids)
    seg_sorted = seg_ids[order]
    vals_sorted = values[order]
    # find segment boundaries
    change = torch.ones_like(seg_sorted, dtype=torch.bool)
    change[1:] = seg_sorted[1:] != seg_sorted[:-1]
    starts = torch.nonzero(change, as_tuple=False).flatten()
    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = seg_sorted.numel()

    # do per-segment max in a tiny Python loop over unique segments
    uniq = seg_sorted[starts]
    for u, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        out[u] = torch.max(vals_sorted[s:e])

    return out


def _slice_entrel_maxpair_scores(
    Eq: Optional[torch.Tensor], Rq: Optional[torch.Tensor],
    Ef_list: List[Optional[torch.Tensor]], Rf_list: List[Optional[torch.Tensor]],
    w_ent: float, w_rel: float
) -> torch.Tensor:
    """
    Exact max-pair cosine per DB item, vectorized over the whole slice.
    Returns (slice_len,) scores.
    """
    device = (Eq.device if (Eq is not None) else
              (Rq.device if (Rq is not None) else _DEFAULT_DEVICE))
    dtype = (Eq.dtype if (Eq is not None) else
             (Rq.dtype if (Rq is not None) else _DEFAULT_DTYPE))
    n = len(Ef_list)

    scores_e = torch.zeros(n, device=device, dtype=torch.float32)
    scores_r = torch.zeros(n, device=device, dtype=torch.float32)

    # ENTITIES
    if Eq is not None and Eq.numel() > 0 and any((E is not None and E.numel() > 0) for E in Ef_list):
        parts = []; segs = []
        for j, E in enumerate(Ef_list):
            if E is not None and E.numel() > 0:
                parts.append(E)
                segs.append(torch.full((E.shape[0],), j, device=device, dtype=torch.int64))
        Ef_all = torch.cat(parts, dim=0)                             # (sum_nc, d)
        seg_id = torch.cat(segs, dim=0)                              # (sum_nc,)
        S = _cosine_sim_matrix_torch(Eq, Ef_all)                     # (nq_i, sum_nc)
        colmax = S.max(dim=0).values                                 # (sum_nc,)
        scores_e = _segment_amax_1d(colmax, seg_id, n).to(torch.float32)

    # RELATIONS
    if Rq is not None and Rq.numel() > 0 and any((R is not None and R.numel() > 0) for R in Rf_list):
        parts = []; segs = []
        for j, R in enumerate(Rf_list):
            if R is not None and R.numel() > 0:
                parts.append(R)
                segs.append(torch.full((R.shape[0],), j, device=device, dtype=torch.int64))
        if parts:
            Rf_all = torch.cat(parts, dim=0)
            seg_id = torch.cat(segs, dim=0)
            S = _cosine_sim_matrix_torch(Rq, Rf_all)
            colmax = S.max(dim=0).values
            scores_r = _segment_amax_1d(colmax, seg_id, n).to(torch.float32)

    return w_ent * scores_e + w_rel * scores_r  # (n,)


# ======================================================
# Coarse retrieval: GPU word-embedding cross-sim top-k
# ======================================================
@torch.no_grad()
def get_topk_word_embedding_batched_cross_sim_gpu(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    top_k: int = 3,
    question_batch_size: Optional[int] = None,          # None => auto
    questions_db_batch_size: Optional[int] = None,      # None => auto
    w_ent: float = 1.0,
    w_rel: float = 0.3,
    target: str = 'questions',
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_amp: bool = True,
    max_vram_utilization: float = 0.8,   # ↑ give more room
    target_min_flops: float = 5e7,       # ↑ fewer, larger launches
) -> Dict[int, List[Dict[str, Any]]]:

    device = device or _DEFAULT_DEVICE
    dtype  = dtype  or _DEFAULT_DTYPE
    amp_dtype = dtype if (dtype in (torch.float16, torch.bfloat16)) else torch.float16

    results: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(questions))}

    # Decide DB source (questions vs answers vs facts)
    if target == 'questions':
        q_groups_hist = codebook_main.get("questions_lst", [])[:-1]
        use_answers_db = not any(len(g) > 0 for g in q_groups_hist)
        if use_answers_db:
            groups_for_db = codebook_main.get("answers_lst", [])
            db_source = "answers"
        else:
            groups_for_db = q_groups_hist
            db_source = "questions"
    elif target == 'facts':
        groups_for_db = codebook_main.get("facts_lst", [])
        db_source = "facts"
    else:
        raise ValueError(f"Unknown target: {target}")

    # Flatten DB
    db_questions: List[List[int]] = []
    db_qi: List[int] = []
    db_qj: List[int] = []
    for qi, group in enumerate(groups_for_db):
        for qj, q_edges in enumerate(group):
            db_questions.append(q_edges)
            db_qi.append(qi)
            db_qj.append(qj)

    N_total = len(questions)
    M_total = len(db_questions)
    if N_total == 0 or M_total == 0:
        return results

    db_qi = np.asarray(db_qi, dtype=np.int32)
    db_qj = np.asarray(db_qj, dtype=np.int32)

    # Pre-extract DB (E,R) tensors on device
    db_ER: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
        _extract_entities_relations_from_run_torch(edge_run, codebook_main, device, dtype)
        for edge_run in db_questions
    ]

    # Auto-tune batch sizes if needed
    if question_batch_size is None or questions_db_batch_size is None:
        avg_ne, avg_nr, d_e, d_r = _avg_rows_and_dims(db_ER)
        q_bs, db_bs = _autotune_batches(
            N_total=N_total, M_total=M_total,
            avg_ne=avg_ne, avg_nr=avg_nr, d_e=d_e, d_r=d_r,
            device=device, use_amp=use_amp,
            max_vram_utilization=max_vram_utilization,
            target_min_flops=target_min_flops,
        )
        if question_batch_size is None:
            question_batch_size = q_bs
        if questions_db_batch_size is None:
            questions_db_batch_size = db_bs

    # Pre-build global index tensor to avoid tiny allocs in loop
    db_indices_all = torch.arange(M_total, device=device, dtype=torch.int32)

    for q_start in range(0, N_total, question_batch_size):
        q_end = min(q_start + question_batch_size, N_total)
        q_batch_idx = list(range(q_start, q_end))

        # Extract (E,R) for this query batch
        q_ER: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            _extract_entities_relations_from_run_torch(questions[i], codebook_main, device, dtype)
            for i in q_batch_idx
        ]

        # per-row running top-k
        best_scores: List[torch.Tensor] = [torch.empty(0, device=device, dtype=torch.float32) for _ in q_batch_idx]
        best_cols:   List[torch.Tensor] = [torch.empty(0, device=device, dtype=torch.int32)   for _ in q_batch_idx]

        for db_start in range(0, M_total, questions_db_batch_size):
            db_end = min(db_start + questions_db_batch_size, M_total)

            with _autocast_ctx(device, use_amp, amp_dtype):
                # build slice lists once
                Ef_slice = [pair[0] for pair in db_ER[db_start:db_end]]
                Rf_slice = [pair[1] for pair in db_ER[db_start:db_end]]

                for i, (Eq, Rq) in enumerate(q_ER):
                    # one vectorized pass for the entire slice
                    row_scores = _slice_entrel_maxpair_scores(Eq, Rq, Ef_slice, Rf_slice, w_ent, w_rel)  # (slice_len,)

                    k_local = min(top_k, row_scores.numel())
                    batch_scores, local_idx = torch.topk(row_scores, k_local, largest=True, sorted=True)
                    batch_cols = db_indices_all[db_start:db_end][local_idx]

                    merged_scores, merged_cols = _merge_topk_torch(
                        best_scores[i], best_cols[i], batch_scores, batch_cols, top_k
                    )
                    best_scores[i], best_cols[i] = merged_scores, merged_cols

        # write results for this batch
        for loc_i, gq_idx in enumerate(q_batch_idx):
            cols = best_cols[loc_i]; scs = best_scores[loc_i]
            if cols.numel() == 0:
                results[gq_idx] = []
                continue
            cols_cpu = cols.to("cpu").numpy()
            scs_cpu  = scs.to("cpu").float().numpy()
            entries = []
            for col, sc in zip(cols_cpu, scs_cpu):
                entries.append({
                    "score": float(sc),
                    "questions_index": int(db_qi[col]),
                    "question_index": int(db_qj[col]),
                    "db_source": db_source,
                })
            results[gq_idx] = entries

    return results


# ======================================================
# Reranker helpers
# ======================================================
def _triples_words(edge_run: List[int], codebook_main: Dict[str, Any]) -> List[List[str]]:
    return decode_question(edge_run, codebook_main, fmt='words')

def _embed_lines(lines: List[str], emb) -> np.ndarray:
    vecs = emb.embed_documents(lines)
    return np.asarray(vecs, dtype=np.float32)


# ======================================================
# Fine reranker scorer (GPU)
# ======================================================
@torch.no_grad()
def _score_query_vs_chunk_with_bonus_optimized_gpu(
    questions,
    coarse_topk,
    chunk_selected,                         # (qi, qj, src)
    q_edges_embeddings_lst_dict,            # i -> (nq_i,d)  (numpy or torch OK)
    questions_embeddings,                   # (N,d) L2-normalized or None  (torch or numpy)
    codebook_main,
    emb,
    q_triple_attention=None,                # i -> (nq_i,) sums to 1 (numpy or torch)
    top_t: int = 3,
    cov_tau: float = 0.45, cov_weight: float = 0.10,
    pair_tau: float = 0.55, pair_temp: float = 0.10, pair_weight: float = 0.20, pair_norm: str = "sqrt",
    distinct_weight: float = 0.10, distinct_tau: float = 0.50,
    whole_weight: float = 0.10,
    whole_gate_tau: float = 0.55,
    whole_gate_temp: float = 0.15,
    whole_len_norm: str = "sqrt_nc",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_amp: bool = True,
) -> float:

    device = device or _DEFAULT_DEVICE
    dtype  = dtype  or _DEFAULT_DTYPE
    amp_dtype = dtype if (dtype in (torch.float16, torch.bfloat16)) else torch.float16

    sel_qi, sel_qj, sel_src = chunk_selected

    # voters
    voters = []
    for i, _ in enumerate(questions):
        for it in coarse_topk.get(i, []):
            key = (int(it["questions_index"]), int(it["question_index"]), it.get("db_source", "questions"))
            if key == chunk_selected:
                voters.append(i); break
    if not voters:
        return 0.0

    # chunk edges by source
    if sel_src == "questions":
        groups = codebook_main["questions_lst"]
    elif sel_src == "answers":
        groups = codebook_main["answers_lst"]
    else:
        groups = codebook_main["facts_lst"]

    chunk_edges = groups[sel_qi][sel_qj]
    c_triples = _triples_words(chunk_edges, codebook_main)
    if not c_triples:
        return 0.0

    # Embed chunk triples (CPU -> GPU)
    c_lines = [f"{h} {r} {t}" for h, r, t in c_triples]
    C_np = _embed_lines(c_lines, emb)     # (nc,d)
    C = _to_tensor(C_np, device, dtype)

    S_list, A_list = [], []

    with _autocast_ctx(device, use_amp, amp_dtype):
        for i in voters:
            Q_np = q_edges_embeddings_lst_dict.get(i, np.zeros((0, C.shape[1]), dtype=np.float32))
            Q = _to_tensor(Q_np, device, dtype)
            if Q.numel() == 0:
                continue

            # (nq_i, nc)
            S = _cosine_sim_matrix_torch(Q, C)
            S_list.append(S)

            # attention
            if (q_triple_attention is not None) and (i in q_triple_attention) and (len(q_triple_attention[i]) > 0):
                a_np = np.asarray(q_triple_attention[i], dtype=np.float32).ravel()
                if a_np.size != S.shape[0]:
                    if a_np.size > S.shape[0]:
                        a_np = a_np[:S.shape[0]]
                    else:
                        pad = np.full((S.shape[0] - a_np.size,), 1.0 / max(1, S.shape[0]), dtype=np.float32)
                        a_np = np.concatenate([a_np, pad])
            else:
                a_np = np.full((S.shape[0],), 1.0 / max(1, S.shape[0]), dtype=np.float32)

            a = _to_tensor(a_np, device, dtype=torch.float32)  # keep attention in fp32
            a = a / (a.sum() + 1e-12)
            A_list.append(a)

        if not S_list:
            return 0.0

        def score_from_S(S: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            nq, nc = S.shape
            if nq == 0 or nc == 0:
                return torch.zeros((), device=S.device, dtype=torch.float32)

            S_attn = (a[:, None].to(S.dtype) * S)

            # adaptive top-t mean
            t_pairs = max(1, min(top_t, nq * nc))
            flat = S_attn.reshape(-1)
            if t_pairs == flat.numel():
                rel = flat.mean()
            else:
                vals, _ = torch.topk(flat, k=t_pairs, largest=True, sorted=False)
                rel = vals.mean()

            # coverage on per-row max
            best_per_q = S.max(dim=1).values
            covered = (best_per_q >= cov_tau).to(torch.float32)
            coverage = (covered * a).sum()

            # soft many-to-many
            soft_hits = torch.sigmoid((S_attn - pair_tau) / max(1e-6, pair_temp))
            good_pairs_soft = soft_hits.sum()
            if pair_norm == "sqrt":
                norm = torch.sqrt(torch.tensor(float(max(1, nq * nc)), device=S.device))
            elif pair_norm == "log":
                norm = torch.log1p(torch.tensor(float(max(1, nq * nc)), device=S.device))
            else:
                norm = torch.tensor(1.0, device=S.device)
            good_pairs_bonus = torch.log1p(good_pairs_soft / (norm + 1e-12))

            # distinct greedy
            distinct_bonus = torch.zeros((), device=S.device, dtype=torch.float32)
            if distinct_weight > 0.0:
                S_work = S_attn.clone()
                taken = 0
                for _ in range(int(min(nq, nc))):
                    val, idx = torch.max(S_work.view(-1), dim=0)
                    if val < distinct_tau:
                        break
                    r = (idx // nc).item()
                    c = (idx %  nc).item()
                    distinct_bonus = distinct_bonus + val.to(torch.float32)
                    S_work[r, :] = -float("inf")
                    S_work[:, c] = -float("inf")
                    taken += 1
                if taken > 0:
                    distinct_bonus = distinct_bonus / torch.sqrt(torch.tensor(float(taken), device=S.device))

            return (
                rel.to(torch.float32)
                + cov_weight * coverage.to(torch.float32)
                + pair_weight * good_pairs_bonus.to(torch.float32)
                + distinct_weight * distinct_bonus.to(torch.float32)
            )

        total = torch.zeros((), device=device, dtype=torch.float32)
        for S, a in zip(S_list, A_list):
            total = total + score_from_S(S, a)

        # whole question vs whole chunk bonus
        if (questions_embeddings is not None) and (whole_weight > 0.0):
            c_text = " <SEP> ".join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in c_triples])
            C_full_np = _embed_lines([c_text], emb)           # (1,d)
            vc = _to_tensor(C_full_np, device, dtype)
            vc = _l2norm_rows_torch(vc)

            if isinstance(questions_embeddings, torch.Tensor):
                vq_all_t = questions_embeddings.to(device=device, dtype=dtype)
            else:
                vq_all_t = _to_tensor(questions_embeddings, device, dtype)
            vq_all_t = _l2norm_rows_torch(vq_all_t)

            sims = []
            for i in voters:
                s = (vq_all_t[i:i+1, :] @ vc.T).squeeze()
                sims.append(s.to(torch.float32))
            if sims:
                s_whole = torch.stack(sims).mean()
                nc = len(c_triples)
                if whole_len_norm == "sqrt_nc":
                    len_factor = torch.sqrt(torch.tensor(float(nc), device=device)) + 1e-12
                elif whole_len_norm == "log_nc":
                    len_factor = torch.log1p(torch.tensor(float(nc), device=device)) + 1e-12
                else:
                    len_factor = torch.tensor(1.0, device=device)
                s_whole_adj = s_whole / len_factor
                whole_gate = torch.sigmoid((s_whole - whole_gate_tau) / max(1e-6, whole_gate_temp))
                total = total + whole_weight * (whole_gate.to(torch.float32) * s_whole_adj.to(torch.float32))

    return float(total.item())


# ======================================================
# Public reranker (GPU)
# ======================================================
@torch.no_grad()
def rerank_with_sentence_embeddings_score_with_coverage_optimized_gpu(
    questions,
    codebook_main,
    coarse_topk,
    emb,
    top_m: int = 1,
    use_attention: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_amp: bool = True,
) -> Dict[str, List[Any]]:

    device = device or _DEFAULT_DEVICE
    dtype  = dtype  or _DEFAULT_DTYPE

    # Build whole-question vectors (embed on CPU -> move to GPU)
    rows = []
    d = None
    for q_edges in questions:
        q_triples = _triples_words(q_edges, codebook_main)
        if not q_triples:
            rows.append(np.zeros((1, 1), dtype=np.float32))
            continue
        txt = " <SEP> ".join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in q_triples])
        v = _embed_lines([txt], emb)  # (1,d)
        d = v.shape[1] if d is None else d
        rows.append(v)
    if d is None:
        questions_embeddings = np.zeros((0, 0), dtype=np.float32)
    else:
        rows = [r if r.shape[1] == d else np.zeros((1, d), dtype=np.float32) for r in rows]
        V = np.vstack(rows).astype(np.float32)
        Vt = _to_tensor(V, device, dtype=torch.float32)   # keep fp32 for normalization
        questions_embeddings = _l2norm_rows_torch(Vt)     # torch (N,d)

    # Per-question per-triple embeddings + attention (Q stays numpy; moved on demand)
    q_edges_embeddings_lst_dict: Dict[int, np.ndarray] = {}
    q_triple_attention: Dict[int, np.ndarray] = {}

    for i, q_edges in enumerate(questions):
        q_triples = _triples_words(q_edges, codebook_main)
        q_lines = [f"{h} {r} {t}" for h, r, t in q_triples]
        if q_lines:
            Q = _embed_lines(q_lines, emb)  # numpy (nq_i,d)
        else:
            dcur = questions_embeddings.shape[1] if isinstance(questions_embeddings, torch.Tensor) else 0
            Q = np.zeros((0, dcur), dtype=np.float32)
        q_edges_embeddings_lst_dict[i] = Q

        if use_attention and Q.size > 0 and isinstance(questions_embeddings, torch.Tensor) and questions_embeddings.numel() > 0:
            vqi = questions_embeddings[i:i+1, :].to(dtype=torch.float32)  # (1,d)
            VQ  = _to_tensor(Q, device, dtype=torch.float32)
            VQ  = _l2norm_rows_torch(VQ)
            sims = (VQ @ vqi.T).squeeze(-1)  # (nq_i,)
            s = sims - sims.max()
            w = torch.exp(s).to(torch.float64)
            w = (w / (w.sum() + 1e-12)).to(torch.float32)
            q_triple_attention[i] = w.cpu().numpy()
        else:
            if Q.shape[0] > 0:
                q_triple_attention[i] = np.full((Q.shape[0],), 1.0 / Q.shape[0], dtype=np.float32)
            else:
                q_triple_attention[i] = np.zeros((0,), dtype=np.float32)

    # Unique candidate set (global)
    all_scored = {
        "score": [],
        "questions_index": [],
        "question_index": [],
        "db_source": [],
        "index_combo": [],
    }
    seen_global = set()
    for i, _ in enumerate(questions):
        for it in coarse_topk.get(i, []):
            key = (int(it["questions_index"]), int(it["question_index"]), it.get("db_source", "questions"))
            seen_global.add(key)

    for chunk_selected in seen_global:
        sc = _score_query_vs_chunk_with_bonus_optimized_gpu(
            questions=questions,
            coarse_topk=coarse_topk,
            chunk_selected=chunk_selected,
            q_edges_embeddings_lst_dict=q_edges_embeddings_lst_dict,
            questions_embeddings=questions_embeddings,  # torch
            codebook_main=codebook_main,
            emb=emb,
            q_triple_attention=q_triple_attention,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
        )
        qi, qj, src = chunk_selected
        all_scored["score"].append(float(sc))
        all_scored["questions_index"].append(qi)
        all_scored["question_index"].append(qj)
        all_scored["db_source"].append(src)
        all_scored["index_combo"].append([qi, qj])

    # take global top-m
    if all_scored["score"]:
        order = np.argsort(all_scored["score"])[::-1][:min(top_m, len(all_scored["score"]))]
        for k in list(all_scored.keys()):
            all_scored[k] = [all_scored[k][idx] for idx in order]

    return all_scored


# ======================================================
# Coarse+Fine wrapper (public API you call)
# ======================================================
def coarse_filter_optimized_gpu_ver(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    sentence_emb: HuggingFaceEmbeddings,
    top_k: int = 3,
    question_batch_size: Optional[int] = None,   # auto if None
    questions_db_batch_size: Optional[int] = None,  # auto if None
    top_m: int = 1,
    custom_linearizer = None,  # kept for API compatibility
    target: str = 'questions',
    w_ent: float = 1.0,
    w_rel: float = 0.3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_amp: bool = True,
) -> Dict[str, List[Any]]:

    # Step 1: word-embedding prefilter (entities/relations) → top_k
    coarse_top_k = get_topk_word_embedding_batched_cross_sim_gpu(
        questions=questions,
        codebook_main=codebook_main,
        top_k=top_k,
        question_batch_size=question_batch_size,
        questions_db_batch_size=questions_db_batch_size,
        w_ent=w_ent,
        w_rel=w_rel,
        target=target,
        device=device,
        dtype=dtype,
        use_amp=use_amp,
    )

    # Step 2: sentence-embedding rerank → top_m
    top_m_results = rerank_with_sentence_embeddings_score_with_coverage_optimized_gpu(
        questions=questions,
        codebook_main=codebook_main,
        coarse_topk=coarse_top_k,
        emb=sentence_emb,
        top_m=top_m,
        use_attention=False,
        device=device,
        dtype=dtype,
        use_amp=use_amp,
    )

    return top_m_results



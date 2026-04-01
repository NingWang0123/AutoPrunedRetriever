import logging
import time, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
from langchain_community.embeddings import HuggingFaceEmbeddings


# ─── Utilities ────────────────────────────────────────────────────

def _torch_l2norm_rows(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / (X.norm(dim=-1, keepdim=True) + eps)


def _stack_float32(x_list: Any, name: str) -> np.ndarray:
    """Robustly stack list-of-vectors to (N,d) float32 — fast paths."""
    if isinstance(x_list, np.ndarray):
        if x_list.dtype == np.float32:
            return x_list                       # zero copy
        return x_list.astype(np.float32, copy=False)
    if isinstance(x_list, (list, tuple)):
        n = len(x_list)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        first = x_list[0]
        # Fast path: homogeneous 1-D numpy arrays → bulk np.array()
        if isinstance(first, np.ndarray) and first.ndim == 1:
            try:
                out = np.array(x_list, dtype=np.float32)
                if out.ndim == 2:
                    return out
            except (ValueError, TypeError):
                pass
        # Fallback: per-element with pre-alloc
        arrs = [np.asarray(v, dtype=np.float32).ravel() for v in x_list]
        d = arrs[0].shape[0]
        result = np.empty((n, d), dtype=np.float32)
        for i, a in enumerate(arrs):
            result[i] = a if a.shape[0] == d else np.zeros(d, dtype=np.float32)
        return result
    return np.zeros((0, 0), dtype=np.float32)


def _build_ent_rel_ids_padded_from_runs(
    runs: List[List[int]],
    edge_matrix_np: np.ndarray,  # (n_edges, 3) [h,r,t]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = len(runs)
    n_em = edge_matrix_np.shape[0]

    ent_lists = []
    rel_lists = []
    max_e = 0
    max_r = 0

    for run in runs:
        if not run:
            ent_lists.append(np.empty(0, dtype=np.int64))
            rel_lists.append(np.empty(0, dtype=np.int64))
            continue
        eids = np.asarray(run, dtype=np.int64)
        valid = eids[(eids >= 0) & (eids < n_em)]
        if valid.size > 0:
            rows = edge_matrix_np[valid]          # (k, 3) vectorized lookup
            ents = np.unique(np.concatenate([rows[:, 0], rows[:, 2]]))
            rels = np.unique(rows[:, 1])
        else:
            ents = np.empty(0, dtype=np.int64)
            rels = np.empty(0, dtype=np.int64)
        ent_lists.append(ents)
        rel_lists.append(rels)
        if ents.size > max_e:
            max_e = ents.size
        if rels.size > max_r:
            max_r = rels.size

    max_e = max(max_e, 1)
    max_r = max(max_r, 1)

    ent_ids = np.full((N, max_e), -1, dtype=np.int64)
    rel_ids = np.full((N, max_r), -1, dtype=np.int64)
    ent_mask = np.zeros((N, max_e), dtype=bool)
    rel_mask = np.zeros((N, max_r), dtype=bool)

    for i, (es, rs) in enumerate(zip(ent_lists, rel_lists)):
        ne, nr = es.size, rs.size
        if ne:
            ent_ids[i, :ne] = es
            ent_mask[i, :ne] = True
        if nr:
            rel_ids[i, :nr] = rs
            rel_mask[i, :nr] = True

    return ent_ids, rel_ids, ent_mask, rel_mask


def _print_timings(timings: Dict[str, float]) -> None:
    total = timings.pop("total", sum(timings.values()))
    if total <= 0:
        return
    print(f"\n  === Retrieval Timings ===")
    print(f"  {'Stage':<40} {'Time (s)':>10}  {'% of total':>10}")
    print(f"  {'-'*62}")
    for k, v in timings.items():
        pct = v / total * 100
        print(f"  {k:<40} {v:>10.4f}s  {pct:>9.1f}%")
    print(f"  {'-'*62}")
    print(f"  {'TOTAL':<40} {total:>10.4f}s  {'100.0%':>10}\n")


# ─── TorchDBCache (with inverted index) ──────────────────────────

@dataclass
class TorchDBCache:
    device: torch.device
    dtype: torch.dtype

    e_emb: torch.Tensor
    r_emb: torch.Tensor
    edge_emb: torch.Tensor

    db_edges: torch.Tensor
    db_edge_mask: torch.Tensor
    db_nc: torch.Tensor

    db_full: torch.Tensor

    db_ent_ids: torch.Tensor
    db_rel_ids: torch.Tensor
    db_ent_mask: torch.Tensor
    db_rel_mask: torch.Tensor

    db_qi: np.ndarray
    db_qj: np.ndarray
    db_src: str
    key_to_flat: Dict[Tuple[int, int, str], int]

    # Inverted index: edge_id → chunk flat indices  (from bottom_up)
    inv_offsets: torch.Tensor   # (n_edges + 1,) long
    inv_chunks: torch.Tensor    # (total_postings,) long

    @staticmethod
    def build(
        codebook_main: Dict[str, Any],
        *,
        target: str = "questions",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ) -> "TorchDBCache":
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        # ---- base embeddings (zero-copy numpy → torch when possible)
        t0 = time.perf_counter()
        e_np = _stack_float32(codebook_main.get("e_embeddings"), "e_embeddings")
        r_np = _stack_float32(codebook_main.get("r_embeddings"), "r_embeddings")
        edge_np = _stack_float32(codebook_main.get("edge_matrix_embedding"), "edge_matrix_embedding")

        # from_numpy shares memory (no copy on CPU), then single .to(dev) for GPU transfer
        e_emb = torch.from_numpy(e_np).to(device=dev, dtype=torch.float32)
        r_emb = torch.from_numpy(r_np).to(device=dev, dtype=torch.float32)
        edge_emb = torch.from_numpy(edge_np).to(device=dev, dtype=torch.float32)
        timings["stack_base_embeddings"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if normalize:
            e_emb = _torch_l2norm_rows(e_emb)
            r_emb = _torch_l2norm_rows(r_emb)
            edge_emb = _torch_l2norm_rows(edge_emb)
        e_emb = e_emb.to(dtype)
        r_emb = r_emb.to(dtype)
        edge_emb = edge_emb.to(dtype)
        timings["normalize_base_embeddings"] = time.perf_counter() - t0

        # ---- DB selection
        t0 = time.perf_counter()
        if target == "questions":
            q_groups_hist = codebook_main.get("questions_lst", [])[:-1]
            use_answers_db = not any(len(g) > 0 for g in q_groups_hist)
            if use_answers_db:
                groups_for_db = codebook_main.get("answers_lst", [])
                groups_emb = codebook_main.get("answers_lst_embedding", [])
                db_src = "answers"
            else:
                groups_for_db = q_groups_hist
                groups_emb = codebook_main.get("questions_lst_embedding", [])[:-1]
                db_src = "questions"
            print(f'db_src is {db_src}')
        elif target == "facts":
            groups_for_db = codebook_main.get("facts_lst", [])
            groups_emb = codebook_main.get("facts_lst_embedding", [])
            db_src = "facts"
        else:
            raise ValueError("target must be 'questions' or 'facts'")
        timings["db_selection"] = time.perf_counter() - t0

        # ---- flatten DB (group-level vectorized)
        t0 = time.perf_counter()
        db_runs: List[List[int]] = []
        db_qi: List[int] = []
        db_qj: List[int] = []
        key_to_flat: Dict[Tuple[int, int, str], int] = {}
        emb_blocks: List[np.ndarray] = []   # collect 2D blocks for single vstack

        if not groups_emb:
            raise KeyError(f"{db_src}_lst_embedding missing in codebook_main")

        # embedding dim fallback from edge_matrix_embedding
        _fallback_d = int(edge_emb.shape[1]) if edge_emb.ndim == 2 and edge_emb.shape[1] > 0 else 0

        for qi, group in enumerate(groups_for_db):
            n_chunks = len(group)
            base_flat = len(db_runs)
            for qj, run in enumerate(group):
                db_runs.append([int(x) for x in run])
                db_qi.append(int(qi))
                db_qj.append(int(qj))
                key_to_flat[(int(qi), int(qj), db_src)] = base_flat + qj

            # Collect group embedding as one 2D block
            if qi < len(groups_emb):
                g_emb = groups_emb[qi]
                g_arr = np.asarray(g_emb, dtype=np.float32) if not isinstance(g_emb, np.ndarray) else g_emb
                # Guard: skip (0,0) or empty arrays — emit zeros with correct dim
                if g_arr.ndim == 2 and g_arr.shape[0] > 0 and g_arr.shape[1] > 0:
                    g_arr = g_arr if g_arr.dtype == np.float32 else g_arr.astype(np.float32)
                    # Fix: ensure embedding rows match actual chunk count
                    if g_arr.shape[0] != n_chunks:
                        if g_arr.shape[0] > n_chunks:
                            g_arr = g_arr[:n_chunks]
                        else:
                            pad = np.zeros((n_chunks - g_arr.shape[0], g_arr.shape[1]), dtype=np.float32)
                            g_arr = np.vstack([g_arr, pad])
                    emb_blocks.append(g_arr)
                else:
                    # DIAGNOSTIC: log why this group was rejected
                    if n_chunks > 0:
                        print(f"[TorchDBCache] EMB REJECT group[{qi}]: n_chunks={n_chunks}, g_arr.ndim={g_arr.ndim}, g_arr.shape={g_arr.shape}, type(g_emb)={type(g_emb).__name__}, len(g_emb)={len(g_emb) if hasattr(g_emb, '__len__') else 'N/A'}")
                    if n_chunks > 0 and _fallback_d > 0:
                        emb_blocks.append(np.zeros((n_chunks, _fallback_d), dtype=np.float32))
                    # else: 0 chunks, nothing to append
            elif n_chunks > 0 and _fallback_d > 0:
                emb_blocks.append(np.zeros((n_chunks, _fallback_d), dtype=np.float32))

        M = len(db_runs)
        d = int(edge_emb.shape[1]) if edge_emb.ndim == 2 else 0
        n_edges = edge_emb.shape[0]
        timings["flatten_db"] = time.perf_counter() - t0

        if M == 0:
            empty = torch.zeros((0, 1), device=dev, dtype=torch.long)
            emptyb = torch.zeros((0, 1), device=dev, dtype=torch.bool)
            emptyf = torch.zeros((0, d), device=dev, dtype=dtype)
            empty1 = torch.zeros((0,), device=dev, dtype=torch.long)
            timings["total"] = time.perf_counter() - t_total
            _print_timings(timings)
            return TorchDBCache(
                device=dev, dtype=dtype,
                e_emb=e_emb, r_emb=r_emb, edge_emb=edge_emb,
                db_edges=empty, db_edge_mask=emptyb, db_nc=empty1,
                db_full=emptyf,
                db_ent_ids=empty, db_rel_ids=empty,
                db_ent_mask=emptyb, db_rel_mask=emptyb,
                db_qi=np.zeros((0,), dtype=np.int32),
                db_qj=np.zeros((0,), dtype=np.int32),
                db_src=db_src, key_to_flat={},
                inv_offsets=torch.zeros((n_edges + 1,), device=dev, dtype=torch.long),
                inv_chunks=torch.zeros((0,), device=dev, dtype=torch.long),
            )

        # ---- padded edge ids
        t0 = time.perf_counter()
        db_nc_np = np.array([len(r) for r in db_runs], dtype=np.int64)
        max_nc = int(db_nc_np.max()) if M > 0 else 0
        db_edges_np = np.full((M, max_nc), -1, dtype=np.int64)
        for i, run in enumerate(db_runs):
            L = len(run)
            if L > 0:
                db_edges_np[i, :L] = run
        db_edges = torch.from_numpy(db_edges_np).to(device=dev, dtype=torch.long)
        db_nc = torch.from_numpy(db_nc_np).to(device=dev, dtype=torch.long)
        db_mask = torch.from_numpy(np.arange(max_nc)[None, :] < db_nc_np[:, None]).to(device=dev, dtype=torch.bool)
        timings["build_padded_edge_ids"] = time.perf_counter() - t0

        # ---- chunk full embeddings (single vstack → single GPU transfer)
        t0 = time.perf_counter()
        if emb_blocks:
            db_full_np = np.vstack(emb_blocks)
            if db_full_np.shape[0] != M:
                print(f"[TorchDBCache] WARNING: emb_blocks rows={db_full_np.shape[0]} != M={M}, truncating/padding")
                if db_full_np.shape[0] > M:
                    db_full_np = db_full_np[:M]
                else:
                    pad = np.zeros((M - db_full_np.shape[0], db_full_np.shape[1]), dtype=np.float32)
                    db_full_np = np.vstack([db_full_np, pad])
            if db_full_np.dtype != np.float32:
                db_full_np = db_full_np.astype(np.float32, copy=False)
            del emb_blocks
            db_full = torch.from_numpy(db_full_np).to(device=dev, dtype=torch.float32)
            del db_full_np
        else:
            db_full = torch.zeros((M, d), device=dev, dtype=torch.float32)
        if normalize and db_full.numel() > 0:
            db_full = _torch_l2norm_rows(db_full)
        db_full = db_full.to(dtype)
        timings["build_chunk_full_embeddings"] = time.perf_counter() - t0

        # ---- coarse-stage ent/rel ids
        t0 = time.perf_counter()
        edge_matrix_np = np.asarray(codebook_main["edge_matrix"], dtype=np.int64)
        db_ent_ids_np, db_rel_ids_np, db_ent_mask_np, db_rel_mask_np = _build_ent_rel_ids_padded_from_runs(
            db_runs, edge_matrix_np
        )
        db_ent_ids = torch.as_tensor(db_ent_ids_np, device=dev, dtype=torch.long)
        db_rel_ids = torch.as_tensor(db_rel_ids_np, device=dev, dtype=torch.long)
        db_ent_mask = torch.as_tensor(db_ent_mask_np, device=dev, dtype=torch.bool)
        db_rel_mask = torch.as_tensor(db_rel_mask_np, device=dev, dtype=torch.bool)
        timings["build_ent_rel_ids"] = time.perf_counter() - t0

        # ---- BUILD INVERTED INDEX: edge_id → chunk flat indices (numpy CSR)
        t0 = time.perf_counter()
        all_eids = []
        all_cids = []
        for chunk_idx, run in enumerate(db_runs):
            arr = np.asarray(run, dtype=np.int64)
            all_eids.append(arr)
            all_cids.append(np.full(arr.shape[0], chunk_idx, dtype=np.int64))

        if all_eids:
            flat_eids = np.concatenate(all_eids)
            flat_cids = np.concatenate(all_cids)
            # Sort by edge_id → CSR offsets via searchsorted
            order = np.argsort(flat_eids, kind='mergesort')  # stable: preserves chunk order
            flat_eids = flat_eids[order]
            flat_cids = flat_cids[order]
            offsets = np.searchsorted(flat_eids, np.arange(n_edges + 1, dtype=np.int64), side='left')
            chunks_flat = flat_cids
        else:
            offsets = np.zeros((n_edges + 1,), dtype=np.int64)
            chunks_flat = np.empty((0,), dtype=np.int64)

        inv_offsets = torch.as_tensor(offsets, device=dev, dtype=torch.long)
        inv_chunks = torch.as_tensor(chunks_flat, device=dev, dtype=torch.long)
        timings["build_inverted_index"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        _print_timings(timings)

        return TorchDBCache(
            device=dev, dtype=dtype,
            e_emb=e_emb, r_emb=r_emb, edge_emb=edge_emb,
            db_edges=db_edges, db_edge_mask=db_mask, db_nc=db_nc,
            db_full=db_full,
            db_ent_ids=db_ent_ids, db_rel_ids=db_rel_ids,
            db_ent_mask=db_ent_mask, db_rel_mask=db_rel_mask,
            db_qi=np.asarray(db_qi, dtype=np.int32),
            db_qj=np.asarray(db_qj, dtype=np.int32),
            db_src=db_src, key_to_flat=key_to_flat,
            inv_offsets=inv_offsets,
            inv_chunks=inv_chunks,
        )


# ─── Reciprocal Rank Fusion ──────────────────────────────────────

def reciprocal_rank_fusion(
    rank_lists: List[np.ndarray],
    k: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse multiple ranked lists of candidate indices into one.
    k=60 is the universal constant (Cormack et al. 2009).
    Returns (sorted_indices, rrf_scores) both 1-D, descending by score.
    """
    scores: Dict[int, float] = {}
    for ranked in rank_lists:
        for rank_pos, idx in enumerate(ranked):
            idx = int(idx)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank_pos + 1)
    if not scores:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    items = sorted(scores.items(), key=lambda x: -x[1])
    indices = np.array([x[0] for x in items], dtype=np.int64)
    vals = np.array([x[1] for x in items], dtype=np.float64)
    return indices, vals


# ═══════════════════════════════════════════════════════════════════
# RANKING SIGNALS
# ═══════════════════════════════════════════════════════════════════

# ── Signal 1: Semantic cosine (global, from non_param) ──

@torch.inference_mode()
def _rank_semantic(
    q_full: torch.Tensor,   # (N, d) normalized
    cache: TorchDBCache,
) -> torch.Tensor:
    """Sentence-level cosine. Returns (M,) scores."""
    sims = q_full.float() @ cache.db_full.float().T   # (N, M)
    agg, _ = sims.max(dim=0)                           # (M,)
    return agg


# ── Signal 2: Entity maxpair cosine (global, from non_param) ──

@torch.inference_mode()
def _rank_entity_maxpair(
    q_ent_emb: torch.Tensor,    # (N, max_qe, d)
    q_ent_mask: torch.Tensor,   # (N, max_qe)
    cache: TorchDBCache,
    block: int = 512,
) -> torch.Tensor:
    """Returns (M,) scores."""
    M = cache.db_ent_ids.shape[0]
    N = q_ent_emb.shape[0]
    dev = cache.device
    scores = torch.zeros((N, M), device=dev, dtype=torch.float32)

    for s in range(0, M, block):
        e = min(s + block, M)
        b_emb = cache.e_emb[cache.db_ent_ids[s:e].clamp(0, cache.e_emb.shape[0] - 1)].float() * cache.db_ent_mask[s:e].unsqueeze(-1)
        S = torch.einsum("nqd,bkd->nbqk", q_ent_emb, b_emb)
        valid = q_ent_mask[:, None, :, None] & cache.db_ent_mask[None, s:e, None, :]
        S.masked_fill_(~valid, float("-inf"))
        m = S.max(dim=-1).values.max(dim=-1).values
        scores[:, s:e] = torch.where(torch.isfinite(m), m, torch.zeros_like(m))

    agg, _ = scores.max(dim=0)
    return agg


# ── Signal 3: Relation maxpair cosine (global, from non_param) ──

@torch.inference_mode()
def _rank_relation_maxpair(
    q_rel_emb: torch.Tensor,    # (N, max_qr, d)
    q_rel_mask: torch.Tensor,   # (N, max_qr)
    cache: TorchDBCache,
    block: int = 512,
) -> torch.Tensor:
    """Returns (M,) scores."""
    M = cache.db_rel_ids.shape[0]
    N = q_rel_emb.shape[0]
    dev = cache.device
    scores = torch.zeros((N, M), device=dev, dtype=torch.float32)

    for s in range(0, M, block):
        e = min(s + block, M)
        b_emb = cache.r_emb[cache.db_rel_ids[s:e].clamp(0, cache.r_emb.shape[0] - 1)].float() * cache.db_rel_mask[s:e].unsqueeze(-1)
        S = torch.einsum("nqd,bkd->nbqk", q_rel_emb, b_emb)
        valid = q_rel_mask[:, None, :, None] & cache.db_rel_mask[None, s:e, None, :]
        S.masked_fill_(~valid, float("-inf"))
        m = S.max(dim=-1).values.max(dim=-1).values
        scores[:, s:e] = torch.where(torch.isfinite(m), m, torch.zeros_like(m))

    agg, _ = scores.max(dim=0)
    return agg


# ── Signal 4: Inverted-index structural (per-query, from bottom_up) ──

@torch.inference_mode()
def _score_structural_per_query(
    q_edge_ids: List[int],
    cache: TorchDBCache,
    top_edges_per_q_edge: int,
) -> np.ndarray:
    """
    For one query, score every chunk via inverted index.
    Returns (M,) float32 scores.

    Algorithm:
        1. For each query edge, cosine to ALL vocabulary edges
        2. Top-K most similar vocabulary edges
        3. Inverted index lookup → which chunks contain those edges
        4. Accumulate similarity scores per chunk
        5. Normalize by sqrt(chunk_length)
    """
    dev = cache.device
    M = int(cache.db_nc.shape[0])
    n_edges_vocab = cache.edge_emb.shape[0]

    if len(q_edge_ids) == 0 or M == 0:
        return np.zeros((M,), dtype=np.float32)

    q_eids_t = torch.as_tensor(q_edge_ids, device=dev, dtype=torch.long)
    # Filter out edge IDs that exceed the cache's vocabulary (newly-added question edges
    # may have IDs beyond what the cache was built with)
    valid_mask = (q_eids_t >= 0) & (q_eids_t < n_edges_vocab)
    q_eids_t = q_eids_t[valid_mask]
    if q_eids_t.numel() == 0:
        return np.zeros((M,), dtype=np.float32)
    q_embs = cache.edge_emb[q_eids_t].float()
    q_embs = q_embs / (q_embs.norm(dim=1, keepdim=True) + 1e-12)

    nq = q_embs.shape[0]
    K = min(top_edges_per_q_edge, n_edges_vocab)

    # Batched cosine: (nq, n_edges_vocab)
    all_sims = q_embs @ cache.edge_emb.float().T

    # Top-K per query edge
    topk_vals, topk_ids = torch.topk(all_sims, K, dim=1)
    topk_vals = topk_vals.cpu().numpy()
    topk_ids = topk_ids.cpu().numpy()

    inv_offsets_np = cache.inv_offsets.cpu().numpy()
    inv_chunks_np = cache.inv_chunks.cpu().numpy()

    chunk_scores = np.zeros((M,), dtype=np.float64)

    for qi in range(nq):
        for rank_j in range(K):
            sim_val = float(topk_vals[qi, rank_j])
            if sim_val <= 0.0:
                break
            eid = int(topk_ids[qi, rank_j])
            start = int(inv_offsets_np[eid])
            end = int(inv_offsets_np[eid + 1])
            if start == end:
                continue
            chunk_indices = inv_chunks_np[start:end]
            chunk_scores[chunk_indices] += sim_val

    # Normalize by sqrt(chunk_length)
    db_nc_np = cache.db_nc.cpu().numpy().astype(np.float64)
    norm = np.sqrt(np.maximum(db_nc_np, 1.0))
    chunk_scores /= norm

    return chunk_scores.astype(np.float32)


# ── Signal 5: Edge-fine similarity (pool only, from non_param) ──

@torch.inference_mode()
def _rank_edge_fine(
    Qe: torch.Tensor,           # (N, max_nq, d)
    q_mask: torch.Tensor,       # (N, max_nq)
    cache: TorchDBCache,
    pool: torch.Tensor,          # (P,) subset of flat indices
    block: int = 256,
) -> torch.Tensor:
    """
    Edge-level fine similarity on candidate pool.
    For each (query, chunk): all edge-to-edge cosines → top-sqrt(K) average.
    Returns (P,) scores.
    """
    N = Qe.shape[0]
    P = pool.shape[0]
    dev = cache.device
    scores = torch.zeros((N, P), device=dev, dtype=torch.float32)

    for s in range(0, P, block):
        e = min(s + block, P)
        flats = pool[s:e]
        B = flats.numel()
        Ce = cache.edge_emb[cache.db_edges[flats].clamp(0, cache.edge_emb.shape[0] - 1)].float() * cache.db_edge_mask[flats].unsqueeze(-1)
        S = torch.einsum("nqd,bkd->nbqk", Qe, Ce)
        valid = q_mask[:, None, :, None] & cache.db_edge_mask[flats][None, :, None, :]
        S.masked_fill_(~valid, float("-inf"))

        flat_s = S.reshape(N, B, -1)
        K = flat_s.shape[-1]
        if K == 0:
            continue
        k_take = max(1, int(K ** 0.5))
        topv = torch.topk(flat_s, min(k_take, K), dim=-1).values
        fin = torch.isfinite(topv)
        scores[:, s:e] = topv.masked_fill(~fin, 0.0).sum(-1) / fin.sum(-1).clamp_min(1)

    agg, _ = scores.max(dim=0)
    return agg


# ── Signal 6: Coverage (pool only, from non_param) ──

@torch.inference_mode()
def _rank_coverage(
    Qe: torch.Tensor,           # (N, max_nq, d)
    q_mask: torch.Tensor,       # (N, max_nq)
    cache: TorchDBCache,
    pool: torch.Tensor,          # (P,) subset of flat indices
    block: int = 256,
) -> torch.Tensor:
    """
    Coverage: for each query edge, best cosine match in the chunk.
    Average over query edges. Returns (P,) scores.
    """
    N = Qe.shape[0]
    P = pool.shape[0]
    dev = cache.device
    scores = torch.zeros((N, P), device=dev, dtype=torch.float32)
    nq_per = q_mask.sum(dim=1, keepdim=True).clamp_min(1).float()

    for s in range(0, P, block):
        e = min(s + block, P)
        flats = pool[s:e]
        Ce = cache.edge_emb[cache.db_edges[flats].clamp(0, cache.edge_emb.shape[0] - 1)].float() * cache.db_edge_mask[flats].unsqueeze(-1)
        S = torch.einsum("nqd,bkd->nbqk", Qe, Ce)
        valid = q_mask[:, None, :, None] & cache.db_edge_mask[flats][None, :, None, :]
        S.masked_fill_(~valid, float("-inf"))

        best_per_q = S.max(dim=-1).values   # (N, B, max_nq)
        best_per_q = torch.where(q_mask[:, None, :], best_per_q, torch.zeros_like(best_per_q))
        scores[:, s:e] = best_per_q.sum(dim=-1) / nq_per

    agg, _ = scores.max(dim=0)
    return agg


# ═══════════════════════════════════════════════════════════════════
# Query embedding helpers
# ═══════════════════════════════════════════════════════════════════

def compute_query_full_embeddings_tagged(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    sentence_emb,
) -> List[np.ndarray]:
    E = codebook_main["e"]
    R = codebook_main["r"]
    EM = codebook_main["edge_matrix"]
    texts = []
    for run in questions:
        parts = []
        for eid in run:
            eid = int(eid)
            h, r, t = EM[eid]
            parts.append(f"[H]{E[h]} [R]{R[r]} [T]{E[t]}")
        texts.append(" <SEP> ".join(parts) if parts else " ")
    return sentence_emb.embed_documents(texts)


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def coarse_filter_torch(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    sentence_emb: HuggingFaceEmbeddings,
    top_k: int = 200,
    top_m: int = 20,
    target: str = "questions",
    w_ent: float = 1.0,       # ignored (kept for signature compat)
    w_rel: float = 0.3,       # ignored (kept for signature compat)
    prebuilt_cache: "TorchDBCache | None" = None,
) -> Dict[str, List]:
    """
    Hybrid retriever combining layered multi-signal + inverted-index approaches.

    Architecture:
        Layer 1 -- DUAL-TRACK pool selection (neither track can veto the other):
            Track A: semantic + entity + relation -> RRF -> pool_a
            Track B: N per-query structural (inverted index) -> RRF -> pool_b
            Pool = union(pool_a, pool_b)

        Layer 2 -- Fine-grained scoring on pool:
            edge-fine | coverage

        Final -- Balanced 4-way RRF:
            fused_global | fused_structural | edge-fine | coverage
            Each signal gets exactly one vote -- no dilution.

    No tunable weights, thresholds, temperatures, or taus.
    Only RRF k=60 (universal constant).
    """
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    N = len(questions)
    empty_out = {"score": [], "questions_index": [], "question_index": [],
                 "db_source": [], "index_combo": []}
    if N == 0:
        return empty_out

    # ── Build cache (includes inverted index) ──
    t0 = time.perf_counter()
    if prebuilt_cache is not None:
        cache = prebuilt_cache
        timings["build_cache (reused)"] = time.perf_counter() - t0
    else:
        cache = TorchDBCache.build(codebook_main, target=target, device="cuda", dtype=torch.float32)
        timings["build_cache"] = time.perf_counter() - t0

    M = int(cache.db_full.shape[0])
    M_edges = int(cache.db_edges.shape[0])
    M_mask = int(cache.db_edge_mask.shape[0])
    if M != M_edges or M != M_mask:
        raise RuntimeError(
            f"[Hybrid] FATAL: tensor size mismatch! db_full={M}, db_edges={M_edges}, db_edge_mask={M_mask}. "
            f"questions_lst and questions_lst_embedding are out of sync."
        )
    if M == 0:
        return empty_out

    dev = cache.device
    n_edges_vocab = cache.edge_emb.shape[0]

    # ── Prepare query embeddings ──

    # Sentence embeddings for semantic signal
    t0 = time.perf_counter()
    questions_full_emb = compute_query_full_embeddings_tagged(questions, codebook_main, sentence_emb)
    q_full = torch.as_tensor(np.asarray(questions_full_emb, dtype=np.float32), device=dev)
    q_full = _torch_l2norm_rows(q_full)
    timings["query_full_emb"] = time.perf_counter() - t0

    # Structured embeddings for entity/relation/edge signals
    t0 = time.perf_counter()
    edge_matrix_np = np.asarray(codebook_main["edge_matrix"], dtype=np.int64)
    q_ent_ids_np, q_rel_ids_np, q_ent_mask_np, q_rel_mask_np = _build_ent_rel_ids_padded_from_runs(
        questions, edge_matrix_np
    )
    q_ent_ids = torch.as_tensor(q_ent_ids_np, device=dev, dtype=torch.long)
    q_rel_ids = torch.as_tensor(q_rel_ids_np, device=dev, dtype=torch.long)
    q_ent_mask = torch.as_tensor(q_ent_mask_np, device=dev, dtype=torch.bool)
    q_rel_mask = torch.as_tensor(q_rel_mask_np, device=dev, dtype=torch.bool)

    q_ent_emb = cache.e_emb[q_ent_ids.clamp(0, cache.e_emb.shape[0] - 1)].float() * q_ent_mask.unsqueeze(-1)
    q_rel_emb = cache.r_emb[q_rel_ids.clamp(0, cache.r_emb.shape[0] - 1)].float() * q_rel_mask.unsqueeze(-1)

    # Query edge embeddings (for Layer 2)
    max_nq = max((len(q) for q in questions), default=0)
    q_edges = torch.full((N, max_nq), -1, device=dev, dtype=torch.long)
    q_mask = torch.zeros((N, max_nq), device=dev, dtype=torch.bool)
    for i, run in enumerate(questions):
        L = len(run)
        if L:
            q_edges[i, :L] = torch.as_tensor([int(x) for x in run], device=dev, dtype=torch.long)
            q_mask[i, :L] = True
    Qe = cache.edge_emb[q_edges.clamp(0, n_edges_vocab - 1)].float() * q_mask.unsqueeze(-1)
    timings["query_struct_emb"] = time.perf_counter() - t0

    # ══════════════════════════════════════════════════════════════
    # LAYER 1: Global signals + per-query structural (full DB)
    # ══════════════════════════════════════════════════════════════

    # --- 3 global signals (from non_param) ---

    t0 = time.perf_counter()
    sem_scores = _rank_semantic(q_full, cache)                         # (M,)
    timings["L1_semantic"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ent_scores = _rank_entity_maxpair(q_ent_emb, q_ent_mask, cache)   # (M,)
    timings["L1_entity"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rel_scores = _rank_relation_maxpair(q_rel_emb, q_rel_mask, cache) # (M,)
    timings["L1_relation"] = time.perf_counter() - t0

    rank_sem = torch.argsort(sem_scores, descending=True).cpu().numpy()
    rank_ent = torch.argsort(ent_scores, descending=True).cpu().numpy()
    rank_rel = torch.argsort(rel_scores, descending=True).cpu().numpy()

    # --- N per-query structural signals (from bottom_up) ---

    t0 = time.perf_counter()
    top_edges_per_q_edge = max(10, min(500, int(n_edges_vocab ** 0.5)))

    structural_rankings: List[np.ndarray] = []
    for i, q_run in enumerate(questions):
        q_edge_ids = [int(x) for x in q_run]
        scores_i = _score_structural_per_query(q_edge_ids, cache, top_edges_per_q_edge)
        ranking_i = np.argsort(-scores_i)
        structural_rankings.append(ranking_i)

    timings["L1_structural_invindex"] = time.perf_counter() - t0

    # --- Layer 1 DUAL-TRACK pool selection ---
    # Track A: global signals (semantic + entity + relation)
    # Track B: per-query structural signals (inverted index)
    # Pool = union(Track A top, Track B top)
    # This guarantees chunks found by EITHER approach survive to Layer 2.

    t0 = time.perf_counter()
    pool_size = min(M, max(top_m * 10, top_k))

    # Track A: fuse 3 global signals → independent pool
    fused_global, _ = reciprocal_rank_fusion([rank_sem, rank_ent, rank_rel])
    pool_a = set(int(x) for x in fused_global[:pool_size])

    # Track B: fuse N structural signals → independent pool
    fused_struct, _ = reciprocal_rank_fusion(structural_rankings)
    pool_b = set(int(x) for x in fused_struct[:pool_size])

    # Union: neither track can veto the other
    pool_union = sorted(x for x in (pool_a | pool_b) if x < M)
    pool_indices = np.array(pool_union, dtype=np.int64) if pool_union else np.array([], dtype=np.int64)
    pool_t = torch.as_tensor(pool_indices, device=dev, dtype=torch.long)
    timings["L1_dual_track_pool"] = time.perf_counter() - t0

    print(f"[Hybrid] Layer 1: {M} chunks → Track A={len(pool_a)}, Track B={len(pool_b)}, overlap={len(pool_a & pool_b)}, union={len(pool_union)}")

    # ══════════════════════════════════════════════════════════════
    # LAYER 2: Fine-grained signals on pool only
    # ══════════════════════════════════════════════════════════════

    if pool_t.numel() == 0:
        # No valid pool candidates — fall back to global + structural signals only
        t0 = time.perf_counter()
        final_indices, final_scores = reciprocal_rank_fusion([fused_global, fused_struct])
        timings["L2_edge_fine"] = 0.0
        timings["L2_coverage"] = 0.0
        timings["final_rrf_fusion"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_total
        _print_timings(timings)
        print(f"[Hybrid] Final: top {min(int(top_m), len(final_indices))} from {M} candidates (empty pool fallback)")

        m = min(int(top_m), len(final_indices))
        out = {"score": [], "questions_index": [], "question_index": [],
               "db_source": [], "index_combo": [],
               "support_edges": [], "support_entities": []}
        edge_matrix_np_fb = np.asarray(codebook_main["edge_matrix"], dtype=np.int64)
        for rank_pos in range(m):
            flat_idx = int(final_indices[rank_pos])
            if flat_idx >= len(cache.db_qi):
                continue
            qi = int(cache.db_qi[flat_idx])
            qj = int(cache.db_qj[flat_idx])
            out["score"].append(float(final_scores[rank_pos]))
            out["questions_index"].append(qi)
            out["question_index"].append(qj)
            out["db_source"].append(cache.db_src)
            out["index_combo"].append([qi, qj])
            out["support_edges"].append([])
            out["support_entities"].append([])
        return out

    t0 = time.perf_counter()
    fine_scores = _rank_edge_fine(Qe, q_mask, cache, pool_t)          # (P,)
    timings["L2_edge_fine"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cov_scores = _rank_coverage(Qe, q_mask, cache, pool_t)            # (P,)
    timings["L2_coverage"] = time.perf_counter() - t0

    # Rankings within pool → map back to flat DB indices
    rank_fine_local = torch.argsort(fine_scores, descending=True).cpu().numpy()
    rank_cov_local = torch.argsort(cov_scores, descending=True).cpu().numpy()
    rank_fine_global = pool_indices[rank_fine_local]
    rank_cov_global = pool_indices[rank_cov_local]

    # ══════════════════════════════════════════════════════════════
    # FINAL: Balanced 6-way RRF fusion
    # ══════════════════════════════════════════════════════════════
    #   3 global (semantic, entity, relation)  — full M
    #   1 structural (pre-fused from N per-query) — full M
    #   2 Layer 2 (edge-fine, coverage) — pool only
    # Each signal gets exactly one vote — no dilution.

    t0 = time.perf_counter()
    final_indices, final_scores = reciprocal_rank_fusion([
        fused_global,        # full M — fused semantic+entity+relation
        fused_struct,        # full M — fused N structural per-query
        rank_fine_global,    # pool only — edge-level detail
        rank_cov_global,     # pool only — coverage
    ])
    timings["final_rrf_fusion"] = time.perf_counter() - t0

    # ── Build output ──
    m = min(int(top_m), len(final_indices))
    out = {"score": [], "questions_index": [], "question_index": [],
           "db_source": [], "index_combo": [],
           "support_edges": [], "support_entities": []}

    edge_matrix_np_out = np.asarray(codebook_main["edge_matrix"], dtype=np.int64)
    n_db = len(cache.db_qi)
    for rank_pos in range(m):
        flat_idx = int(final_indices[rank_pos])
        if flat_idx >= n_db:
            continue
        qi = int(cache.db_qi[flat_idx])
        qj = int(cache.db_qj[flat_idx])
        out["score"].append(float(final_scores[rank_pos]))
        out["questions_index"].append(qi)
        out["question_index"].append(qj)
        out["db_source"].append(cache.db_src)
        out["index_combo"].append([qi, qj])

        # Collect support edges and entities for this candidate
        edges_np = cache.db_edges[flat_idx].cpu().numpy()
        mask_np = cache.db_edge_mask[flat_idx].cpu().numpy()
        valid_eids = [int(e) for e, v in zip(edges_np, mask_np) if v and int(e) >= 0]
        ents: set = set()
        for eid in valid_eids:
            if eid < len(edge_matrix_np_out):
                h, _r, t = edge_matrix_np_out[eid]
                ents.add(int(h)); ents.add(int(t))
        out["support_edges"].append(valid_eids)
        out["support_entities"].append(sorted(ents))

    timings["total"] = time.perf_counter() - t_total
    _print_timings(timings)
    print(f"[Hybrid] Final: top {m} from {len(final_indices)} candidates (4-way balanced RRF: global + structural + edge-fine + coverage)")

    return out
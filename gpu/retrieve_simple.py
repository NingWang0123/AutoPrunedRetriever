"""
Simple retrieval: direct coarse_filter_torch without structure-aware
multi-hop logic (no bridge extraction, continuation reranking, or group solving).

Drop-in replacement for retrieve_top_m_by_structure — same function signature
and return format, but uses only the base coarse_filter_torch from
retrieve_gpu_cached_combined.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from retrieve_gpu_cached_combined import (
    TorchDBCache, coarse_filter_torch, _print_timings,
)


def retrieve_top_m_by_structure_simple(
    question_edge_runs: List[List[int]],
    codebook_main: Dict[str, Any],
    sentence_emb,
    top_k: int = 200,
    top_m: int = 20,
    target: str = 'facts',
    n_ents_before: int = 0,
    prebuilt_cache: Optional[TorchDBCache] = None,
    compressed_runs: Optional[List[List[int]]] = None,
    max_hops: int = 3,
    parser_groups: Optional[List[List[str]]] = None,
    **_kwargs,
) -> Dict[str, list]:
    """
    Simple retrieval using coarse_filter_torch directly.

    Accepts the same arguments as retrieve_top_m_by_structure /
    retrieve_top_m_by_structure_ann for drop-in compatibility,
    but ignores structure-specific args (compressed_runs, parser_groups,
    max_hops) and delegates entirely to coarse_filter_torch.
    """
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    if not question_edge_runs:
        return {"score": [], "questions_index": [], "question_index": [],
                "db_source": [], "index_combo": []}

    # Build cache once if not provided
    t0 = time.perf_counter()
    cache = prebuilt_cache or TorchDBCache.build(codebook_main, target=target)
    timings["cache"] = time.perf_counter() - t0

    # Use all edge runs directly as input to coarse_filter_torch
    t0 = time.perf_counter()
    result = coarse_filter_torch(
        question_edge_runs,
        codebook_main,
        sentence_emb,
        top_k=top_k,
        top_m=top_m,
        target=target,
        prebuilt_cache=cache,
    )
    timings["coarse_filter"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_total
    _print_timings(timings)

    n_results = len(result.get("score", []))
    print(f"[simple] {target}: {n_results} results from coarse_filter_torch "
          f"({len(question_edge_runs)} edge runs)")

    return result

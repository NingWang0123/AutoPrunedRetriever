"""
Concurrent LLM triplet parser.

Same API‐call logic as llm_parser.py, but fires ALL texts concurrently
via ThreadPoolExecutor, collects all triplet sets, then returns them.

Usage:
    from graph_generator.llm_parser_concurrent import triplet_parser_llm_concurrent

    # Returns List[Set[Triplet]] — one set per input text
    all_triplets = triplet_parser_llm_concurrent(list_of_texts, api=api, model=model)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, Set, Tuple, List, Any, Mapping

from graph_generator.llm_parser import (
    _extract_triplets_single,
    _truncate_to_max_tokens,
    _coerce_to_text,
    _client_from_api,
    record_usage,
    TOKEN_STATS,
)

Triplet = Tuple[str, str, str]


def triplet_parser_llm_concurrent(
    text_list: List[str],
    *,
    max_new_tokens: int = 256,
    max_workers: int = 40,
    api: Optional[Union[str, Mapping]] = None,
    client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
) -> List[Set[Triplet]]:
    """
    Extract triplets from ALL texts concurrently.

    Unlike triplet_parser_llm which processes batches sequentially,
    this fires all API calls at once and waits for all to complete.

    Returns: List[Set[Triplet]] — one triplet set per input text,
             in the same order as the input.
    """
    _client = client or _client_from_api(api)

    texts = [_coerce_to_text(t) for t in text_list]
    truncated = [_truncate_to_max_tokens(t, max_tokens=8000) for t in texts]

    n = len(truncated)
    if n == 0:
        return []

    results: list = [set()] * n

    def _call(idx_text):
        idx, text = idx_text
        return idx, _extract_triplets_single(_client, text, max_new_tokens, model)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, (i, t)): i for i, t in enumerate(truncated)}
        for fut in as_completed(futures):
            idx, triplets = fut.result()
            results[idx] = triplets

    return results

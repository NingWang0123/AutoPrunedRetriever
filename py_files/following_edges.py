from typing import List, Tuple, Dict, Set

Triple = Tuple[str, str, str]  # (head, rel, tail)

def longest_path_including_edge(triples: List[Triple], selected: Triple, max_steps: int = None) -> List[Triple]:
    """
    Find the longest simple path (no repeated nodes) that includes `selected`.
    Path orientation follows edge directions. The result is a list of triples in order.

    Args
    ----
    triples  : list of (h, r, t)
    selected : the triple (h, r, t) that must be in the returned path
    max_steps: optional soft cap to prevent pathological runtime on very loopy graphs

    Notes
    -----
    - If the graph is large and has many cycles, this DFS-based search can be expensive.
      Use `max_steps` to cap exploration or pre-prune the graph (e.g., restrict relations).
    """
    # Normalize to tuples and index edges to allow duplicates between same nodes.
    triples = [tuple(tr) for tr in triples]
    if tuple(selected) not in triples:
        raise ValueError("Selected triple not found in triples.")

    # Build adjacency by node with edge indices (retain relation labels)
    fwd: Dict[str, List[int]] = {}
    rev: Dict[str, List[int]] = {}
    for ei, (h, r, t) in enumerate(triples):
        fwd.setdefault(h, []).append(ei)
        rev.setdefault(t, []).append(ei)

    sel_idx = triples.index(tuple(selected))
    sel_h, sel_r, sel_t = triples[sel_idx]

    # Simple-path DFS that extends forward from a node (using fwd),
    # respecting a visited node set; returns a list of edge indices.
    steps_used = [0]

    def extend_forward(node: str, visited: Set[str]) -> List[int]:
        if max_steps is not None and steps_used[0] >= max_steps:
            return []
        best: List[int] = []
        for ei in fwd.get(node, []):
            h, r, t = triples[ei]
            if t in visited:
                continue
            steps_used[0] += 1
            cand = [ei] + extend_forward(t, visited | {t})
            if len(cand) > len(best):
                best = cand
        return best

    # Similar, but extend backward by following reverse edges into predecessors.
    def extend_backward(node: str, visited: Set[str]) -> List[int]:
        if max_steps is not None and steps_used[0] >= max_steps:
            return []
        best: List[int] = []
        for ei in rev.get(node, []):
            h, r, t = triples[ei]
            # rev edges are indexed by t; moving backward means we go ... -> h -(r)-> t(node)
            if h in visited:
                continue
            steps_used[0] += 1
            cand = extend_backward(h, visited | {h}) + [ei]
            if len(cand) > len(best):
                best = cand
        return best

    # Seed path with the selected edge, then grow both sides.
    # Mark the selected edge's endpoints as visited to keep path simple.
    visited_seed = {sel_h, sel_t}
    back = extend_backward(sel_h, visited_seed.copy())
    fwd_ext = extend_forward(sel_t, visited_seed.copy())

    # Combine: backward edges (already oriented left→right), selected, forward edges
    best_edge_indices = back + [sel_idx] + fwd_ext
    return [triples[i] for i in best_edge_indices]


def get_all_following_triples_for_topt(all_triples_lst: List[Triple],selected_triples_lst: List[Triple]):
  all_following_edges = []
  for triple in selected_triples_lst:
    following_edges = longest_path_including_edge(all_triples_lst, triple)
    all_following_edges.extend(following_edges)

    for edge in following_edges:
      all_triples_lst.remove(edge)

  return all_triples_lst


# following_edges.py
# linearization_utils.py
# Adapter-style utilities that reuse functions from retrievel_with_json.py

from typing import List, Dict, Optional, Tuple
import json
import re
import networkx as nx
from rag_workflow_v1 import *

 

# -----------------------------
# Utility: simple JSON dump
# -----------------------------
def _json_dump(obj, pretty: bool = False) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":")
    )

# -----------------------------
# Normalize input into triples
# -----------------------------
def _triples_from_graph_or_relations(
    G: Optional[nx.Graph],
    relations: Optional[List[Dict]],
) -> List[Tuple[str, str, str]]:
    """
    Prefer triples from `relations` if they contain tail entities,
    otherwise fall back to graph edges (relation taken from rel/label/causal_type).
    """
    triples: List[Tuple[str, str, str]] = []

    if relations:
        for r in relations:
            h = r.get("cause") or r.get("head") or r.get("source")
            rel = r.get("causal_type") or r.get("rel") or r.get("relation") or "related_to"
            t = r.get("effect") or r.get("tail") or r.get("target")
            if h and t:
                triples.append((str(h), str(rel), str(t)))

    if not triples and G is not None:
        iterator = (
            G.edges(data=True) if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
            else G.edges(keys=True, data=True)
        )
        for e in iterator:
            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                u, v, k, d = e
            else:
                u, v, d = e
            rel = (d or {}).get("rel") or (d or {}).get("label") or (d or {}).get("causal_type") or "related_to"
            triples.append((str(u), str(rel), str(v)))

    return triples

# -----------------------------
# v1 / v2 / v3 / v3_indexed unified entry (based on retrievel_with_json)
# -----------------------------
def to_json_v1_from_sentence(question: str) -> Dict:
    """
    v1: {entity_dict, relation_dict, edges}
    Uses sentence_relations + triples_to_id_dictionary (from retrievel_with_json).
    """
    triples = sentence_relations(question, include_det=False)
    return triples_to_id_dictionary(triples, tasks="answer the questions")

def to_v2_codebook_from_triples(triples: List[Tuple[str, str, str]]):
    """
    v2: return (codebook, edges)
    codebook: {sid, e, r, rule}, edges: [[h,r,t], ...]
    """
    codebook, ent2id, rel2id = build_codebook_from_triples(list(triples))
    edges = edges_from_triples(list(triples), ent2id, rel2id)
    return codebook, edges

def to_v3_from_sentence(question: str, include_rule: bool = False) -> Tuple[Dict, List[List[int]]]:
    """
    v3 ("merged" form): returns (lite_obj, questions_edges)
    - lite_obj only contains {"e","r"} (optionally with "rule")
    - questions_edges: [[e,r,e], ...] (full triples in id form)

    Reuses get_merged_message(use_full_edges=True) (from retrievel_with_json),
    which usually returns:
      {"e": [...], "r": [...], "rule": "...", "questions([[e,r,e], ...])": [[...], ...]}
    """
    merged = get_merged_message(question, use_full_edges=True)
    q_key = "questions([[e,r,e], ...])"
    q_edges = merged.get(q_key) or merged.get("questions") or []
    lite = {"e": merged["e"], "r": merged["r"]}
    if include_rule and "rule" in merged:
        lite["rule"] = merged["rule"]
    return lite, q_edges

def to_v3_indexed_from_sentence(question: str, include_rule: bool = False) -> Tuple[Dict, List[List[int]], List[int]]:
    """
    v3_indexed: returns (lite_obj, edges, questions_idx)
    - lite_obj only contains {"e","r"} (optionally with "rule")
    - edges: [[e,r,e], ...]
    - questions_idx: [[edge_idx, ...], ...] or flattened list (depending on retrievel implementation)

    Reuses get_merged_message(use_full_edges=False) (from retrievel_with_json),
    which usually returns:
      {
        "e": [...], "r": [...], "rule": "...",
        "edges([e,r,e])": [[...], ...],
        "questions(edges[i])": [[...], ...]
      }
    """
    merged = get_merged_message(question, use_full_edges=False)
    edges = merged.get("edges([e,r,e])") or merged.get("edges") or []
    q_idx = merged.get("questions(edges[i])") or merged.get("questions") or []
    lite = {"e": merged["e"], "r": merged["r"]}
    if include_rule and "rule" in merged:
        lite["rule"] = merged["rule"]
    return lite, edges, q_idx

# -----------------------------
# Backward compatibility: build_relationship_text
# -----------------------------
def build_relationship_text(
    question: str,
    G: Optional[nx.Graph] = None,
    relations: Optional[List[Dict]] = None,
    *,
    include_json_block: bool = True,      # whether to output JSON (usually True)
    json_style: str = "codebook_main",               # "v3" | "v3_indexed" | "id_dict" | "codebook"
    json_pretty: bool = False,            # whether to pretty-print JSON
) -> Tuple[str, List[List[int]], List[int]]:
    """
    Unified return:
      - page_content_str: JSON string (according to json_style)
      - edges: [[e,r,e], ...]               (only valid for v3_indexed; empty for v3)
      - questions_idx: [[edge_idx,...], ...] (only valid for v3_indexed; empty for v3)

    Note: To maximize reuse of retrievel_with_json,
          this function prefers "parse sentence question directly".
          Only when you explicitly provide relations/G will it use them
          (mainly for v1/v2). For v3 it is recommended to parse from sentence.
    """
    if not include_json_block:
        # In old logic, multiple blocks were concatenated; here we only reuse JSON.
        return "", [], []

    # If relations/G are provided, allow generating directly (mainly supports v1/v2).
    triples: List[Tuple[str, str, str]] = _triples_from_graph_or_relations(G, relations)

    if json_style == "id_dict":
        # v1: if we have triples, don't re-parse
        if triples:
            obj = triples_to_id_dictionary(triples, tasks="answer the questions")
        else:
            obj = to_json_v1_from_sentence(question)
        return _json_dump(obj, json_pretty), [], []

    if json_style == "codebook":
        codebook_sub = get_code_book(question)
        codebook = merging_codebook(None, codebook_sub, word_emb=word_emb)
        codebook_main = {
                        "e": codebook['e'],  
                        "r": codebook['r']
                        }
        return str(codebook_main), codebook["edge_matrix"], codebook["questions_lst"]
    
    if json_style == "codebook_main":
        codebook_sub = get_code_book(question)
        codebook_main = merging_codebook(None, codebook_sub, word_emb=word_emb)
        return codebook_main

    if json_style == "v3":
        # v3: return {"e","r"} object; questions returned separately
        lite, q_edges = to_v3_from_sentence(question, include_rule=False)
        return _json_dump(lite, json_pretty), [], q_edges

    if json_style == "v3_indexed":
        # v3_indexed: return {"e","r"} + metadata edges/questions_idx
        lite, edges, q_idx = to_v3_indexed_from_sentence(question, include_rule=False)
        return _json_dump(lite, json_pretty), edges, q_idx

    # default fallback: v3
    lite, q_edges = to_v3_from_sentence(question, include_rule=False)
    return _json_dump(lite, json_pretty), [], q_edges

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys.stderr, 'reconfigure'):
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json, hashlib, re, time, copy
import os as _os
from typing import List, Tuple, Dict, Optional, Iterable, Any, Callable, Set, Union, Mapping
from collections import defaultdict
from copy import deepcopy
from math import ceil
from textwrap import dedent
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from combine_ent_cached_aligned import combine_ents_auto, combine_ents_ann_knn, coarse_combine
from mem_debug import snapshot, codebook_mem_breakdown, force_gc_and_report
from graph_generator.llm_parser import (
    triplet_parser_llm, triplet_parser_llm_question,
    triplet_parser_llm_question_structured, TOKEN_STATS,
)
from graph_generator.llm_parser_concurrent import triplet_parser_llm_concurrent
from sentence_embed_overlap_cached import get_unique_or_overlap_by_sentence_embedded
from test_continous_chunk_cached import embed_triples_as_sentences, segment_by_centroid_sim, merge_chunks_by_boundary
from retrieve_gpu_cached_combined import coarse_filter_torch, _print_timings
from retrieve_simple import retrieve_top_m_by_structure_simple as retrieve_top_m_by_structure

DEBUG_VERBOSE = _os.environ.get("DEBUG_VERBOSE", "0") == "1"



Triplet = Tuple[str, str, str]



def get_context(final_merged_json):
    def _triples_to_words(triples, cb):
        E, R = cb["e"], cb["r"]
        return [[E[h], R[r], E[t]] for (h, r, t) in triples]
    
    def _decode_block(block, cb):
        if not block:
            return []
        if isinstance(block[0], (list, tuple)) and len(block[0]) == 3 and all(isinstance(x, int) for x in block[0]):
            return _triples_to_words(block, cb)
        if isinstance(block[0], int):
            edges = cb.get("edges([e,r,e])", cb.get("edge_matrix"))
            triples = [edges[i] for i in block]
            return _triples_to_words(triples, cb)
        if isinstance(block[0], (list, tuple)) and isinstance(block[0][0], str):
            return block
        return []

    def _linearize_triples_block(triples, sep=", ", end=""):
        if not triples:
            return "None."
        return sep.join(f"{h} {r} {t}{end}" for h, r, t in triples)

    def _extract_txt(keys):
        for k in keys:
            groups = final_merged_json.get(k, [])
            if groups:
                all_words = []
                for g in groups:
                    words = _decode_block(g, final_merged_json)
                    if words:
                        all_words.append(_linearize_triples_block(words))
                return " | ".join(all_words) if all_words else "None."
        return "None."

    q_txt  = _extract_txt(["questions([[e,r,e], ...])"])
    gk_txt = _extract_txt(["given knowledge([[e,r,e], ...])"])
    st_txt = _extract_txt(["start thinking with([[e,r,e], ...])"])
    ft_txt = _extract_txt(["facts([[e,r,e], ...])"])  

    return q_txt, gk_txt, st_txt, ft_txt


# for edges one
def get_context_edge_index(final_merged_json):
  def _linearize_triples_block(triples, sep=", ", end=""):
      if not triples:
          return "None."
      return sep.join(f"{h} {r} {t}{end}" for h, r, t in triples)

  def _extract_txt_groups(groups):
    if groups:
        all_words = []
        for g in groups:
            if g:
                all_words.append(_linearize_triples_block(g))
        return " | ".join(all_words) if all_words else "None."

  q_txt,gk_txt,st_txt,ft_txt = [None]*4
  if "questions(edges[i])" in final_merged_json:
    q_txt  = decode_questions(final_merged_json["questions(edges[i])"], final_merged_json,fmt='words')
    q_txt = _extract_txt_groups(q_txt)
  if "given knowledge(edges[i])" in final_merged_json:
    gk_txt = decode_questions(final_merged_json["given knowledge(edges[i])"], final_merged_json,fmt='words')
    gk_txt = _extract_txt_groups(gk_txt)
  if "start thinking with(edges[i])" in final_merged_json:
    st_txt = decode_questions(final_merged_json["start thinking with(edges[i])"], final_merged_json,fmt='words')
    st_txt = _extract_txt_groups(st_txt)
  if "facts(edges[i])" in final_merged_json:
    ft_txt = decode_questions(final_merged_json["facts(edges[i])"],final_merged_json, fmt='words')  
    ft_txt = _extract_txt_groups(ft_txt)

  return q_txt, gk_txt, st_txt, ft_txt

# automatically select context
def select_best_context_by_keys(final_merged_json):
    # The keys each function relies on
    triple_keys = [
        "questions([[e,r,e], ...])",
        "given knowledge([[e,r,e], ...])",
        "start thinking with([[e,r,e], ...])",
        "facts([[e,r,e], ...])",
    ]
    edge_keys = [
        "questions(edges[i])",
        "given knowledge(edges[i])",
        "start thinking with(edges[i])",
        "facts(edges[i])",
    ]

    # Count how many of those keys exist in the JSON
    triple_count = sum(1 for k in triple_keys if k in final_merged_json)
    edge_count   = sum(1 for k in edge_keys   if k in final_merged_json)

    # Pick the function with more matching keys
    if edge_count > triple_count:
        return get_context_edge_index(final_merged_json)
    else:
        return get_context(final_merged_json)

# -------- node labels --------

# -------- graph build/plot --------


# ---------- ) JSON with ID + Dictionary ----------





def build_codebook_from_triples(
    triples: List[Tuple[str, str, str]],
    rule: str = "Reply with a Y/N/? string in order only; no explanations."
):
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    entities: List[str] = []
    relations: List[str] = []

    def _eid(x: str) -> int:
        if x not in ent2id:
            ent2id[x] = len(entities)
            entities.append(x)
        return ent2id[x]

    def _rid(x: str) -> int:
        if x not in rel2id:
            rel2id[x] = len(relations)
            relations.append(x)
        return rel2id[x]

    # Touch all nodes/relations to populate dictionaries
    for h, r, t in triples:
        _eid(h); _rid(r); _eid(t)

    # Stable short id for this codebook
    sid_src = json_dump_str({"e": entities, "r": relations})
    sid = hashlib.sha1(sid_src.encode("utf-8")).hexdigest()[:10]

    codebook = {
        "sid": sid,
        "e": entities,   # entity dictionary 
        "r": relations,  # relation dictionary 
        "rule": rule
    }
    return codebook, ent2id, rel2id

# ---------- Edges from triples using the codebook ----------
def edges_from_triples(
    triples: List[Tuple[str, str, str]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
) -> List[List[int]]:
    g = []
    for h, r, t in triples:
        g.append([ent2id[h], rel2id[r], ent2id[t]])
    return g


#### add answers to code book



########### Aug 30-31,2025: merging code book method

# change back to your own path


def get_word_embeddings(list_of_text,word_emb):
    """
    list_of_text: ['str1 str2 ...',]
    word_emb: embedding model

    list_of_text_embeddings:  [embedding_vals,...]
    """
    # Check if it's HuggingFaceEmbeddings or Word2VecEmbeddings
    if hasattr(word_emb, '_embed_text'):
        # Word2VecEmbeddings or WordAvgEmbeddings
        list_of_text_embeddings = [word_emb._embed_text(text) for text in list_of_text]
    elif hasattr(word_emb, 'embed_documents'):
        # HuggingFaceEmbeddings
        list_of_text_embeddings = word_emb.embed_documents(list_of_text)
    else:
        raise AttributeError(f"Unsupported embedding model type: {type(word_emb)}")

    # Ensure all embeddings are numpy arrays with consistent shape
    list_of_text_embeddings = [np.asarray(emb, dtype=np.float32) for emb in list_of_text_embeddings]
    
    return list_of_text_embeddings

def _normalize_embeddings_shape(embeddings_list, target_dim=None):
    """
    Normalize embedding shapes to ensure consistency
    """
    if embeddings_list is None:
        return []
    
    # Convert to numpy arrays if not already
    embeddings_list = [np.asarray(emb, dtype=np.float32) for emb in embeddings_list]
    
    # Determine target dimension
    if target_dim is None:
        target_dim = max(emb.shape[0] if emb.ndim > 0 else 1 for emb in embeddings_list)
    
    normalized_embeddings = []
    for emb in embeddings_list:
        if emb.ndim == 0:  # scalar
            emb = np.array([float(emb)], dtype=np.float32)
        elif emb.ndim > 1:  # flatten if multi-dimensional
            emb = emb.flatten().astype(np.float32)
        else:
            emb = emb.astype(np.float32)
            
        # Resize to target dimension
        if len(emb) > target_dim:
            emb = emb[:target_dim]  # truncate
        elif len(emb) < target_dim:
            # pad with zeros
            padding = np.zeros(target_dim - len(emb), dtype=np.float32)
            emb = np.concatenate([emb, padding])
            
        normalized_embeddings.append(emb)
    
    return normalized_embeddings


# ========= Build ONE global codebook from chunked triples =========

Triple = Tuple[str, str, str]

def build_codebook_from_chunks(
    chunks: List[List[Triple]],
    rule: str = "Reply with a Y/N/? string in order only; no explanations."
):
    """
    Builds a single global codebook (shared dictionaries) across all chunks.
    Returns:
      codebook: {"sid","e","r","rule"}
      ent2id, rel2id: global maps
    """
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    entities: List[str] = []
    relations: List[str] = []

    def _eid(x: str) -> int:
        if x not in ent2id:
            ent2id[x] = len(entities)
            entities.append(x)
        return ent2id[x]

    def _rid(x: str) -> int:
        if x not in rel2id:
            rel2id[x] = len(relations)
            relations.append(x)
        return rel2id[x]

    # Touch all nodes/relations
    for triples in chunks:
        for h, r, t in triples:
            _eid(h); _rid(r); _eid(t)

    # Stable short id
    sid_src = json_dump_str({"e": entities, "r": relations})
    sid = hashlib.sha1(sid_src.encode("utf-8")).hexdigest()[:10]

    codebook = {"sid": sid, "e": entities, "r": relations, "rule": rule}
    return codebook, ent2id, rel2id

# ========= Convert chunked triples to edges using GLOBAL ids =========
def edges_from_chunks(
    chunks: List[List[Triple]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
) -> List[List[List[int]]]:
    """
    Returns chunked edges with global ids: [[[h,r,t],...], ...]
    """
    out: List[List[List[int]]] = []
    for triples in chunks:
        g = []
        for h, r, t in triples:
            g.append([ent2id[h], rel2id[r], ent2id[t]])
        out.append(g)
    return out

def flatten_edges_with_index(
    chunked_edges: List[List[List[int]]]
):
    """
    Flattens [[[...]], [[...], [...]], ...] -> [[...], [...], ...]
    and returns:
      - flat: the flattened edges
      - idx_map: {(chunk_idx, triple_idx) -> global_edge_idx}
      - chunk_to_global: [[global_idx,...] per chunk], e.g. [[0],[1,2]]
    """
    flat: List[List[int]] = []
    idx_map: Dict[Tuple[int, int], int] = {}
    chunk_to_global: List[List[int]] = []
    k = 0

    for ci, edges in enumerate(chunked_edges):
        row: List[int] = []
        for ti, e in enumerate(edges):
            flat.append(e)
            idx_map[(ci, ti)] = k
            row.append(k)
            k += 1
        chunk_to_global.append(row)

    return flat, idx_map, chunk_to_global


### edit codebook to also take the answers
def _merge_sets(sets: Iterable[Set[Triplet]]) -> Set[Triplet]:
    merged: Set[Triplet] = set()
    for s in sets:
        if s:
            merged |= s
    return merged

def get_code_book(
    prompt: Union[str, List[str]],
    type: str = 'questions',
    rule: str = "Answer questions.",
    factparser: bool = False,   
    *,
    batch_size: int = 1,
    sent_emb = word_emb,
    parser_choice = 'rebel',
    api: Optional[Union[str, Mapping]] = None,
    model: str = "gpt-4o-mini"      
):
    
    if parser_choice == 'rebel':
        parser = triplet_parser
    elif parser_choice == 'llm':
        parser = partial(triplet_parser_llm,api = api, model = model)
    elif parser_choice == 'llm_question':
        parser = partial(triplet_parser_llm_question, api=api, model=model)
    elif parser_choice == 'llm_question_structured':
        return _get_code_book_structured(prompt, type, rule, api=api, model=model)
    else:
        raise ValueError("parser must be one of 'rebel', 'llm', 'llm_question', 'llm_question_structured'")
    
    valid_types = {'questions', 'answers', 'thinkings', 'facts'}

    processing_texts_lst = False

    if type not in valid_types:
        raise ValueError(f"type must be one of {valid_types}, got: {type}")

    if isinstance(prompt, str):
        triples_merged: Set[Triplet] = parser(prompt)  # Set[Triplet]
    else:
        processing_texts_lst = True
        texts: List[str] = [t for t in prompt if isinstance(t, str) and t.strip()]
        if not texts:
            triples_merged = set()
        elif len(texts) <= batch_size:
            parsed = parser(texts)
            if isinstance(parsed, set):
                triples_merged = parsed
            elif isinstance(parsed, list):
                triples_merged = _merge_sets(parsed)          
            else:
                triples_merged = set(parsed)                 
        else:
            acc: List[Iterable[Triplet]] = []
            for i in range(0, len(texts), batch_size):
                parsed = parser(texts[i:i+batch_size])
                if isinstance(parsed, set):
                    acc.append(parsed)  
                elif isinstance(parsed, list):
                    for s in parsed:    
                        acc.append(s if isinstance(s, set) else set(s))
                else:
                    acc.append(set(parsed))
            triples_merged = _merge_sets(acc)

    if not triples_merged:
        feat_name = f"{type}(edges[i])"
        return {
            "e": [],
            "r": [],
            "edges([e,r,e])": [],
            feat_name: [],
            "rule": rule,
        }

    feat_name = f"{type}(edges[i])"

    if parser_choice in ('llm', 'llm_question') and DEBUG_VERBOSE:
        print(f"token stats are {TOKEN_STATS}")

    if not processing_texts_lst:
        codebook, ent2id, rel2id = build_codebook_from_triples(triples_merged, rule)
        edges = edges_from_triples(triples_merged, ent2id, rel2id)
        codebook.update({
            "edges([e,r,e])": edges,
            feat_name: all_chains_no_subchains(edges, False),
        })
    else:
        T_vecs = embed_triples_as_sentences(triples_merged, sent_emb)
        chunks = segment_by_centroid_sim(
            triples_merged, T_vecs,
            tau=0.7,
            patience=0,
            bonus_tail_head=True,
        )
        codebook, ent2id, rel2id = build_codebook_from_chunks(chunks, rule)
        chunked_edges = edges_from_chunks(chunks, ent2id, rel2id)
        edges, idx_map,chunk_to_global = flatten_edges_with_index(chunked_edges)
        codebook.update({
            "edges([e,r,e])": edges,
            feat_name: chunk_to_global,
        })

        if DEBUG_VERBOSE:
            print(codebook)

    codebook.pop('sid', None)
    return codebook


def _get_code_book_structured(
    prompt: str,
    type: str,
    rule: str,
    *,
    api=None,
    model: str = "gpt-4o-mini",
):
    """
    Structured codebook builder using the unified parser.

    Calls triplet_parser_llm_question_structured to get HOPS + RETRIEVAL + GROUPS,
    builds ONE shared entity/relation space from their union, then stores:
        questions(edges[i])              — HOPS edges (one run per hop)
        questions_compressed(edges[i])   — RETRIEVAL edges (one run per branch)
        questions_groups                 — group structure from parser
    """
    _t0_gcbs = time.perf_counter()
    parsed = triplet_parser_llm_question_structured(prompt, api=api, model=model)
    _t_llm = time.perf_counter() - _t0_gcbs

    hops_triples = parsed.get("hops_triples", set())
    retrieval_triples = parsed.get("retrieval_triples", set())

    all_triples = hops_triples | retrieval_triples
    if not all_triples:
        feat_name = f"{type}(edges[i])"
        return {
            "e": [], "r": [], "edges([e,r,e])": [],
            feat_name: [],
            f"{type}_compressed(edges[i])": [],
            "questions_groups": [],
            "rule": rule,
        }

    # Build shared codebook from union
    codebook, ent2id, rel2id = build_codebook_from_triples(list(all_triples), rule)

    # Build edges for each triple set
    all_edges_list = edges_from_triples(list(all_triples), ent2id, rel2id)
    # Deduplicate edge_matrix
    seen_edges = set()
    unique_edges = []
    for e in all_edges_list:
        key = tuple(e)
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)
    edge_to_idx = {tuple(e): i for i, e in enumerate(unique_edges)}

    # HOPS: one run per hop, in parser order
    hops_dict = parsed.get("hops", {})
    hop_runs = []
    for label in sorted(hops_dict.keys()):
        triple = hops_dict[label]
        try:
            eid = edge_to_idx[tuple(edges_from_triples([triple], ent2id, rel2id)[0])]
            hop_runs.append([eid])
        except (KeyError, IndexError):
            hop_runs.append([])

    # RETRIEVAL: one run per retrieval branch
    ret_dict = parsed.get("retrieval", {})
    ret_runs = []
    for label in sorted(ret_dict.keys()):
        triple = ret_dict[label]
        try:
            eid = edge_to_idx[tuple(edges_from_triples([triple], ent2id, rel2id)[0])]
            ret_runs.append([eid])
        except (KeyError, IndexError):
            ret_runs.append([])

    feat_name = f"{type}(edges[i])"
    compressed_feat_name = f"{type}_compressed(edges[i])"

    codebook.update({
        "edges([e,r,e])": unique_edges,
        feat_name: hop_runs,                           # HOPS as edge runs
        compressed_feat_name: ret_runs,                # RETRIEVAL as edge runs
        "questions_groups": parsed.get("groups", []),  # group structure
    })

    codebook.pop('sid', None)
    _t_build = time.perf_counter() - _t0_gcbs - _t_llm
    print(f"[_get_code_book_structured] llm={_t_llm*1000:.0f}ms build={_t_build*1000:.0f}ms")
    return codebook


def get_code_books(
    all_texts: List[str],
    type: str = 'facts',
    rule: str = "Store factual statements.",
    *,
    batch_size: int = 0,
    sent_emb=None,
    api: Optional[Union[str, Mapping]] = None,
    model: str = "gpt-4o-mini",
    max_workers: int = 40,
) -> List[dict]:
    """
    Concurrent version of get_code_book for list inputs.

    1. Fires ALL API calls concurrently → collects per-text triplet sets
    2. Groups texts into batches (same batching as preload_context_json)
    3. Builds one codebook per batch from the pre-collected triplets

    Returns: List[codebook_dict] — one per batch.
    """
    from graph_generator.llm_parser_concurrent import triplet_parser_llm_concurrent

    texts = [t for t in all_texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    # ── Step 1: fire all API calls concurrently ──
    print(f"[get_code_books] Firing {len(texts)} concurrent API calls (max_workers={max_workers})...")
    import time as _time
    _t0 = _time.time()
    per_text_triplets: List[Set[Tuple[str,str,str]]] = triplet_parser_llm_concurrent(
        texts, api=api, model=model, max_workers=max_workers,
    )
    _elapsed = _time.time() - _t0
    if DEBUG_VERBOSE:
        print(f"[get_code_books] All API calls done in {_elapsed:.1f}s")
        print(f"[get_code_books] token stats are {TOKEN_STATS}")

    # ── Step 2: batch and build codebooks (same logic as preload_context_json) ──
    if batch_size <= 0:
        batch_size = len(texts)

    feat_name = f"{type}(edges[i])"
    codebook_lst = []

    for i in range(0, len(texts), batch_size):
        batch_triplet_sets = per_text_triplets[i:i+batch_size]

        # Merge all triplets in this batch
        triples_merged: Set[Tuple[str,str,str]] = set()
        for s in batch_triplet_sets:
            if s:
                triples_merged |= s

        if not triples_merged:
            codebook_lst.append({
                "e": [], "r": [], "edges([e,r,e])": [],
                feat_name: [], "rule": rule,
            })
            continue

        # Build codebook with chunking (same as get_code_book list path)
        if sent_emb is not None:
            T_vecs = embed_triples_as_sentences(triples_merged, sent_emb)
            chunks = segment_by_centroid_sim(
                triples_merged, T_vecs,
                tau=0.7, patience=0, bonus_tail_head=True,
            )
            codebook, ent2id, rel2id = build_codebook_from_chunks(chunks, rule)
            chunked_edges = edges_from_chunks(chunks, ent2id, rel2id)
            edges, idx_map, chunk_to_global = flatten_edges_with_index(chunked_edges)
            codebook.update({
                "edges([e,r,e])": edges,
                feat_name: chunk_to_global,
            })
        else:
            codebook, ent2id, rel2id = build_codebook_from_triples(triples_merged, rule)
            edges = edges_from_triples(triples_merged, ent2id, rel2id)
            codebook.update({
                "edges([e,r,e])": edges,
                feat_name: all_chains_no_subchains(edges, False),
            })

        codebook.pop('sid', None)
        codebook_lst.append(codebook)
        print(f"[get_code_books] batch {i//batch_size} codebook built ({len(triples_merged)} triplets)")

    return codebook_lst


def update_the_index(codebook_main, codebook_sub, select_feature):
    items_needs_merged = codebook_sub[select_feature]   # list of strings
    items_main = codebook_main[select_feature]          # list of strings

    index_item_sub  = {val: idx for idx, val in enumerate(items_needs_merged)}
    index_item_main = {val: idx for idx, val in enumerate(items_main)}

    # Case-insensitive fallback: lowercase → first index in main
    index_item_main_lower = {}
    for val, idx in index_item_main.items():
        low = val.lower().strip() if isinstance(val, str) else val
        if low not in index_item_main_lower:
            index_item_main_lower[low] = idx

    next_idx = len(items_main)
    new_index_replacement_for_sub = {}
    new_added_items = []

    for item_sub in items_needs_merged:
        if item_sub in index_item_main:
            # Exact match
            new_index_replacement_for_sub[index_item_sub[item_sub]] = index_item_main[item_sub]
        else:
            # Case-insensitive fallback
            low = item_sub.lower().strip() if isinstance(item_sub, str) else item_sub
            if low in index_item_main_lower:
                matched_idx = index_item_main_lower[low]
                new_index_replacement_for_sub[index_item_sub[item_sub]] = matched_idx
                # Also register exact form so future lookups hit
                index_item_main[item_sub] = matched_idx
            else:
                new_index_replacement_for_sub[index_item_sub[item_sub]] = next_idx
                index_item_main[item_sub] = next_idx
                index_item_main_lower[low] = next_idx
                new_added_items.append(item_sub)
                next_idx += 1

    return new_index_replacement_for_sub, index_item_main, new_added_items


def remap_edges_matrix(edges_matrix, ent_map, rel_map):
    # edges_matrix: list[list[[e,r,e], ...]]
    if not edges_matrix:
        return []
    arr = np.asarray(edges_matrix, dtype=np.int64)
    e_cols = arr[:, [0, 2]]
    r_col  = arr[:, 1:2]

    e_max = int(e_cols.max()) if e_cols.size else -1
    r_max = int(r_col.max())  if r_col.size else -1

    # Build dense LUTs (identity by default)
    e_LUT = np.arange(max(e_max + 1, 0), dtype=np.int64)
    r_LUT = np.arange(max(r_max + 1, 0), dtype=np.int64)

    for k, v in ent_map.items():
        if 0 <= k <= e_max:
            e_LUT[k] = v
    for k, v in rel_map.items():
        if 0 <= k <= r_max:
            r_LUT[k] = v

    # Apply remap: columns 0 & 2 are entities; column 1 is relation
    arr[:, 0] = e_LUT[arr[:, 0]]
    arr[:, 2] = e_LUT[arr[:, 2]]
    arr[:, 1] = r_LUT[arr[:, 1]]

    return arr.tolist()


def combine_updated_edges(edges_main, edges_sub):
    index_item_main = {tuple(val): idx for idx, val in enumerate(edges_main)}
    new_index_replacement_for_sub = {}

    next_idx = len(edges_main)
    for idx_sub, item in enumerate(edges_sub):
        key = tuple(item)
        if key in index_item_main:
            new_index_replacement_for_sub[idx_sub] = index_item_main[key]
        else:
            index_item_main[key] = next_idx
            new_index_replacement_for_sub[idx_sub] = next_idx
            next_idx += 1

    edges_main_out = [None] * len(index_item_main)
    for k, v in index_item_main.items():
        edges_main_out[v] = list(k)

    return new_index_replacement_for_sub, edges_main_out



def remap_question_indices(questions, idx_map, max_dense_size=10_000_000):
    """
    Remap edge indices inside questions via idx_map using NumPy.

    Args:
        questions: list[list[int]]  e.g., [[0,2,5], [1,4]]
        idx_map: dict {old_idx -> new_idx}
        max_dense_size: fallback threshold for extremely large/sparse indices.

    Returns:
        list[list[int]] with same nested shape, indices remapped.
    """
    lens = [len(q) for q in questions]
    if not lens:
        return []

    flat = np.fromiter((i for q in questions for i in q), dtype=np.int64)
    if flat.size == 0:
        return [[] for _ in questions]

    if not idx_map:
        # nothing to map; just rebuild shape
        out, k = [], 0
        for n in lens:
            out.append(flat[k:k+n].tolist())
            k += n
        return out

    # Decide dense LUT vs dict get fallback
    size = int(max(flat.max(), max(idx_map))) + 1
    use_dense = size <= max_dense_size

    if use_dense:
        # Dense LUT (identity by default)
        lut = np.arange(size, dtype=np.int64)
        for k, v in idx_map.items():
            if 0 <= k < size:
                lut[k] = v
        mapped = lut[flat]
    else:
        # Fallback for huge/sparse index spaces
        get = idx_map.get
        mapped = np.fromiter((get(x, x) for x in flat), dtype=np.int64)
    # Rebuild original nested shape
    out, k = [], 0
    for n in lens:
        out.append(mapped[k:k+n].tolist())
        k += n
    return out



def _chunk_text_from_edge_run(edge_run: List[int], codebook: Dict[str, Any], sep: str = " <SEP> ") -> str:
    """
    edge_run: List[int] edge indices into codebook['edge_matrix']
    Returns the SAME style as your scorer's c_text: "[H].. [R].. [T].." joined by <SEP>.
    """
    E = codebook["e"]
    R = codebook["r"]
    EM = codebook["edge_matrix"]

    parts = []
    for edge_i in edge_run:
        h_id, r_id, t_id = EM[edge_i]
        parts.append(f"[H]{E[h_id]} [R]{R[r_id]} [T]{E[t_id]}")
    return sep.join(parts)

def _embed_group_full_texts(
    group_edge_runs: List[List[int]],
    codebook: Dict[str, Any],
    word_emb,
    *,
    normalize: bool = False,
    zero_on_empty: bool = False,
    dim_hint: Optional[int] = None,
) -> np.ndarray:
    """
    group_edge_runs: List[edge_run], each edge_run is List[int]
    Returns: (n_chunks, d) float32
    """
    texts = []
    empty_mask = []
    for edge_run in group_edge_runs:
        txt = _chunk_text_from_edge_run(edge_run, codebook)
        is_empty = (len(txt) == 0)
        empty_mask.append(is_empty)
        # avoid embedding empty if you want deterministic zeros
        texts.append(txt if (not is_empty or not zero_on_empty) else " ")  # placeholder

    vecs = get_word_embeddings(texts, word_emb) if texts else []
    V = np.asarray(vecs, dtype=np.float32) if len(vecs) else np.zeros((0, dim_hint or 0), dtype=np.float32)

    if V.size == 0 and dim_hint is not None:
        V = np.zeros((len(group_edge_runs), dim_hint), dtype=np.float32)

    # if we used placeholders for empty, optionally overwrite with zeros
    if zero_on_empty and V.size and any(empty_mask):
        d = V.shape[1]
        for i, is_empty in enumerate(empty_mask):
            if is_empty:
                V[i] = 0.0

    if normalize and V.size:
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        V = V / norms
    return V

def merging_codebook(codebook_main, codebook_sub, type='questions', word_emb=None, use_thinkings=False):
    if type == 'fact':
        type = 'facts'

    feat_name_candidates = [f"{type}(edges[i])", f"{type}_lst"]
    feat_name = next((k for k in feat_name_candidates if k in codebook_sub), None)
    if feat_name is None:
        raise KeyError(f"Expected one of {feat_name_candidates} in codebook_sub")

    if type == 'questions':
        main_feat_name = 'questions_lst'
        unupdated_feat_name1 = 'answers_lst'
        unupdated_feat_name2 = 'thinkings_lst'
    elif type == 'answers':
        main_feat_name = 'answers_lst'
        unupdated_feat_name1 = 'questions_lst'
        unupdated_feat_name2 = 'thinkings_lst'
    elif type == 'thinkings':
        main_feat_name = 'thinkings_lst'
        unupdated_feat_name1 = 'questions_lst'
        unupdated_feat_name2 = 'answers_lst'
    elif type == 'facts':
        main_feat_name = 'facts_lst'
        unupdated_feat_name1 = 'questions_lst'
        unupdated_feat_name2 = 'answers_lst'
    else:
        raise ValueError(f"Unknown type={type}")

    # convenience: embedding key for chunk-level C_full cache
    main_feat_emb_key = f"{main_feat_name}_embedding"

    # === CASE 1: main codebook exists ===
    if codebook_main:
        codebook_main.setdefault('e', [])
        codebook_main.setdefault('r', [])
        codebook_main.setdefault('edge_matrix', [])
        codebook_main.setdefault('e_embeddings', [])
        codebook_main.setdefault('r_embeddings', [])
        codebook_main.setdefault('answers_lst', [])
        codebook_main.setdefault('thinkings_lst', [])
        codebook_main.setdefault('questions_lst', [])
        codebook_main.setdefault('facts_lst', [])
        codebook_main.setdefault('questions_to_thinkings', {})

        # ALSO ensure embedding lists exist (parallel to *_lst)
        codebook_main.setdefault("questions_lst_embedding", [])
        codebook_main.setdefault("answers_lst_embedding", [])
        codebook_main.setdefault("facts_lst_embedding", [])
        codebook_main.setdefault("thinkings_lst_embedding", [])

        items_needs_merged = codebook_sub[feat_name]
        lst_main = codebook_main[main_feat_name]

        edge_mat_needs_merged = codebook_sub.get('edges([e,r,e])', codebook_sub.get('edge_matrix'))
        edge_mat_main = codebook_main['edge_matrix']
        old_edge_len = len(edge_mat_main)

        # --- Update entity and relation indices (incremental) ---
        new_idx_map_ent_sub, index_ent_main, new_added_ents = update_the_index(codebook_main, codebook_sub, 'e')
        new_idx_map_r_sub,   index_r_main,  new_added_rs   = update_the_index(codebook_main, codebook_sub, 'r')

        # --- Compute embeddings ONLY for newly added entities & relations ---
        existing_e_embeds = codebook_main.get('e_embeddings', [])
        existing_r_embeds = codebook_main.get('r_embeddings', [])

        e_target_dim = len(existing_e_embeds[0]) if len(existing_e_embeds) > 0 else None
        r_target_dim = len(existing_r_embeds[0]) if len(existing_r_embeds) > 0 else None

        new_ent_embeds = get_word_embeddings(new_added_ents, word_emb) if new_added_ents else []
        new_r_embeds   = get_word_embeddings(new_added_rs,   word_emb) if new_added_rs   else []

        if new_ent_embeds and e_target_dim is not None:
            new_ent_embeds = _normalize_embeddings_shape(new_ent_embeds, e_target_dim)
        if new_r_embeds and r_target_dim is not None:
            new_r_embeds   = _normalize_embeddings_shape(new_r_embeds, r_target_dim)

        # Convert to numpy float32 immediately to save ~75% memory
        if new_ent_embeds:
            new_ent_embeds = [np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in new_ent_embeds]
        if new_r_embeds:
            new_r_embeds = [np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v for v in new_r_embeds]

        # --- Embedding-based entity matching for "new" entities ---
        # If a newly-added entity has high cosine similarity to an existing entity,
        # remap it instead of keeping it as a new entity.
        if new_added_ents and len(new_ent_embeds) > 0 and len(existing_e_embeds) > 0 and e_target_dim:
            import torch as _torch
            _EMB_MATCH_THRESH = 0.93  # same as combine_ent_sim
            try:
                new_e_t = _torch.tensor(np.array(new_ent_embeds, dtype=np.float32))
                exist_e_t = _torch.tensor(np.array(existing_e_embeds, dtype=np.float32))
                # L2 normalize
                new_e_t = new_e_t / (new_e_t.norm(dim=1, keepdim=True) + 1e-9)
                exist_e_t = exist_e_t / (exist_e_t.norm(dim=1, keepdim=True) + 1e-9)
                # Cosine similarity: (n_new, n_existing)
                sims = _torch.mm(new_e_t, exist_e_t.T)
                best_sims, best_idxs = sims.max(dim=1)

                remapped_count = 0
                old_main_len = len(codebook_main.get('e', []))  # before new ents appended
                keep_new_ents = []
                keep_new_embeds = []
                for i, ent_name in enumerate(new_added_ents):
                    sim_val = best_sims[i].item()
                    match_idx = best_idxs[i].item()
                    if sim_val >= _EMB_MATCH_THRESH:
                        # Remap: this "new" entity should point to existing entity
                        assigned_new_idx = old_main_len + i - remapped_count
                        # Find the sub index that was mapped to this new idx
                        for sub_idx, main_idx in new_idx_map_ent_sub.items():
                            if main_idx == old_main_len + i:
                                new_idx_map_ent_sub[sub_idx] = match_idx
                        remapped_count += 1
                    else:
                        keep_new_ents.append(ent_name)
                        keep_new_embeds.append(new_ent_embeds[i])

                if remapped_count > 0:
                    # Reassign sequential indices for remaining new entities
                    remaining_idx = old_main_len
                    new_ent_positions = {}  # old position in new_added_ents -> new sequential idx
                    for i, ent_name in enumerate(new_added_ents):
                        if ent_name in keep_new_ents:
                            new_ent_positions[old_main_len + i] = remaining_idx
                            remaining_idx += 1
                    # Fix up the index map for kept new entities
                    for sub_idx, main_idx in new_idx_map_ent_sub.items():
                        if main_idx in new_ent_positions:
                            new_idx_map_ent_sub[sub_idx] = new_ent_positions[main_idx]
                    new_added_ents = keep_new_ents
                    new_ent_embeds = keep_new_embeds
            except Exception:
                pass  # Fall through to original behavior

        # --- Remap edges and item edge-index lists ---
        edge_mat_needs_merged_remapped = remap_edges_matrix(
            edge_mat_needs_merged, new_idx_map_ent_sub, new_idx_map_r_sub
        )
        new_idx_map_edges_sub, index_edges_main = combine_updated_edges(edge_mat_main, edge_mat_needs_merged_remapped)

        # --- dirty-tracking: record new edge indices ---
        if not isinstance(codebook_main.get("_dirty_edges"), set):
            codebook_main["_dirty_edges"] = set(codebook_main.get("_dirty_edges") or [])
        if not isinstance(codebook_main.get("_dirty_entities"), set):
            codebook_main["_dirty_entities"] = set(codebook_main.get("_dirty_entities") or [])
        new_edge_indices = set(range(old_edge_len, len(index_edges_main)))
        codebook_main["_dirty_edges"].update(new_edge_indices)
        if new_added_ents:
            old_ent_len = len(codebook_main.get("e", []))
            codebook_main["_dirty_entities"].update(range(old_ent_len, old_ent_len + len(new_added_ents)))

        updated_items_sub = remap_question_indices(items_needs_merged, new_idx_map_edges_sub)
        lst_main.append(updated_items_sub)

        # --- Carry RETRIEVAL (compressed) edges and GROUPS through remap ---
        if type == 'questions':
            compressed_feat = f"{type}_compressed(edges[i])"
            if compressed_feat in codebook_sub:
                compressed_items = codebook_sub[compressed_feat]
                updated_compressed = remap_question_indices(compressed_items, new_idx_map_edges_sub)
                codebook_main.setdefault('questions_compressed_lst', []).append(updated_compressed)
            else:
                codebook_main.setdefault('questions_compressed_lst', []).append(None)

            # Carry groups structure (no remap needed, just hop indices)
            if 'questions_groups' in codebook_sub:
                codebook_main.setdefault('questions_groups_lst', []).append(codebook_sub['questions_groups'])
            else:
                codebook_main.setdefault('questions_groups_lst', []).append(None)

        # --- Extend E/R before building texts for newly-added edges ---
        if new_added_ents:
            codebook_main["e"].extend(new_added_ents)
        if new_added_rs:
            codebook_main["r"].extend(new_added_rs)

        # --- Incremental edge_matrix_embedding update (cached C per edge) ---
        existing_edge_embeds = codebook_main.get('edge_matrix_embedding', [])
        existing_edge_embeds = np.asarray(existing_edge_embeds, dtype=np.float32) if len(existing_edge_embeds) else np.zeros((0, 0), dtype=np.float32)

        edge_target_dim = existing_edge_embeds.shape[1] if existing_edge_embeds.size else None

        new_edges = index_edges_main[old_edge_len:]
        if new_edges:
            E_all = codebook_main['e']
            R_all = codebook_main['r']
            edge_texts_new = [f"{E_all[h]} {R_all[r]} {E_all[t]}" for (h, r, t) in new_edges]
            new_edge_embeds = np.asarray(get_word_embeddings(edge_texts_new, word_emb), dtype=np.float32)
            if edge_target_dim is not None:
                new_edge_embeds = np.asarray(_normalize_embeddings_shape(new_edge_embeds, edge_target_dim), dtype=np.float32)
        else:
            new_edge_embeds = np.zeros((0, edge_target_dim or 0), dtype=np.float32)

        if DEBUG_VERBOSE:
            snapshot("merging_codebook: before edge vstack")
            force_gc_and_report("pre-vstack GC")
        if existing_edge_embeds.size and new_edge_embeds.size:
            # Release codebook reference so old array can be freed after copy
            codebook_main['edge_matrix_embedding'] = None
            n_old, d = existing_edge_embeds.shape
            n_new = new_edge_embeds.shape[0]
            edge_embeds_merged = np.empty((n_old + n_new, d), dtype=np.float32)
            edge_embeds_merged[:n_old] = existing_edge_embeds
            del existing_edge_embeds  # free ~233MB
            edge_embeds_merged[n_old:] = new_edge_embeds
            del new_edge_embeds
        elif new_edge_embeds.size:
            edge_embeds_merged = new_edge_embeds
        else:
            edge_embeds_merged = existing_edge_embeds

        # --- Commit merged structures ---
        codebook_main["edge_matrix"] = index_edges_main
        codebook_main[main_feat_name] = lst_main

        # commit e/r embeddings incrementally (extend in-place, no copy)
        if len(new_ent_embeds) > 0:
            cur_e = codebook_main.get("e_embeddings")
            if cur_e is existing_e_embeds and isinstance(existing_e_embeds, list):
                existing_e_embeds.extend(new_ent_embeds)
            else:
                prev = list(existing_e_embeds) if not isinstance(existing_e_embeds, list) else (existing_e_embeds or [])
                codebook_main["e_embeddings"] = prev + list(new_ent_embeds)
        if len(new_r_embeds) > 0:
            cur_r = codebook_main.get("r_embeddings")
            if cur_r is existing_r_embeds and isinstance(existing_r_embeds, list):
                existing_r_embeds.extend(new_r_embeds)
            else:
                prev = list(existing_r_embeds) if not isinstance(existing_r_embeds, list) else (existing_r_embeds or [])
                codebook_main["r_embeddings"] = prev + list(new_r_embeds)
        codebook_main["edge_matrix_embedding"] = edge_embeds_merged  # now 2D np array

        # --- NEW: incremental chunk-level C_full cache for the appended group ---
        # updated_items_sub is ONE group: List[edge_run]
        # Ensure parallel embedding list exists for this main_feat_name
        if main_feat_emb_key not in codebook_main:
            codebook_main[main_feat_emb_key] = []

        # dim hint from edge embeddings
        dim_hint = int(codebook_main["edge_matrix_embedding"].shape[1]) if codebook_main["edge_matrix_embedding"].size else None

        group_full_embeds = _embed_group_full_texts(
            updated_items_sub,
            codebook_main,
            word_emb,
            dim_hint=dim_hint,
        )  # (n_chunks, d)
        codebook_main[main_feat_emb_key].append(group_full_embeds)

        # keep your existing mapping behavior
        if type == 'thinkings':
            codebook_main['questions_to_thinkings'][len(codebook_main['questions_lst']) - 1] = len(codebook_main[main_feat_name]) - 1

    # === CASE 2: main codebook is empty ===
    else:
        edge_matrix = codebook_sub.get('edges([e,r,e])', codebook_sub.get('edge_matrix'))
        codebook_main = {
            "e": codebook_sub['e'],
            "r": codebook_sub['r'],
            "edge_matrix": edge_matrix,
            main_feat_name: [codebook_sub[feat_name]],
            unupdated_feat_name1: [],
            "rule": codebook_sub.get('rule'),
            "e_embeddings": [np.asarray(v, dtype=np.float32) for v in get_word_embeddings(codebook_sub['e'], word_emb)],
            "r_embeddings": [np.asarray(v, dtype=np.float32) for v in get_word_embeddings(codebook_sub['r'], word_emb)],
            "questions_lst_embedding": [],
            "answers_lst_embedding": [],
            "facts_lst_embedding": [],
            "thinkings_lst_embedding": [],
            "questions_to_thinkings": {},
        }

        # per-edge embeddings
        edge_texts = [f"{codebook_main['e'][h]} {codebook_main['r'][r]} {codebook_main['e'][t]}" for (h, r, t) in edge_matrix]
        codebook_main["edge_matrix_embedding"] = np.asarray(get_word_embeddings(edge_texts, word_emb), dtype=np.float32)

        # per-chunk C_full cache for the first (only) group we inserted
        main_feat_emb_key = f"{main_feat_name}_embedding"
        first_group = codebook_main[main_feat_name][0]  # List[edge_run]
        dim_hint = int(codebook_main["edge_matrix_embedding"].shape[1]) if codebook_main["edge_matrix_embedding"].size else None
        codebook_main[main_feat_emb_key] = [
            _embed_group_full_texts(first_group, codebook_main, word_emb, dim_hint=dim_hint)
        ]

        if use_thinkings:
            codebook_main[unupdated_feat_name2] = []
            codebook_main['questions_to_thinkings'] = {}

        # RETRIEVAL (compressed) + GROUPS for structured parser (CASE 2)
        if type == 'questions':
            compressed_feat = f"{type}_compressed(edges[i])"
            if compressed_feat in codebook_sub:
                codebook_main['questions_compressed_lst'] = [codebook_sub[compressed_feat]]
            else:
                codebook_main['questions_compressed_lst'] = [None]
            if 'questions_groups' in codebook_sub:
                codebook_main['questions_groups_lst'] = [codebook_sub['questions_groups']]
            else:
                codebook_main['questions_groups_lst'] = [None]

        # dirty-tracking: everything is new
        codebook_main["_dirty_edges"] = set(range(len(edge_matrix)))
        codebook_main["_dirty_entities"] = set(range(len(codebook_sub['e'])))

    return codebook_main



#### for the merging functions only use when the codebook_sub are not empty

## merging questions and answers codebook in the main code book (questions and answers only, no thinkings)
def merge_questions_and_answers_code_book(codebook_main,codebook_sub_q,codebook_sub_a):
    codebook_with_q = merging_codebook(codebook_main,codebook_sub_q,type='questions')
    final_codebook = merging_codebook(codebook_with_q,codebook_sub_a,type='answers')

    return final_codebook

## merging questions,thinkings and answers codebook in the main code book
def merge_all_code_book(codebook_main,codebook_sub_q,codebook_sub_a,codebook_sub_t):
    codebook_with_q = merging_codebook(codebook_main,codebook_sub_q,type='questions')
    codebook_with_qa = merging_codebook(codebook_with_q,codebook_sub_a,type='answers')
    final_codebook = merging_codebook(codebook_with_qa,codebook_sub_t,type='thinkings')

    return final_codebook



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

def decode_questions(questions, questions_source_codebook, fmt='words'):

    """
    questions_source_codebook must be the codebook that contain the questions
    Decode a list of questions using decode_question.
    
    questions: list of list[int]
        Each inner list is a sequence of edge indices.
    """
    return [decode_question(q, questions_source_codebook, fmt=fmt) for q in questions]


##### word embedding top k search
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
                    # Ensure all vectors have the same dimension
                    if vv.shape[0] != dim:
                        # Resize vector to match expected dimension
                        if vv.shape[0] > dim:
                            vv = vv[:dim]  # truncate
                        else:
                            # pad with zeros
                            padding = np.zeros(dim - vv.shape[0], dtype=np.float32)
                            vv = np.concatenate([vv.astype(np.float32), padding])
                    parts.append(vv.astype(np.float32, copy=False))
    if not parts:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(parts, axis=0), axis=0)

def _embed_questions_with_decode(
    questions_batch: List[List[int]],
    codebook_main: Dict[str, Any],
    dim: int
) -> np.ndarray:
    """
    Use decode_question(..., fmt='embeddings') for each question in the batch,
    then reduce to a single vector via averaging components.
    Returns (B, d) float32 matrix.
    """
    out = np.zeros((len(questions_batch), dim), dtype=np.float32)
    for i, q_edges in enumerate(questions_batch):
        decoded = decode_question(q_edges, codebook_main, fmt='embeddings')
        out[i] = _avg_vec_from_decoded(decoded, dim)
    return out

def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: (n, d), B: (m, d) -> (n, m) cosine similarity matrix.
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def _topk_merge(existing_scores: np.ndarray, existing_cols: np.ndarray,
                batch_scores: np.ndarray, batch_cols: np.ndarray, k: int):
    """
    Merge two top-k candidate sets (scores/cols) -> keep best k, sorted desc.
    """
    if existing_scores.size == 0:
        if batch_scores.size <= k:
            order = np.argsort(-batch_scores)
            return batch_scores[order], batch_cols[order]
        top_idx = np.argpartition(-batch_scores, k - 1)[:k]
        order = np.argsort(-batch_scores[top_idx])
        top_idx = top_idx[order]
        return batch_scores[top_idx], batch_cols[top_idx]

    cand_scores = np.concatenate([existing_scores, batch_scores], axis=0)
    cand_cols   = np.concatenate([existing_cols,   batch_cols],   axis=0)
    if cand_scores.shape[0] <= k:
        order = np.argsort(-cand_scores)
        return cand_scores[order], cand_cols[order]
    top_idx = np.argpartition(-cand_scores, k - 1)[:k]
    order = np.argsort(-cand_scores[top_idx])
    top_idx = top_idx[order]
    return cand_scores[top_idx], cand_cols[top_idx]

def _ensure_embeddings_in_codebook(codebook_main, dim_fallback: int = 64):
    """
    Ensure codebook_main has e_embeddings and r_embeddings.
    Priority:
      1) Use global `word_emb` + get_word_embeddings if available
      2) Fallback to stable random embeddings with fixed dim (dim_fallback)
    """
    import random

    def _hash_embed(tokens, dim=64):
        out = []
        for t in tokens:
            rnd = random.Random(hash(t) & 0xffffffff)
            out.append([rnd.uniform(-1, 1) for _ in range(dim)])
        return out

    def _detect_embedding_dim(word_emb):
        """Detect the embedding dimension from the word_emb model"""
        try:
            if hasattr(word_emb, '_embed_text'):
                # Word2VecEmbeddings or WordAvgEmbeddings
                test_embed = word_emb._embed_text("test")
                return len(test_embed)
            elif hasattr(word_emb, 'embed_documents'):
                # HuggingFaceEmbeddings
                test_embed = word_emb.embed_documents(["test"])[0]
                return len(test_embed)
        except Exception:
            pass
        return dim_fallback

    # Detect actual embedding dimension
    actual_dim = _detect_embedding_dim(word_emb) if 'word_emb' in globals() else dim_fallback

    # --- entities ---
    if "e_embeddings" not in codebook_main or not codebook_main["e_embeddings"]:
        if "e" not in codebook_main:
            raise ValueError("codebook_main missing key 'e' to compute e_embeddings.")
        try:
            # try your word_emb pipeline
            e_embeddings = get_word_embeddings(codebook_main["e"], word_emb)
            codebook_main["e_embeddings"] = _normalize_embeddings_shape(e_embeddings, actual_dim)
        except Exception:
            # fallback stable random
            codebook_main["e_embeddings"] = _hash_embed(codebook_main["e"], dim=actual_dim)

    # --- relations ---
    if "r_embeddings" not in codebook_main or not codebook_main["r_embeddings"]:
        if "r" not in codebook_main:
            raise ValueError("codebook_main missing key 'r' to compute r_embeddings.")
        try:
            r_embeddings = get_word_embeddings(codebook_main["r"], word_emb)
            codebook_main["r_embeddings"] = _normalize_embeddings_shape(r_embeddings, actual_dim)
        except Exception:
            codebook_main["r_embeddings"] = _hash_embed(codebook_main["r"], dim=actual_dim)



##### getting the best sentence embedding results from the top k word embedding results

def _linearize_words_triples(triples: List[List[str]]) -> str:
    """
    Simple, robust fallback linearizer:
      [[h, r, t], ...]  -> "h r t ; h r t ; ..."
    """
    parts = []
    for h, r, t in triples:
        parts.append(f"{h} {r} {t}")
    return " ; ".join(parts)

def make_question_text(
    q_edges: List[int],
    codebook_main: Dict[str, Any],
    custom_linearizer: Optional[Callable[[List[List[str]]], str]] = None,
) -> str:
    """
    Decode with words and turn into a short sentence-ish string for sentence embedding.
    Optionally pass a custom linearizer; otherwise use a simple fallback.
    """
    decoded_words = decode_question(q_edges, codebook_main, fmt='words')  # [[h,r,t], ...]
    if custom_linearizer is not None:
        return custom_linearizer(decoded_words)
    return _linearize_words_triples(decoded_words)




######### new reranker
def _l2norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def _extract_entities_relations_from_run(edge_run: List[int],
                                         codebook_main: Dict[str, Any]
                                         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      E: (n_e, de) or None    # heads+tails
      R: (n_r, dr) or None    # relations
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

    E = np.vstack(Es).astype(np.float32) if Es else None
    R = np.vstack(Rs).astype(np.float32) if Rs else None
    return E, R

def _pairwise_max_cos(A: Optional[np.ndarray], B: Optional[np.ndarray]) -> float:
    """
    Max cosine similarity over all pairs between rows of A and B.
    Returns 0.0 if A or B is None/empty.
    """
    if A is None or B is None or A.size == 0 or B.size == 0:
        return 0.0
    An = _l2norm_rows(A)
    Bn = _l2norm_rows(B)
    # (na, nb) cosine matrix
    S = An @ Bn.T
    return float(S.max())

def entrel_maxpair_similarity(Eq, Rq,Ef, Rf,w_ent: float = 1.0, w_rel: float = 0.5) -> float:
    """
      score = w_ent * max_{ent pair} cos + w_rel * max_{rel pair} cos
    """
    ent_score = _pairwise_max_cos(Eq, Ef)
    rel_score = _pairwise_max_cos(Rq, Rf)
    return w_ent * ent_score + w_rel * rel_score




# --- tiny utilities for rerank ---


# --- scorer with attention row-weights---

# used cached chunk embeddingsß



######################## gpu boosted ver for coarse_filter



# -----------------------------------------------------------



def get_all_results(top_m_results,codebook_main,target = 'facts'):
    all_results = []
    chunk_dict = {}
    all_indexes = []
    feat_name = target+'_lst'
    for i in range(len(top_m_results["score"])):
        row = {key: top_m_results[key][i] for key in top_m_results}
        cur_qid,cur_qjd = row['index_combo']
        if cur_qid in chunk_dict.keys():
            chunk_dict[cur_qid].append(cur_qjd)
        else:
            chunk_dict[cur_qid] = [cur_qjd]

    feat_data = codebook_main[feat_name]
    n_groups = len(feat_data)
    for q_idx, q_jdx_lst in chunk_dict.items():
        if q_idx >= n_groups:
            continue
        chunks = []
        group = feat_data[q_idx]
        for q_jdx in q_jdx_lst:
            if q_jdx >= len(group):
                continue
            chunks.append(group[q_jdx])

        if chunks:
            all_results.append(chunks)
            all_indexes.append(q_idx)

    return all_results,all_indexes

def get_all_results_entire_chunk(top_m_results,codebook_main,target = 'facts'):
    all_results = []
    chunk_dict = {}
    all_indexes = []
    feat_name = target+'_lst'
    for i in range(len(top_m_results["score"])):
        row = {key: top_m_results[key][i] for key in top_m_results}
        cur_qid,cur_qjd = row['index_combo']
        if cur_qid in chunk_dict.keys():
            chunk_dict[cur_qid].append(cur_qjd)
        else:
            chunk_dict[cur_qid] = [cur_qjd]

    feat_data = codebook_main[feat_name]
    n_groups = len(feat_data)
    for q_idx, q_jdx_lst in chunk_dict.items():
        if q_idx >= n_groups:
            continue
        all_results.append(feat_data[q_idx])
        all_indexes.append(q_idx)

    return all_results,all_indexes


# ============================================================
# Adaptive graph retrieval — per-triple routing + iterative hops
# No tunable weights, thresholds, or temperatures.
# ============================================================



## instead of using all top m answers, find the overlapped texts
def _all_contiguous_subseqs(seq, min_len=2):
    n = len(seq)
    for i in range(n):
        for j in range(i + min_len, n + 1):
            yield tuple(seq[i:j])

def _is_subrun(a, b):
    # Is tuple a a contiguous subrun of tuple b?
    if len(a) >= len(b): 
        return False
    L = len(a)
    return any(b[i:i+L] == a for i in range(len(b) - L + 1))

def common_contiguous_overlaps(answers_lst, min_len=2, min_support_ratio=0.7, mode="advanced"):
    if mode == "advanced":
        return common_contiguous_overlaps_advanced(answers_lst, min_len, min_support_ratio)
    elif mode == "naive":
        return common_contiguous_overlaps_naive(answers_lst, min_len=2)
    elif mode == "hash":
        return common_contiguous_overlaps_hash(answers_lst, min_len=2)


def get_unique_knowledge_naive(overlapped_answers,flat_answers_lsts):
    """
    For each overlap run in overlapped_answers['overlaps'], keep that run only
    in the 'owner' sequence (the one with the longest continuation after the run),
    and remove the run from all other sequences where it appears.

    Inputs:
    overlapped_answers: {'overlaps': [[edges_index, edges_index,...],...]}

    flat_answers_lsts: [[edges_index,...],...] ; get from get_flat_answers_lsts(answers_lsts)
    """

    # Normalize inputs
    out_answers: List[List[int]] = [list(map(int, seq)) for seq in flat_answers_lsts]
    runs: List[List[int]] = [list(map(int, run)) for run in overlapped_answers.get("overlaps", [])]

    def find_run_positions(run: List[int], seq: List[int]) -> List[int]:
        L = len(run)
        if L == 0 or L > len(seq):
            return []
        return [i for i in range(len(seq) - L + 1) if seq[i:i + L] == run]

    def remove_all_runs(seq: List[int], run: List[int]) -> List[int]:
        """Remove all non-overlapping occurrences of run from seq (greedy left-to-right)."""
        res: List[int] = []
        i = 0
        L = len(run)
        n = len(seq)
        while i <= n - L:
            if seq[i:i+L] == run:
                i += L  # skip the run
            else:
                res.append(seq[i])
                i += 1
        # append trailing tail
        res.extend(seq[i:])
        return res

    # Process longer overlaps first to avoid smaller runs interfering
    runs_sorted = sorted(runs, key=len, reverse=True)

    assignments = []
    for run in runs_sorted:
        # Find occurrences in each sequence
        occs: Dict[int, List[int]] = {idx: find_run_positions(run, seq) for idx, seq in enumerate(out_answers)}
        present = {i: pos for i, pos in occs.items() if pos}
        if not present:
            continue  # this run doesn't appear anywhere

        # Choose owner: sequence with the maximum tail length after the best occurrence
        L = len(run)
        owner = None
        best_tail = -1
        best_total_len = -1
        for i, positions in present.items():
            for pos in positions:
                tail_len = len(out_answers[i]) - (pos + L)
                # Tie-breakers: longer total sequence length, then smaller index
                if (tail_len > best_tail or
                    (tail_len == best_tail and len(out_answers[i]) > best_total_len) or
                    (tail_len == best_tail and len(out_answers[i]) == best_total_len and (owner is None or i < owner))):
                    owner = i
                    best_tail = tail_len
                    best_total_len = len(out_answers[i])

        # Remove this run from all non-owner sequences where it occurs
        for j in range(len(out_answers)):
            if j != owner and occs.get(j):
                out_answers[j] = remove_all_runs(out_answers[j], run)

        assignments.append({
            'run': run,
            'owner': owner,
            'occurrences': {i: occs[i] for i in present}
        })


    return {'assignments': assignments, 'out_answers': out_answers}

# --- Optimized Unique Knowledge ---
def get_unique_knowledge_efficient(overlapped_answers, flat_answers_lsts):
    out_answers: List[List[int]] = [list(map(int, seq)) for seq in flat_answers_lsts]
    runs: List[List[int]] = [list(map(int, run)) for run in overlapped_answers.get("overlaps", [])]

    runs_sorted = sorted(runs, key=len, reverse=True)
    assignments = []

    for run in runs_sorted:
        L = len(run)
        if L == 0: continue

        # Find occurrences
        occs = {}
        for idx, seq in enumerate(out_answers):
            positions = [i for i in range(len(seq) - L + 1) if seq[i:i+L] == run]
            if positions:
                occs[idx] = positions

        if not occs: continue

        # Pick owner
        owner, best_tail, best_len = None, -1, -1
        for i, positions in occs.items():
            for pos in positions:
                tail_len = len(out_answers[i]) - (pos + L)
                if (tail_len > best_tail or
                   (tail_len == best_tail and len(out_answers[i]) > best_len) or
                   (tail_len == best_tail and len(out_answers[i]) == best_len and (owner is None or i < owner))):
                    owner, best_tail, best_len = i, tail_len, len(out_answers[i])

        # Remove from non-owners
        for j in occs:
            if j != owner:
                new_seq, skip = [], 0
                seq = out_answers[j]
                for k in range(len(seq)):
                    if skip: skip -= 1; continue
                    if seq[k:k+L] == run:
                        skip = L - 1
                        continue
                    new_seq.append(seq[k])
                out_answers[j] = new_seq

        assignments.append({"run": run, "owner": owner, "occurrences": occs})

    return {"assignments": assignments, "out_answers": out_answers}

from typing import List, Dict, Any

def get_unique_knowledge_advanced(overlapped_answers, flat_answers_lsts,
                                  alpha=1.0, beta=0.5, gamma=0.5):
    """
    Smarter version: assign each overlap run to exactly ONE owner sequence
    using a weighted scoring function.
    score=α⋅tail_len + β⋅seq_len + γ⋅frequency (frequency: how many times this run appears in the sequence)
    high tail_len -> more following description
    high seq_len -> more general information
    high frequency -> likely closer description of the overlapped_answers
    """
    out_answers: List[List[int]] = [list(map(int, seq)) for seq in flat_answers_lsts]
    runs: List[List[int]] = [list(map(int, run)) for run in overlapped_answers.get("overlaps", [])]

    def find_run_positions(run: List[int], seq: List[int]) -> List[int]:
        L = len(run)
        if L == 0 or L > len(seq):
            return []
        return [i for i in range(len(seq) - L + 1) if seq[i:i + L] == run]

    def remove_all_runs(seq: List[int], run: List[int]) -> List[int]:
        """Remove all non-overlapping occurrences of run from seq (greedy left-to-right)."""
        res, i, L, n = [], 0, len(run), len(seq)
        while i <= n - L:
            if seq[i:i+L] == run:
                i += L
            else:
                res.append(seq[i]); i += 1
        res.extend(seq[i:])
        return res

    runs_sorted = sorted(runs, key=len, reverse=True)
    assignments = []

    for run in runs_sorted:
        occs: Dict[int, List[int]] = {idx: find_run_positions(run, seq) for idx, seq in enumerate(out_answers)}
        present = {i: pos for i, pos in occs.items() if pos}
        if not present:
            continue

        # --- scoring ---
        best_score, owner = -1e9, None
        for i, positions in present.items():
            seq = out_answers[i]
            seq_len = len(seq)
            freq = len(positions)
            for pos in positions:
                tail_len = seq_len - (pos + len(run))
                score = alpha * tail_len + beta * seq_len + gamma * freq
                if score > best_score:
                    best_score, owner = score, i

        # remove from all non-owner
        for j in range(len(out_answers)):
            if j != owner and occs.get(j):
                out_answers[j] = remove_all_runs(out_answers[j], run)

        assignments.append({'run': run, 'owner': owner, 'score': best_score})

    return {'assignments': assignments, 'out_answers': out_answers}


def get_flat_answers_lsts(answers_lsts):    
    return [[x for group in bucket for x in (group if isinstance(group, (list, tuple)) else [group])] for bucket in answers_lsts]


def find_overlapped_answers(answers_lsts):
    flat_answers_lsts = get_flat_answers_lsts(answers_lsts)
    # default 2 for the overlap
    final_flat_answers_lsts = common_contiguous_overlaps(flat_answers_lsts,2, mode="advanced")
    return final_flat_answers_lsts


def find_overlapped_thinkings(all_q_indices,codebook_main):
    # from q indices to get answers

    selected_thinkings_lsts = []
    questions_to_thinkings_dict = codebook_main['questions_to_thinkings']

    for q_index in all_q_indices:
        if q_index in questions_to_thinkings_dict.keys():
            selected_thinkings_lsts.append(codebook_main['thinkings_lst'][questions_to_thinkings_dict[q_index]])


    if selected_thinkings_lsts:
        flat_thinkings_lsts = get_flat_answers_lsts(selected_thinkings_lsts)
        # default 2 for the overlap
        final_flat_thinkings_lsts = common_contiguous_overlaps(flat_thinkings_lsts,2, mode="advanced")
    else:
        final_flat_thinkings_lsts = selected_thinkings_lsts

    return final_flat_thinkings_lsts

##### new function for getting selected_thinkings_lsts
def get_thinkings_lsts(all_q_indices,codebook_main):
    selected_thinkings_lsts = []
    questions_to_thinkings_dict = codebook_main['questions_to_thinkings']

    for q_index in all_q_indices:
        if q_index in questions_to_thinkings_dict.keys():
            selected_thinkings_lsts.append(codebook_main['thinkings_lst'][questions_to_thinkings_dict[q_index]])

    return selected_thinkings_lsts


def _list_from_index_map(index_map: Dict[str, int]) -> List[str]:
    """把 {item -> idx} 映射还原为按 idx 排序的列表"""
    out = [None] * len(index_map)
    for item, idx in index_map.items():
        out[idx] = item
    return out

def decode_chain_to_text(edge_idx_chain: List[int], codebook_main: Dict[str, Any]) -> str:
    """
    把一条 answers 的边索引链解码成可读文本（用与问题相同的线性化逻辑）。
    """
    return make_question_text(edge_idx_chain, codebook_main)

def decode_answers_bucket_to_texts(answers_bucket, codebook_main: Dict[str, Any]) -> List[str]:
    """
    answers_bucket 的结构来源于 add_answers_to_filtered_lst 组装的
    item['answers(edges[i])']：通常是一个“答案候选集合”，内部是多条“边索引链”。
    这个函数把它统一解码成一批可读短句，便于展示。
    """
    texts = []
    # answers_bucket 可能是 List[List[int]]（多条链），也可能包含更深的嵌套
    # 这里尽量稳健地拍平一层
    if isinstance(answers_bucket, (list, tuple)):
        for maybe_chain in answers_bucket:
            if isinstance(maybe_chain, (list, tuple)) and len(maybe_chain) > 0 and isinstance(maybe_chain[0], int):
                # 单条边索引链
                texts.append(decode_chain_to_text(maybe_chain, codebook_main))
            elif isinstance(maybe_chain, (list, tuple)):
                # 可能是更深一层的嵌套
                for chain in maybe_chain:
                    if isinstance(chain, (list, tuple)) and len(chain) > 0 and isinstance(chain[0], int):
                        texts.append(decode_chain_to_text(chain, codebook_main))
    return texts


#### get the all unique knowledge

def get_unique_knowledge(overlapped_answers,flat_answers_lsts,alpha=1.0, beta=0.5, gamma=0.5, mode="advanced"):
    if mode == "naive":
        return get_unique_knowledge_naive(overlapped_answers,flat_answers_lsts)
    elif mode == "advanced":
        return get_unique_knowledge_advanced(overlapped_answers, flat_answers_lsts, alpha, beta, gamma)
    elif mode == "efficient":
        return get_unique_knowledge_efficient(overlapped_answers, flat_answers_lsts)


    return {'assignments': assignments, 'out_answers': out_answers}

def find_unique_thinkings(all_q_indices, codebook_main):
    selected_thinkings_lsts = []
    questions_to_thinkings_dict = codebook_main['questions_to_thinkings']

    for q_index in all_q_indices:
        if q_index in questions_to_thinkings_dict.keys():
            selected_thinkings_lsts.append(
                codebook_main['thinkings_lst'][questions_to_thinkings_dict[q_index]]
            )

    if selected_thinkings_lsts:
        flat_thinkings_lsts = get_flat_answers_lsts(selected_thinkings_lsts)
        overlapped_runs = common_contiguous_overlaps(flat_thinkings_lsts, 2, mode="advanced")  # list
        unique_dict = get_unique_knowledge({'overlaps': overlapped_runs},     # ✅ wrap
                                           flat_thinkings_lsts, mode="advanced")
        uniqie_thinkings = unique_dict['out_answers']
    else:
        uniqie_thinkings = selected_thinkings_lsts

    return uniqie_thinkings


# add find unique answers

def find_unique_answers(answers_lsts):
    flat_answers_lsts = get_flat_answers_lsts(answers_lsts)
    overlapped_runs = common_contiguous_overlaps(flat_answers_lsts, 2, mode="advanced")   # list[list[int]]
    unique_dict = get_unique_knowledge({'overlaps': overlapped_runs},    # ✅ wrap
                                       flat_answers_lsts, mode="advanced")
    uniqie_answers = unique_dict['out_answers']
    return uniqie_answers


# get the entities from codebook_main
def get_json_with_given_knowledge(flat_answers_lsts,codebook_main,codebook_sub_q,decode = True):
    # used flat here since trying to flat answers for each answers trunk to get longer overlapp
    # if change the answers here also change the format for other func related

    # get all unique edges

    all_unique_edges_mat_indexes = list(set([x for sublist in flat_answers_lsts for x in sublist]))

    # find all unique entities and r
    entitie_set = []
    r_set = []
    entitie_index_set = []
    r_index_set = []
    entitie_index_dict = {}
    r_index_dict = {}
    edge_matrix_sub = []
    edge_mat_index_dict = {}

    new_edge_mat_index = 0 
    
    # build new edge_mat_index
    for edge_mat_index in all_unique_edges_mat_indexes:
        edge = codebook_main['edge_matrix'][edge_mat_index]
        e_index1,r_index,e_index2 = edge
        entitie_index_set.append(e_index1)
        entitie_index_set.append(e_index2)
        r_index_set.append(r_index)
        edge_matrix_sub.append(edge)
        edge_mat_index_dict[edge_mat_index] = new_edge_mat_index
        new_edge_mat_index+=1

    # update edge index in flat_answers_lsts
    flat_answers_lsts = [[edge_mat_index_dict.get(x, x) for x in sublist] for sublist in flat_answers_lsts]

    # build new entities index and relations index

    entitie_index_set = list(set(entitie_index_set))
    r_index_set = list(set(r_index_set))

    new_ent_index = 0
    new_r_index = 0

    for ent_index in entitie_index_set:

        entitie_set.append(codebook_main['e'][ent_index])
        entitie_index_dict[ent_index] = new_ent_index
        new_ent_index+=1

    for r_index in r_index_set:
        r_set.append(codebook_main['r'][r_index])
        r_index_dict[r_index] = new_r_index
        new_r_index+=1

    # update ent index and r index for the edge_matrix_sub
    def remap_edges(edges: List[List[int]], e_dict: Dict[int, int], r_dict: Dict[int, int]) -> List[List[int]]:
        """
        Remap edges of format [[e, r, e], ...] using given entity and relation mappings.

        Parameters
        ----------
        edges : List[List[int]]
            List of edges in format [entity1, relation, entity2].
        e_dict : Dict[int, int]
            Mapping dictionary for entity indices (applies to positions 0 and 2).
        r_dict : Dict[int, int]
            Mapping dictionary for relation indices (applies to position 1).

        Returns
        -------
        List[List[int]]
            New edges with remapped indices.
        """
        mapped_edges = []
        for e1, r, e2 in edges:
            new_e1 = e_dict.get(e1, e1)  
            new_r  = r_dict.get(r, r)
            new_e2 = e_dict.get(e2, e2)
            mapped_edges.append([new_e1, new_r, new_e2])
        return mapped_edges
    
    edge_matrix_sub = remap_edges(edge_matrix_sub, entitie_index_dict, r_index_dict)

    entitie_index_dict_q = {}
    r_index_dict_q = {}
    entitie_set_len = len(entitie_set)
    r_set_len = len(r_set)
    # do the samilar steps for combine the sub codebook and sub q codebook

    # update the entities index and relation index for questions and combine the entities lst and relations lst

    # update the entities index
    ent_pos = 0
    for ent in codebook_sub_q['e']:
        # check the ent in entities_lst or not
        if ent in entitie_set:
            new_ent_pos = entitie_set.index(ent)
        else:
            new_ent_pos = entitie_set_len
            entitie_set.append(ent)
            entitie_set_len+=1

        entitie_index_dict_q[ent_pos] = new_ent_pos

        ent_pos+=1

    # update relation index
    r_pos = 0
    for r in codebook_sub_q['r']:
        if r in r_set:
            new_r_pos = r_set.index(r)
        else:
            new_r_pos = r_set_len
            r_set.append(r)
            r_set_len+=1


        r_index_dict_q[r_pos] = new_r_pos

        r_pos+=1

    # map the q edge matrix
    edge_mat_for_q_sub = remap_edges(codebook_sub_q['edges([e,r,e])'], entitie_index_dict_q, r_index_dict_q)

    # update the edges
    edge_matrix_sub_len = len(edge_matrix_sub)
    edge_pos = 0
    edge_mat_for_q_sub_dict = {}

    for edge in edge_mat_for_q_sub:
        if edge in edge_matrix_sub:
            new_edge_pos = edge_matrix_sub.index(edge)
        else:
            new_edge_pos = edge_matrix_sub_len
            edge_matrix_sub.append(edge)
            edge_matrix_sub_len+=1

        edge_mat_for_q_sub_dict[edge_pos] = new_edge_pos


    # update the questions
    if DEBUG_VERBOSE:
        print(edge_mat_for_q_sub_dict)
        print(codebook_sub_q)
        print(codebook_sub_q['questions(edges[i])'])
    questions = [
        [edge_mat_for_q_sub_dict.get(val, val) for val in inner]
        for inner in codebook_sub_q['questions(edges[i])']
    ]


    # get the final merged json

       
    final_merged_json = {
        'e':entitie_set,
        'r':r_set,
        'edge_matrix':edge_matrix_sub,
        'questions(edges[i])':questions,
        'given knowledge(edges[i])': flat_answers_lsts,
        'rule':codebook_sub_q['rule']
    }

    if decode:
        final_merged_json = {
            'e':entitie_set,
            'r':r_set,
            'edge_matrix':edge_matrix_sub,
            'questions([[e,r,e], ...])':decode_questions(questions, final_merged_json, 'edges'),
            'given knowledge([[e,r,e], ...])': decode_questions(flat_answers_lsts, final_merged_json, 'edges'),
            'rule':codebook_sub_q['rule']

        }

    return final_merged_json


#### get_json_with_given_knowledge with thinkings



### CompressRag RL version

###### adding slicing func

rule_ere_exact = dedent("""\
    ---Knowledge Base---
    [JSON format]
    - e: list of entities (e[i] = entity string)
    - r: list of relations (r[j] = relation string)
    - edge_matrix: [[head_e_idx, r_idx, tail_e_idx]]
        * NOTE: edges[i] is just shorthand for edge_matrix[i]
    - questions(edges[i]): questions linked by edge i
    - given knowledge(edges[i]): prior answers linked by edge i
    - start thinking with(edges[i]): reasoning steps linked by edge i
    - facts(edges[i]): factss linked by edge i
""")

rule_edge_exact = dedent("""\
    ---Knowledge Base---
    [JSON format]
    - e: list of entities (e[i] = entity string)
    - r: list of relations (r[j] = relation string)
    - [e,r,e]: triple [head_e_idx, r_idx, tail_e_idx]
    - questions([[e,r,e], ...]): question triples 
    - given knowledge([[e,r,e], ...]): prior answer triples
    - start thinking with([[e,r,e], ...]): reasoning steps
    - facts([[e,r,e], ...]): fact triples
""")
                         
rule_words = dedent("""\
    ---Knowledge Base---
    [JSON format]
    - questions(words): question triples 
    - given knowledge(words): prior answer triples
    - start thinking with(words): reasoning triples
    - facts(words): fact triples
""")

def _approx_token_len(obj: Any) -> int:
    """
    Cheap token-length proxy: JSON length // 4.
    Keeps this util self-contained without tiktoken.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback if non-serializable bits exist
        s = str(obj)
    return max(1, len(s) // 4)

def _drop_keys(d: Dict[str, Any], matchers) -> Dict[str, Any]:
    """
    Return a shallow-pruned copy of dict `d` dropping any key that matches any matcher.
    A matcher can be:
      - exact string (k == matcher)
      - callable (matcher(k) -> bool)
    """
    out = {}
    for k, v in d.items():
        drop = False
        for m in matchers:
            if callable(m):
                if m(k):
                    drop = True
                    break
            else:
                if k == m:
                    drop = True
                    break
        if not drop:
            out[k] = v
    return out


# decode ere or edge into words
def decode_into_words_for_ere_and_edge(final_merged_json: Dict[str, Any],format : str = 'edge',choices = []):
  final_merged_json_copy = deepcopy(final_merged_json)
  all_transformed_feats = []

  if format == 'edge':
    for choice in choices:
      if choice in final_merged_json_copy:
        trans_formed_choice = decode_questions(final_merged_json_copy[choice], final_merged_json_copy, fmt='words')
        final_merged_json_copy[choice.split("(")[0]+'words'] = trans_formed_choice
        all_transformed_feats.append(choice.split("(")[0]+'words')
        final_merged_json_copy.pop(choice)


  elif format == 'ere':
    def _triples_to_words(triples, cb):
      E, R = cb["e"], cb["r"]
      return [[E[h], R[r], E[t]] for (h, r, t) in triples]

    for choice in choices:
      trans_formed_choice = []
      if choice in final_merged_json_copy:
        for triples in final_merged_json_copy[choice]:
          trans_formed_choice.append(_triples_to_words(triples, final_merged_json_copy))
        final_merged_json_copy[choice.split("(")[0]+'words'] = trans_formed_choice
        all_transformed_feats.append(choice.split("(")[0]+'words')
        final_merged_json_copy.pop(choice)

  return final_merged_json_copy,all_transformed_feats


def slice_for_final_merged_json(final_merged_json: Dict[str, Any],use_word_format : bool = True) -> Dict[str, Any]:
    """
    Produce a sliced view of `final_merged_json`:

    - format == 'edge_matrix': keep edge-matrix-centric fields (drop ERE/edge-list blocks).
    - format == 'ere': keep E-R-E list-style fields (drop the matrix + per-index blocks).
    - format is None/other: return whichever of the two slices is shorter (approx token count).

    This function is defensive against small key-name variations you've used:
      * 'questions([[e,r,e], ...])'
      * 'given knowledge([[e,r,e], ...])'
      * 'facts([[e,r,e], ...])'
      * 'questions(edges[i])'
      * 'given knowledge(edges[i])'
      * 'facts(edges[i])'
      * 'edge_matrix'

    Env: FORCE_SLICE_WORDS=1 → always return word format (skip token comparison).
    """
    _force_slice_words = _os.environ.get("FORCE_SLICE_WORDS", "0") == "1"
    data = dict(final_merged_json)  # shallow copy: only top-level keys are modified

    # --- Define key matchers -------------------------------------------------
    # Keys that correspond to ERE (triple-list) style content
    ere_exact = [
        'questions([[e,r,e], ...])',
        'given knowledge([[e,r,e], ...])',
        'facts([[e,r,e], ...])',
        'start thinking with([[e,r,e], ...])'
    ]
    # Keys that correspond to per-edge indexed content (edges[i])

    per_edge_exact = [
        'questions(edges[i])',
        'given knowledge(edges[i])',
        'facts(edges[i])',
        'start thinking with(edges[i])'
    ]

    edge_feats = [
        'questions(edges[i])',
        'given knowledge(edges[i])',
        'facts(edges[i])',
        'start thinking with(edges[i])'
    ]

    # check if ready to build which one (some will include missing vals, we will choose the one more compelted)
    features_contain_ere_exact = 1 
    for e in ere_exact:
      if e in data:
        features_contain_ere_exact += 1

    features_contain_per_edge_exact = 0 
    for e in per_edge_exact:
      if e in data:
        features_contain_per_edge_exact += 1

    # missing edge matrix cannot do edges[i] format
    # ere format
    if 'edge_matrix' not in data:
      data['rule'] = rule_edge_exact
      final_format = _drop_keys(data, per_edge_exact)

      # get word format and transformed feats
      if use_word_format:
        word_format,all_transformed_feats = decode_into_words_for_ere_and_edge(final_format,format = 'ere',choices =ere_exact )

        # check improvement in tokens
        word_format['rule']  = rule_words
        if _force_slice_words or _approx_token_len(word_format) < _approx_token_len(final_format):

          return word_format

      return final_format

    # ere format
    # ere format does not need edge_matrix
    elif features_contain_ere_exact > features_contain_per_edge_exact:
      data['rule'] = rule_edge_exact
      final_format = _drop_keys(data, per_edge_exact)
      if 'edge_matrix' in final_format:
        final_format.pop('edge_matrix')

      if use_word_format:
        word_format,all_transformed_feats = decode_into_words_for_ere_and_edge(final_format,format = 'ere',choices =ere_exact )

        # check improvement in tokens
        word_format['rule']  = rule_words
        word_format = {k: word_format[k] for k in all_transformed_feats if k in word_format}

        if _force_slice_words or _approx_token_len(word_format) < _approx_token_len(final_format):

          return word_format

      return final_format

    # edge format
    elif features_contain_ere_exact > features_contain_per_edge_exact:
      data['rule'] =  rule_ere_exact
      final_format = _drop_keys(data, rule_ere_exact)

      if use_word_format:
        word_format,all_transformed_feats = decode_into_words_for_ere_and_edge(final_format,format = 'edge',choices =edge_feats )
        # check improvement in tokens
        word_format['rule']  = rule_words
        word_format = {k: word_format[k] for k in all_transformed_feats if k in word_format}

        if _force_slice_words or _approx_token_len(word_format) < _approx_token_len(final_format):

          return word_format

      return final_format
    
    else:
      edge_matrix_view = _drop_keys(data, ere_exact)
      edge_matrix_view['rule'] =  rule_ere_exact

      ere_view = _drop_keys(data, per_edge_exact)
      ere_view['rule'] =  rule_edge_exact

      if _approx_token_len(edge_matrix_view) <= _approx_token_len(ere_view):

        if use_word_format:

          word_format,all_transformed_feats = decode_into_words_for_ere_and_edge(edge_matrix_view,format = 'edge',choices =edge_feats )
          word_format['rule']  = rule_words
          word_format = {k: word_format[k] for k in all_transformed_feats if k in word_format}

          if _force_slice_words or _approx_token_len(word_format) < _approx_token_len(edge_matrix_view):
            return word_format

        return edge_matrix_view
      else:

        if use_word_format:

          word_format,all_transformed_feats = decode_into_words_for_ere_and_edge(ere_view,format = 'ere',choices =ere_exact)
          word_format['rule']  = rule_words
          word_format = {k: word_format[k] for k in all_transformed_feats if k in word_format}

          if _force_slice_words or _approx_token_len(word_format) < _approx_token_len(ere_view):
            return word_format

        return ere_view
      

def remove_duplicate_inner_lists(lst, seen=None):
    if seen is None:
        seen = set()

    if isinstance(lst, list) and all(isinstance(x, int) for x in lst):
        tup = tuple(lst)
        if tup in seen:
            return None  # mark as duplicate
        seen.add(tup)
        return lst

    elif isinstance(lst, list):
        new_list = []
        for x in lst:
            res = remove_duplicate_inner_lists(x, seen)
            if res is not None:
                new_list.append(res)
        return new_list

    else:
        return lst
    

# def dedup_facts_and_embeddings(
#     facts: Any,
#     embeds: Any,
#     seen: Optional[Set[Tuple[int, ...]]] = None,
# ):
#     """
#     Deduplicate *inner lists of ints* in `facts` and drop the aligned entries in `embeds`.
#     Assumes `facts` and `embeds` have the same nested list structure.
#     """
#     if seen is None:
#         seen = set()

#     # Leaf case: facts is a list of ints -> dedupe unit
#     if isinstance(facts, list) and all(isinstance(x, int) for x in facts):
#         key = tuple(facts)
#         if key in seen:
#             return None, None
#         seen.add(key)
#         return facts, embeds

#     # Nested list case: walk in parallel
#     if isinstance(facts, list):
#         if not isinstance(embeds, list):
#             raise ValueError(f"Structure mismatch: facts is list but embeds is {type(embeds)}")

#         if len(facts) != len(embeds):
#             raise ValueError(f"Length mismatch: len(facts)={len(facts)} vs len(embeds)={len(embeds)}")

#         new_facts = []
#         new_embeds = []
#         for f_item, e_item in zip(facts, embeds):
#             f_res, e_res = dedup_facts_and_embeddings(f_item, e_item, seen)
#             if f_res is not None:
#                 new_facts.append(f_res)
#                 new_embeds.append(e_res)
#         return new_facts, new_embeds

#     # Non-list facts: keep as-is (and embed as-is)
#     return facts, embeds

def dedup_facts_and_embeddings(
    facts: Any,
    embeds: Any,
    seen: Optional[Set[Tuple[int, ...]]] = None,
):
    if seen is None:
        seen = set()

    # Leaf case: facts is a list of ints -> dedupe unit
    if isinstance(facts, list) and all(isinstance(x, int) for x in facts):
        key = tuple(facts)
        if key in seen:
            return None, None
        seen.add(key)
        return facts, embeds

    # Nested list case: walk in parallel
    if isinstance(facts, list):
        # Allow embeds to be either a list or a numpy array
        if not isinstance(embeds, (list, np.ndarray)):
            raise ValueError(f"Structure mismatch: facts is list but embeds is {type(embeds)}")

        if len(facts) != len(embeds):
            raise ValueError(f"Length mismatch: len(facts)={len(facts)} vs len(embeds)={len(embeds)}")

        new_facts = []
        new_embeds = []
        for f_item, e_item in zip(facts, embeds):
            f_res, e_res = dedup_facts_and_embeddings(f_item, e_item, seen)
            if f_res is not None:
                new_facts.append(f_res)
                new_embeds.append(e_res)

        # Preserve numpy array output if input was numpy
        if isinstance(embeds, np.ndarray) and new_embeds:
            return new_facts, np.array(new_embeds)
        return new_facts, new_embeds

    # Non-list facts: keep as-is
    return facts, embeds
      
# thinkings extraction choice: keep the overlap(default), not include the thinking, keep the unique thinking
# answers extraction choice: keep the unique (default), not include the answers, keep the overlap
# combine ents choice: not combine, combine per round, combine per 3 round
thinkings_choice = ['overlap','unique','not_include']
answers_choice = ['overlap','unique','not_include']
facts_choice = ['unique','include_all']
combine_ents_choice = [0,1,2]




not_recorded_answers = ['The answer is not provided in the given context','not provided in the given context', 'not in the given context']

NO_ANSWER_PATTERNS = [
    r"\b(not\s+provided|not\s+in|absent)\b.*\b(context|given text|passage|documents)\b",
    r"\b(cannot|can't|unable to|insufficient)\b.*\b(answer|determin(e|e)|find)\b",
    r"\b(no|not enough|insufficient)\b.*\b(information|evidence|details)\b",
    r"\b(unknown|unclear|ambiguous)\b",
    r"\b(answer|information)\b.*\b(not|never)\b.*\b(provided|available|present)\b",
    r"\b(the answer is not provided in the (?:given )?context)\b",
]

NO_ANSWER_RE = re.compile("|".join(NO_ANSWER_PATTERNS), flags=re.I)

def is_no_answer_text(s: str) -> bool:
    s = (s or "").strip()
    # normalize punctuation/whitespace
    s_norm = re.sub(r"\s+", " ", s)
    return bool(NO_ANSWER_RE.search(s_norm))


class AutoPrunedRetriver:
    def __init__(
        self,
        ini_meta_codebook = {},
        sentence_emb: Optional[Embeddings] = None,
        word_emb: Optional[Embeddings] = None,
        llm = None,
        thinkings_choice = 'not_include',
        answers_choice = 'overlap',
        facts_choice = 'include_all',
        use_word = False,
        top_m = 5,
        top_k = 20,
        combine_ent_sim = 0.9,
        q_combine_sim = 0.9,
        aft_combine_sim = 0.9,
        semantic_overlap_sim = 0.9,
        chunking_use = 'rebel',
        chunking_api = None,
    ):
        """
        thinkings_choice and answers_choice must be one of 'overlap','unique','not_include'
        combine_ents_rounds must be interger-> how many rounds after combine ents

        
        """

        # meta
        # start with empty codebook
        self.meta_codebook = ini_meta_codebook
        self.llm = llm
        self.cur_fact_context = None
        self.use_word = use_word
        self.chunking_use = chunking_use
        self.chunking_api = chunking_api

        # Embeddings
        self.sentence_emb = sentence_emb 
        self.word_emb = word_emb 

        #coarse filter params
        self.top_k = top_k
        self.top_m = top_m
        self.question_batch_size = 1
        self.questions_db_batch_size = 1
        self.custom_linearizer = None


        # combine ents
        self.min_exp_num =2
        self.max_exp_num = 10
        self.k_grid_size = 8
        self.sample_size_prop = 20

        self.combine_ent_sim = combine_ent_sim
        self.q_combine_sim = q_combine_sim
        self.aft_combine_sim = aft_combine_sim



        # params for dpo
        # ### ents param
        # self.combine_ents_rounds = combine_ents_rounds
        # self.round = 1

        ###### Extraction params
        self.semantic_overlap_sim = semantic_overlap_sim

        ### thinkings param
        self.thinkings_choice = thinkings_choice
        if thinkings_choice == "not_include":
            self.include_thinkings = False
        else:
            self.include_thinkings = True
            if self.thinkings_choice == "overlap":
                self.thinking_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,sim_threshold=self.semantic_overlap_sim)
            elif self.thinkings_choice == "unique":
                self.thinking_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,unique=True,sim_threshold=self.semantic_overlap_sim)

        self.llm.include_thinkings = self.include_thinkings
        ### answers param
        self.answers_choice   = answers_choice
        if answers_choice == "not_include":
            self.include_answers = False
        else:
            self.include_answers = True
            if self.answers_choice == "overlap":
                self.answers_extract_function =  partial(get_unique_or_overlap_by_sentence_embedded,sim_threshold=self.semantic_overlap_sim)
            elif self.answers_choice == "unique":
                self.answers_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,unique=True,sim_threshold=self.semantic_overlap_sim)


        # facts params
        self.facts_choice   = facts_choice
        if facts_choice == "not_include":
            self.include_facts = False
        else:
            self.include_facts = True
            if self.facts_choice == "overlap":
                self.facts_extract_function =  partial(get_unique_or_overlap_by_sentence_embedded,sim_threshold=self.semantic_overlap_sim)
            elif self.facts_choice == "unique":
                self.facts_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,unique=True,sim_threshold=self.semantic_overlap_sim)

        ### context fact param
        self.context_json_path = None  
        self._facts_preloaded = False 

    def set_include_thinkings(self):
        if self.thinkings_choice == "not_include":
            self.include_thinkings = False
            self.llm.include_thinkings = False

        else:
            self.include_thinkings = True
            self.llm.include_thinkings = True

            if self.thinkings_choice == "overlap":
                self.thinking_extract_function =  partial(get_unique_or_overlap_by_sentence_embedded,sim_threshold=self.semantic_overlap_sim)
            elif self.thinkings_choice == "unique":
                self.thinking_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,unique=True,sim_threshold=self.semantic_overlap_sim)

    def set_include_answers(self):
        if self.answers_choice == "not_include":
            self.include_answers = False

        else:
            self.include_answers = True
            if self.answers_choice == "overlap":
                self.answers_extract_function =  partial(get_unique_or_overlap_by_sentence_embedded,sim_threshold=self.semantic_overlap_sim)
            elif self.answers_choice == "unique":
                self.answers_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,unique=True,sim_threshold=self.semantic_overlap_sim)


    def set_include_facts(self):
        if self.facts_choice == "not_include":
            self.include_facts = False

        else:
            self.include_facts = True
            if self.facts_choice == "overlap":
                self.facts_extract_function =  partial(get_unique_or_overlap_by_sentence_embedded,sim_threshold=self.semantic_overlap_sim)
            elif self.facts_choice == "unique":
                self.facts_extract_function = partial(get_unique_or_overlap_by_sentence_embedded,unique=True,sim_threshold=self.semantic_overlap_sim)


    def set_includings(self):
        self.set_include_thinkings()
        self.set_include_answers()
        self.set_include_facts()

    # def preload_context_json(self, json_path: str, chunk_tokens: int = 1200, overlap_tokens: int = 100, sub_chunk_chars: int = 300, sub_chunk_overlap: int = 50, tokenizer_name: str = "gpt-4o-mini", subchunk_batch: int = 500):
    #     import json
    #     import tiktoken

    #     with open(json_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)

    #     try:
    #         tokenizer = tiktoken.encoding_for_model(tokenizer_name)
    #     except KeyError:
    #         tokenizer = tiktoken.get_encoding("cl100k_base")

    #     def _chunk_text(text: str, *, chunk_tokens: int = 1200, overlap_tokens: int = 100, sub_chunk_chars: int = 300, sub_chunk_overlap: int = 50, tokenizer=tokenizer):
    #         text = (text or "").strip()
    #         if not text:
    #             return []
    #         tokens = tokenizer.encode(text)
    #         token_chunks = []
    #         step = max(1, chunk_tokens - overlap_tokens)
    #         i = 0
    #         while i < len(tokens):
    #             j = min(len(tokens), i + chunk_tokens)
    #             chunk_text = tokenizer.decode(tokens[i:j]).strip()
    #             if chunk_text:
    #                 token_chunks.append(chunk_text)
    #             if j == len(tokens):
    #                 break
    #             i += step

    #         def _sub_chunk_by_chars(text, chunk_size, overlap):
    #             if not text or len(text) <= chunk_size:
    #                 return [text] if text else []
    #             sub_chunks = []
    #             step = max(1, chunk_size - overlap)
    #             i = 0
    #             while i < len(text):
    #                 j = min(len(text), i + chunk_size)
    #                 sub_chunk = text[i:j].strip()
    #                 if sub_chunk:
    #                     sub_chunks.append(sub_chunk)
    #                 if j == len(text):
    #                     break
    #                 i += step
    #             return sub_chunks
    #         all_sub_chunks = []
    #         for token_chunk in token_chunks:
    #             all_sub_chunks.extend(_sub_chunk_by_chars(token_chunk, sub_chunk_chars, sub_chunk_overlap))
    #         return all_sub_chunks

    #     items = data if isinstance(data, list) else [data]
    #     all_chunks = []
    #     for item in items:
    #         ctx = (item.get("context") or "").strip()
    #         if ctx:
    #             item_chunks = _chunk_text(ctx, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, sub_chunk_chars=sub_chunk_chars, sub_chunk_overlap=sub_chunk_overlap)
    #             all_chunks.extend(item_chunks)

    #     if not all_chunks:
    #         return None

    #     total_chunks = len(all_chunks)
    #     batch_size = max(1, len(all_chunks) // subchunk_batch) if subchunk_batch > 0 else len(all_chunks)
    #     num_batches = (total_chunks + batch_size - 1) // batch_size
    #     print(f"[preload_context_json] Total chunks: {total_chunks}, batch_size={batch_size}, num_batches: {num_batches}")

    #     # combined = None
    #     batch_num = 0
    #     facts_codebook_lst = []
    #     for i in range(0, total_chunks, batch_size):

    #         batch_chunks = all_chunks[i:i+batch_size]
    #         fact_cb = get_code_book(
    #             batch_chunks,
    #             type='facts',
    #             rule="Store factual statements.",
    #             batch_size=1,
    #             parser_choice=self.chunking_use,
    #             api=self.chunking_api,
    #         )

    #         print(f'batch {batch_num} codebook is generated')

    #         facts_codebook_lst.append(fact_cb)


    #         # facts_codebook_lst.append(fact_cb)


    #         batch_num+=1

    #     return facts_codebook_lst
    

    def preload_context_json(
        self,
        json_path: str,
        *,
        # primary (sentence → token chunks)
        chunk_tokens: int = 1200,
        overlap_tokens: int = 100,
        tokenizer_name: str = "gpt-4o-mini",
        # secondary (sub-chunk) controls
        subchunk_mode: str = "chars",          # "chars" (original) or "tokens" (new)
        sub_chunk_chars: int = 300,
        sub_chunk_overlap: int = 50,
        sub_chunk_token_size: int = 256,       # used when subchunk_mode="tokens"
        sub_chunk_token_overlap: int = 50,     # used when subchunk_mode="tokens"
        # batching
        subchunk_batch: int = 500,
    ):
        import json, re
        import tiktoken

        # ---------- load ----------
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]

        # ---------- tokenizer ----------
        try:
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        # ---------- helpers ----------
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[。！？])\s*'

        def split_sentences(text: str):
            sents = re.split(sentence_pattern, text)
            return [s.strip() for s in sents if s.strip()]

        def force_split_by_tokens(sentence: str, max_tokens: int):
            ids = tokenizer.encode(sentence)
            if len(ids) <= max_tokens:
                return [sentence]
            parts = []
            start = 0
            while start < len(ids):
                end = min(start + max_tokens, len(ids))
                parts.append(tokenizer.decode(ids[start:end]))
                start = end
            return parts

        def chunk_by_tokens_with_overlap(sentences, chunk_tokens: int, overlap_tokens: int):
            """Token-aware chunking over sentence list with overlap (by tokens)."""
            # precompute token counts to avoid re-encoding repeatedly
            tok_counts = [len(tokenizer.encode(s)) for s in sentences]

            chunks = []
            cur, cur_tok = [], 0

            i = 0
            while i < len(sentences):
                s, t = sentences[i], tok_counts[i]

                # handle sentences longer than chunk size by force-splitting first
                if t > chunk_tokens:
                    # flush current before inserting forced splits
                    if cur:
                        chunks.append(" ".join(cur).strip())
                        cur, cur_tok = [], 0

                    long_parts = force_split_by_tokens(s, chunk_tokens)
                    # first n-1 parts are full chunks
                    for p in long_parts[:-1]:
                        chunks.append(p.strip())
                    # the last part can start a new working chunk
                    last_p = long_parts[-1].strip()
                    cur = [last_p] if last_p else []
                    cur_tok = len(tokenizer.encode(last_p)) if last_p else 0
                    i += 1
                    continue

                # normal packing
                if cur_tok + t > chunk_tokens and cur:
                    chunks.append(" ".join(cur).strip())

                    # build token-overlap from the tail of cur
                    if overlap_tokens > 0 and cur:
                        overlap_sents = []
                        overlap_tok = 0
                        # walk backward over *cur* using cached counts
                        j = len(cur) - 1
                        while j >= 0 and overlap_tok + len(tokenizer.encode(cur[j])) <= overlap_tokens:
                            overlap_sents.insert(0, cur[j])
                            overlap_tok += len(tokenizer.encode(cur[j]))
                            j -= 1
                        cur = overlap_sents
                        cur_tok = overlap_tok
                    else:
                        cur, cur_tok = [], 0

                cur.append(s)
                cur_tok += t
                i += 1

                # if exactly hits the limit, flush immediately
                if cur_tok == chunk_tokens:
                    chunks.append(" ".join(cur).strip())
                    cur, cur_tok = [], 0

            if cur:
                chunks.append(" ".join(cur).strip())
            return chunks

        def subchunk_by_chars(text: str, chunk_size: int, overlap: int):
            """Original char-based sub-chunking (sentence-aware)."""
            if not text:
                return []
            if len(text) <= chunk_size:
                return [text]

            sents = split_sentences(text) or [text]
            sub_chunks, cur, cur_chars = [], [], 0

            for s in sents:
                sc = len(s)
                if cur and (cur_chars + sc + 1 > chunk_size):  # +1 for the joining space
                    sub_text = " ".join(cur).strip()
                    if sub_text:
                        sub_chunks.append(sub_text)

                    # overlap (by chars)
                    if overlap > 0 and cur:
                        overlap_sents, overlap_chars = [], 0
                        for prev in reversed(cur):
                            pc = len(prev)
                            if overlap_chars + pc + (1 if overlap_sents else 0) <= overlap:
                                overlap_sents.insert(0, prev)
                                overlap_chars += pc + (1 if overlap_sents else 0)
                            else:
                                break
                        cur, cur_chars = overlap_sents, overlap_chars
                    else:
                        cur, cur_chars = [], 0

                # add current sentence
                if cur:
                    cur_chars += 1  # space
                cur.append(s)
                cur_chars += sc

            if cur:
                sub_text = " ".join(cur).strip()
                if sub_text:
                    sub_chunks.append(sub_text)
            return sub_chunks or [text]

        def subchunk_by_tokens(text: str, chunk_tok: int, overlap_tok: int):
            """New token-based sub-chunking over sentence list, with token overlap."""
            if not text:
                return []
            sents = split_sentences(text) or [text]
            # precompute counts once
            tok_counts = [len(tokenizer.encode(s)) for s in sents]

            subs, cur, cur_tok = [], [], 0
            i = 0
            while i < len(sents):
                s, t = sents[i], tok_counts[i]

                if t > chunk_tok:
                    # flush current
                    if cur:
                        subs.append(" ".join(cur).strip())
                        cur, cur_tok = [], 0
                    # force split the long sentence
                    parts = force_split_by_tokens(s, chunk_tok)
                    subs.extend(p.strip() for p in parts[:-1] if p.strip())
                    last_p = parts[-1].strip()
                    if last_p:
                        cur = [last_p]
                        cur_tok = len(tokenizer.encode(last_p))
                    i += 1
                    continue

                if cur_tok + t > chunk_tok and cur:
                    subs.append(" ".join(cur).strip())

                    # token overlap
                    if overlap_tok > 0 and cur:
                        overlap_sents, overlap_cnt = [], 0
                        j = len(cur) - 1
                        while j >= 0:
                            tt = len(tokenizer.encode(cur[j]))
                            if overlap_cnt + tt <= overlap_tok:
                                overlap_sents.insert(0, cur[j])
                                overlap_cnt += tt
                                j -= 1
                            else:
                                break
                        cur, cur_tok = overlap_sents, overlap_cnt
                    else:
                        cur, cur_tok = [], 0

                cur.append(s)
                cur_tok += t
                i += 1

                if cur_tok == chunk_tok:
                    subs.append(" ".join(cur).strip())
                    cur, cur_tok = [], 0

            if cur:
                subs.append(" ".join(cur).strip())
            return subs or [text]

        def chunk_then_subchunk(text: str):
            text = (text or "").strip()
            if not text:
                return []

            # 1) token-aware sentence chunking with token overlap
            sentences = split_sentences(text)
            if not sentences:
                # fallback: treat entire text as a "sentence", still enforce long-sentence splitting
                sentences = force_split_by_tokens(text, chunk_tokens)

            token_chunks = chunk_by_tokens_with_overlap(
                sentences, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
            )

            # 2) sub-chunk each token-chunk by chosen mode
            out = []
            if subchunk_mode.lower() == "tokens":
                for tc in token_chunks:
                    out.extend(subchunk_by_tokens(tc, sub_chunk_token_size, sub_chunk_token_overlap))
            else:  # "chars" (original)
                for tc in token_chunks:
                    out.extend(subchunk_by_chars(tc, sub_chunk_chars, sub_chunk_overlap))
            return out

        # ---------- process ----------
        all_chunks = []
        for item in items:
            ctx = (item.get("context") or "").strip()
            if not ctx:
                continue
            all_chunks.extend(chunk_then_subchunk(ctx))

        if not all_chunks:
            # match original spirit but friendlier to callers
            print("[preload_context_json] No chunks produced.")
            return []

        total_chunks = len(all_chunks)
        if subchunk_batch and subchunk_batch > 0:
            batch_size  = max(1, ceil(total_chunks / subchunk_batch))
        else:
            batch_size  = total_chunks
        num_batches = ceil(total_chunks / batch_size)
        print(f"[preload_context_json] Total chunks: {total_chunks}, batch_size={batch_size}, num_batches: {num_batches}")

        # ── Concurrent path: fire all API calls at once, then build codebooks ──
        if self.chunking_use == 'llm_concurrent':
            print(f"[preload_context_json] Using CONCURRENT LLM parser")
            facts_codebook_lst = get_code_books(
                all_chunks,
                type='facts',
                rule="Store factual statements.",
                batch_size=batch_size,
                sent_emb=self.word_emb,
                api=self.chunking_api,
            )
            return facts_codebook_lst

        # ── Sequential path (original) ──
        batch_num = 0
        facts_codebook_lst = []
        for i in range(0, total_chunks, batch_size):

            batch_chunks = all_chunks[i:i+batch_size]


            fact_cb = get_code_book(
                batch_chunks,
                type='facts',
                rule="Store factual statements.",
                batch_size=1,
                sent_emb=self.word_emb,
                parser_choice=self.chunking_use,
                api=self.chunking_api,
            )

            print(f'batch {batch_num} codebook is generated')

            facts_codebook_lst.append(fact_cb)


            batch_num+=1

        return facts_codebook_lst


    def encode_question(self,q_prompt,rule):
        # Use question-specific LLM parser when LLM chunking is enabled
        if self.chunking_use in ('llm', 'llm_concurrent'):
            use_simple = _os.environ.get("USE_RETRIEVE_SIMPLE", "0") == "1"
            use_structure = _os.environ.get("USE_STRUCTURE_RETRIEVE", "1") == "1"
            if use_simple:
                parser_choice = 'llm'
            else:
                parser_choice = 'llm_question_structured' if use_structure else 'llm_question'
            return get_code_book(q_prompt, 'questions', rule,
                                 parser_choice=parser_choice,
                                 api=self.chunking_api)
        return get_code_book(q_prompt,'questions',rule)

    def _embed_edge_run(self, edge_run, codebook_main):
        decoded = decode_question(edge_run, codebook_main, fmt='embeddings')
        # 利用你已有的 _avg_vec_from_decoded
        dim = len(codebook_main["e_embeddings"][0]) if codebook_main.get("e_embeddings") else 64
        return _avg_vec_from_decoded(decoded, dim)
    

    # newly added

    # c_text = " <SEP> ".join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in c_triples])
    def ensure_edge_matrix_embedding_in(self):
        if "edge_matrix_embedding" not in self.meta_codebook.keys():
            edge_texts = [f"{self.meta_codebook['e'][h]} {self.meta_codebook['r'][r]} {self.meta_codebook['e'][t]}"
                        for h, r, t in self.meta_codebook['edge_matrix']]
            self.meta_codebook["edge_matrix_embedding"] = get_word_embeddings(edge_texts, self.word_emb)

        if "questions_lst_embedding" not in self.meta_codebook.keys():
            all_q_texts = []
            for q_edges in self.meta_codebook['questions_lst']:
                q_text_chunk = []
                for q_edge in q_edges:
                    q_triples = _triples_words(q_edge, self.meta_codebook)
                    q_text = " <SEP> ".join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in q_triples])
                    q_text_chunk.append(q_text)

                all_q_texts.append(q_text_chunk)

            self.meta_codebook["questions_lst_embedding"] = [get_word_embeddings(q_text_chunk, self.word_emb) for q_text_chunk in all_q_texts]
            print('questions_lst_embedding compeleted')


        if "answers_lst_embedding" not in self.meta_codebook.keys():
            all_a_texts = []
            for a_edges in self.meta_codebook['answers_lst']:
                a_text_chunk = []
                for a_edge in a_edges:
                    a_triples = _triples_words(a_edge, self.meta_codebook)
                    a_text = " <SEP> ".join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in a_triples])
                    a_text_chunk.append(a_text)
                all_a_texts.append(a_text_chunk)

            self.meta_codebook["answers_lst_embedding"] = [get_word_embeddings(a_text_chunk, self.word_emb) for a_text_chunk in all_a_texts]
            print('answers_lst_embedding compeleted')


        if "facts_lst_embedding" not in self.meta_codebook.keys() and len(self.meta_codebook['facts_lst'])>0:
            all_f_texts = []
            for f_edges in self.meta_codebook['facts_lst']:
                f_text_chunk = []
                for f_edge in f_edges:
                    f_triples = _triples_words(f_edge, self.meta_codebook)
                    f_text = " <SEP> ".join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in f_triples])
                    f_text_chunk.append(f_text)

                all_f_texts.append(f_text_chunk)

            self.meta_codebook["facts_lst_embedding"] = [get_word_embeddings(f_text_chunk, self.word_emb) for f_text_chunk in all_f_texts]
            print('facts_lst_embedding compeleted')


    
    def _flatten_facts(self, meta):
        flat, map_idx = [], []
        for gi, group in enumerate(meta.get('facts_lst', [])):
            for fj, run in enumerate(group):
                if run and isinstance(run, (list, tuple)) and isinstance(run[0], int):
                    flat.append(run)
                    map_idx.append([gi, fj])        # ← 用列表
                elif isinstance(run, (list, tuple)):
                    for r2 in run:
                        if r2 and isinstance(r2, (list, tuple)) and isinstance(r2[0], int):
                            flat.append(r2)
                            map_idx.append([gi, fj])  # ← 用列表
        return flat, map_idx

    def retrieve_new(self, q_json, parallel=True):

        # capture entity count BEFORE merge — used for grounding detection
        n_ents_before = len(self.meta_codebook.get('e', [])) if self.meta_codebook else 0

        self.meta_codebook = merging_codebook(self.meta_codebook, q_json, 'questions', self.word_emb, True)
        questions_edges_index = self.meta_codebook['questions_lst'][-1]

        # Compact query-edge summary (always printed)
        E = self.meta_codebook.get('e', [])
        R = self.meta_codebook.get('r', [])
        EM = self.meta_codebook.get('edge_matrix', [])
        for ri, run in enumerate(questions_edges_index):
            edges_str = ' → '.join(
                f"({E[EM[int(eid)][0]]}, {R[EM[int(eid)][1]]}, {E[EM[int(eid)][2]]})"
                for eid in run if int(eid) < len(EM) and EM[int(eid)][0] < len(E) and EM[int(eid)][2] < len(E) and EM[int(eid)][1] < len(R)
            )
            print(f"  edges[{ri}]: {edges_str}")

        _answers_lst = self.meta_codebook.get('answers_lst', [])
        adapted_m = min(max(1, int(0.1 * len(_answers_lst))), self.top_m)
        has_answers = len(_answers_lst) > 0
        use_structure_retrieve = _os.environ.get("USE_STRUCTURE_RETRIEVE", "1") == "1"

        if DEBUG_VERBOSE:
            print(f"[retrieve_new] use_structure_retrieve={use_structure_retrieve}, has_answers={has_answers}, n_ents_before={n_ents_before}")

        if use_structure_retrieve:
            # FIX C: minimum floor for questions-side budget.
            # The original adapted_m = 0.1 * len(answers_lst) collapses to 1
            # early in the run, making chain retrieval on questions-side useless
            # even when the chain retriever finds good candidates.
            adapted_m = max(adapted_m, min(5, self.top_m))
            # Prebuild caches once per target and reuse inside the structure-aware retriever.
            # v8_ann manages its own caching internally (incremental updates, base emb caching),
            # so skip prebuilding to avoid redundant ~1s stack_base_embeddings per target.
            question_cache = None
            facts_cache = None
            _use_ann = _os.environ.get("USE_RETRIEVE_ANN", "0") == "1"
            if not _use_ann:
                if has_answers:
                    try:
                        from retrieve_gpu_cached_combined import TorchDBCache as _CombinedCache
                        question_cache = _CombinedCache.build(self.meta_codebook, target='questions')
                    except Exception as e:
                        if DEBUG_VERBOSE:
                            print(f"[retrieve_new] question cache build failed, falling back: {e}")
                            question_cache = None
                try:
                    from retrieve_gpu_cached_combined import TorchDBCache as _CombinedCache
                    facts_cache = _CombinedCache.build(self.meta_codebook, target='facts')
                except Exception as e:
                    if DEBUG_VERBOSE:
                        print(f"[retrieve_new] facts cache build failed, falling back: {e}")
                        facts_cache = None

            # Extract RETRIEVAL (compressed) runs from structured parser output.
            # questions_lst[-1] = HOPS (expanded edges, one run per hop)
            # questions_compressed_lst[-1] = RETRIEVAL (compressed edges, one run per branch)
            # Falls back to HOPS if structured parser wasn't used.
            compressed_lst = self.meta_codebook.get('questions_compressed_lst', [])
            compressed_edges_index = None
            if compressed_lst:
                last_compressed = compressed_lst[-1]
                if last_compressed is not None:
                    compressed_edges_index = last_compressed

            # Extract parser GROUPS if available (from structured parser)
            groups_lst = self.meta_codebook.get('questions_groups_lst', [])
            parser_groups = None
            if groups_lst:
                last_groups = groups_lst[-1]
                if last_groups is not None:
                    parser_groups = last_groups

            if parallel and has_answers:
                def _retrieve_answers():
                    top_m_results = retrieve_top_m_by_structure(
                        questions_edges_index,
                        self.meta_codebook,
                        self.sentence_emb,
                        top_k=self.top_k,
                        top_m=adapted_m,
                        target='questions',
                        n_ents_before=n_ents_before,
                        prebuilt_cache=question_cache,
                        compressed_runs=compressed_edges_index,
                        parser_groups=parser_groups,
                    )
                    return get_all_results_entire_chunk(top_m_results, self.meta_codebook, 'answers')

                def _retrieve_facts():
                    top_m_results_for_facts = retrieve_top_m_by_structure(
                        questions_edges_index,
                        self.meta_codebook,
                        self.sentence_emb,
                        top_k=self.top_k,
                        top_m=self.top_m,
                        target='facts',
                        n_ents_before=n_ents_before,
                        prebuilt_cache=facts_cache,
                        compressed_runs=compressed_edges_index,
                        parser_groups=parser_groups,
                    )
                    return get_all_results(top_m_results_for_facts, self.meta_codebook)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_answers = executor.submit(_retrieve_answers)
                    fut_facts = executor.submit(_retrieve_facts)
                    all_answers, all_q_indices = fut_answers.result()
                    all_facts, _ = fut_facts.result()
            else:
                if has_answers:
                    top_m_results = retrieve_top_m_by_structure(
                        questions_edges_index,
                        self.meta_codebook,
                        self.sentence_emb,
                        top_k=self.top_k,
                        top_m=adapted_m,
                        target='questions',
                        n_ents_before=n_ents_before,
                        prebuilt_cache=question_cache,
                        compressed_runs=compressed_edges_index,
                        parser_groups=parser_groups,
                    )
                    all_answers, all_q_indices = get_all_results_entire_chunk(top_m_results, self.meta_codebook, 'answers')
                else:
                    all_answers = []
                    all_q_indices = []

                top_m_results_for_facts = retrieve_top_m_by_structure(
                    questions_edges_index,
                    self.meta_codebook,
                    self.sentence_emb,
                    top_k=self.top_k,
                    top_m=self.top_m,
                    target='facts',
                    n_ents_before=n_ents_before,
                    prebuilt_cache=facts_cache,
                    compressed_runs=compressed_edges_index,
                    parser_groups=parser_groups,
                )
                all_facts, _ = get_all_results(top_m_results_for_facts, self.meta_codebook)

            if DEBUG_VERBOSE:
                print('all_facts', all_facts)
            return all_answers, all_q_indices, all_facts

        # -------- legacy fallback --------
        if parallel and has_answers:
            def _retrieve_answers():
                top_m_results = coarse_filter_torch(
                                questions_edges_index,
                                self.meta_codebook,
                                self.sentence_emb,
                                self.top_k,
                                adapted_m,
                                'questions')
                top_m_results = adaptive_graph_expand(
                                self.meta_codebook, top_m_results, 'questions',
                                questions_edges_index, n_ents_before)
                return get_all_results_entire_chunk(top_m_results, self.meta_codebook, 'answers')

            def _retrieve_facts():
                top_m_results_for_facts = coarse_filter_torch(
                                            questions_edges_index,
                                            self.meta_codebook,
                                            self.sentence_emb,
                                            self.top_k,
                                            self.top_m,
                                            'facts')
                top_m_results_for_facts = adaptive_graph_expand(
                                            self.meta_codebook, top_m_results_for_facts, 'facts',
                                            questions_edges_index, n_ents_before)
                return get_all_results(top_m_results_for_facts, self.meta_codebook)

            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_answers = executor.submit(_retrieve_answers)
                fut_facts = executor.submit(_retrieve_facts)
                all_answers, all_q_indices = fut_answers.result()
                all_facts, _ = fut_facts.result()
        else:
            if has_answers:
                top_m_results = coarse_filter_torch(
                                questions_edges_index,
                                self.meta_codebook,
                                self.sentence_emb,
                                self.top_k,
                                adapted_m,
                                'questions')
                top_m_results = adaptive_graph_expand(
                                self.meta_codebook, top_m_results, 'questions',
                                questions_edges_index, n_ents_before)
                all_answers, all_q_indices = get_all_results_entire_chunk(top_m_results, self.meta_codebook, 'answers')
            else:
                all_answers = []
                all_q_indices = []

            top_m_results_for_facts = coarse_filter_torch(
                                        questions_edges_index,
                                        self.meta_codebook,
                                        self.sentence_emb,
                                        self.top_k,
                                        self.top_m,
                                        'facts')
            top_m_results_for_facts = adaptive_graph_expand(
                                        self.meta_codebook, top_m_results_for_facts, 'facts',
                                        questions_edges_index, n_ents_before)
            all_facts, _ = get_all_results(top_m_results_for_facts, self.meta_codebook)

        if DEBUG_VERBOSE:
            print('all_facts', all_facts)

        return all_answers, all_q_indices, all_facts


    def find_related_knowledge(self, all_answers, all_q_indices, all_facts):
        domain_knowledge_lst = []

        # remove the flatten part
        # get_overped_or_unique_edge_lists_sentence_emebed

        # answers
        if self.include_answers and all_answers:
            final_answers_lsts = self.answers_extract_function(self.meta_codebook, get_flat_answers_lsts(all_answers),self.sentence_emb)
            if DEBUG_VERBOSE:
                print(f'self.answers_choice  {self.answers_choice}')
                print(f'final_answers_lsts {final_answers_lsts}')
            if final_answers_lsts:
                domain_knowledge_lst.append(final_answers_lsts)
            else:
                domain_knowledge_lst.append([])


        # thinkings
        if self.include_thinkings:
            thinkings_lsts = get_thinkings_lsts(all_q_indices, self.meta_codebook)
            final_thinkings_lsts = self.thinking_extract_function(self.meta_codebook,get_flat_answers_lsts(thinkings_lsts),self.sentence_emb)
            if DEBUG_VERBOSE:
                print(f'self.thinkings_choice  {self.thinkings_choice}')
                print(f'final_thinkings_lsts {final_thinkings_lsts}')
            if final_thinkings_lsts:
                domain_knowledge_lst.append(final_thinkings_lsts)
            else:
                domain_knowledge_lst.append([])


        # facts 
        if self.include_facts:
            if all_facts:
                def is_effectively_empty(x):
                    # empty, None, or all inner items are empty
                    if not x:
                        return True
                    if isinstance(x, list):
                        return all(is_effectively_empty(i) for i in x)
                    return False
                

                if DEBUG_VERBOSE:
                    print(f'original facts_lsts {all_facts}')
                if self.facts_choice == 'include_all':
                    extracted_facts_lsts = get_flat_answers_lsts(all_facts)
                else:
                    extracted_facts_lsts = self.facts_extract_function(self.meta_codebook,get_flat_answers_lsts(all_facts),self.sentence_emb)

                if DEBUG_VERBOSE:
                    print(f'extracted_facts_lsts is {extracted_facts_lsts}')

                # if empty takes oriiginal (deepcopy only when needed)
                if  is_effectively_empty(extracted_facts_lsts):
                    if DEBUG_VERBOSE:
                        print('keep original facts_lsts')
                    final_facts_lsts = get_flat_answers_lsts(copy.deepcopy(all_facts))
                else:
                    if DEBUG_VERBOSE:
                        print('use extracted_facts_lsts') 
                    final_facts_lsts = extracted_facts_lsts

                    
                if final_facts_lsts:                
                    domain_knowledge_lst.append(final_facts_lsts)
                else:
                    domain_knowledge_lst.append([])

                if DEBUG_VERBOSE:
                    print(f'final_facts_lsts{final_facts_lsts}')

        return domain_knowledge_lst

    def compact_indicies_for_prompt(self, codebook_sub_q, domain_knowledge_lst):
        flat_answers_lsts: Optional[list] = None
        flat_thinkings_lsts: Optional[list] = None
        flat_facts_lsts: Optional[list] = None

        ptr = 0
        if self.include_answers and ptr < len(domain_knowledge_lst):
            flat_answers_lsts = domain_knowledge_lst[ptr]
            ptr += 1

        if self.include_thinkings and ptr < len(domain_knowledge_lst):
            flat_thinkings_lsts = domain_knowledge_lst[ptr]
            ptr += 1

        include_facts = getattr(self, "include_facts", True)
        if include_facts and ptr < len(domain_knowledge_lst):
            flat_facts_lsts = domain_knowledge_lst[ptr]
            ptr += 1

        if self.include_answers and self.include_thinkings and flat_answers_lsts and flat_thinkings_lsts:
            final_merged_json = get_json_with_given_knowledge_and_thinkings(
                flat_answers_lsts , 
                flat_thinkings_lsts ,
                self.meta_codebook, 
                codebook_sub_q
            )
        elif self.include_answers and flat_answers_lsts:
            final_merged_json = get_json_with_given_knowledge(
                flat_answers_lsts , 
                self.meta_codebook, 
                codebook_sub_q
            )
        elif self.include_thinkings and flat_thinkings_lsts:
            final_merged_json = get_json_with_given_thinkings(
                flat_thinkings_lsts , 
                self.meta_codebook, 
                codebook_sub_q
            )
        else:
            final_merged_json = codebook_sub_q.copy()

            # make sure final merged json have same output keys
            final_merged_json['edge_matrix']  = final_merged_json['edges([e,r,e])']
            final_merged_json.pop('edges([e,r,e])')

        if include_facts and flat_facts_lsts:
            em_final = final_merged_json['edge_matrix']
            E_final  = final_merged_json['e']
            R_final  = final_merged_json['r']

            e_name2idx = {name: i for i, name in enumerate(E_final)}
            r_name2idx = {name: i for i, name in enumerate(R_final)}
            tuple2idx  = {tuple(e): i for i, e in enumerate(em_final)}

            def _ensure_ent_from_meta(old_e_idx: int) -> int:
                name = self.meta_codebook['e'][old_e_idx]
                idx = e_name2idx.get(name)
                if idx is None:
                    idx = len(E_final)
                    E_final.append(name)
                    e_name2idx[name] = idx
                return idx

            def _ensure_rel_from_meta(old_r_idx: int) -> int:
                name = self.meta_codebook['r'][old_r_idx]
                idx = r_name2idx.get(name)
                if idx is None:
                    idx = len(R_final)
                    R_final.append(name)
                    r_name2idx[name] = idx
                return idx

            def ensure_edge_from_meta(meta_edge_idx: int) -> int:
                e1_old, r_old, e2_old = self.meta_codebook['edge_matrix'][meta_edge_idx]
                h = _ensure_ent_from_meta(e1_old)
                r = _ensure_rel_from_meta(r_old)
                t = _ensure_ent_from_meta(e2_old)
                tup = (h, r, t)
                idx = tuple2idx.get(tup)
                if idx is None:
                    idx = len(em_final)
                    em_final.append([h, r, t])
                    tuple2idx[tup] = idx
                return idx
            remapped_facts = [[ensure_edge_from_meta(i) for i in run] for run in flat_facts_lsts]

            final_merged_json['e'] = E_final
            final_merged_json['r'] = R_final
            final_merged_json['edge_matrix'] = em_final
            final_merged_json['facts(edges[i])'] = remapped_facts
            final_merged_json['facts([[e,r,e], ...])'] = decode_questions(
                remapped_facts, final_merged_json, 'edges'
            )

        return final_merged_json


    # might also change these functions,now keep always merge with answers json, and only merge with thinking json if use thinkings
    
    def collect_results(self, final_merged_json, questions, retrieval_time: float = 0.0):
        llm = self.llm

        new_json_lst = []
        new_result = None

        if self.include_thinkings:
            a_new, t_new = llm.take_questions(final_merged_json, questions, retrieval_time=retrieval_time)
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')
            t_new_json = get_code_book(t_new, type='thinkings')

            if is_no_answer_text(a_new):
                a_new_json["edges([e,r,e])"] = []

            new_json_lst.extend([a_new_json, t_new_json])
        else:
            a_new = llm.take_questions(final_merged_json, questions, retrieval_time=retrieval_time)
            if DEBUG_VERBOSE:
                print(a_new)
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')

            if is_no_answer_text(a_new):
                a_new_json["edges([e,r,e])"] = []

            new_json_lst.append(a_new_json)
        return new_result,new_json_lst
    

    # only being used for dpo version, collecting more info for reward
    def update_meta(self, new_json_lst, facts_cb=None):
        if self.include_thinkings:
            codebook_sub_a, codebook_sub_t = new_json_lst
            if len(codebook_sub_a["edges([e,r,e])"])>0:
                self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_a, 'answers',   self.word_emb, True)
                self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_t, 'thinkings', self.word_emb, True)
            else:
                self.meta_codebook['questions_lst'].pop()
                if self.meta_codebook.get('questions_lst_embedding'):
                    self.meta_codebook['questions_lst_embedding'].pop()
                if self.meta_codebook.get('questions_compressed_lst'):
                    self.meta_codebook['questions_compressed_lst'].pop()
                if self.meta_codebook.get('questions_groups_lst'):
                    self.meta_codebook['questions_groups_lst'].pop()

        else:
            codebook_sub_a = new_json_lst[0]

            if len(codebook_sub_a["edges([e,r,e])"])>0:
                self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_a, 'answers',   self.word_emb, True)
            else:
                self.meta_codebook['questions_lst'].pop()
                if self.meta_codebook.get('questions_lst_embedding'):
                    self.meta_codebook['questions_lst_embedding'].pop()
                if self.meta_codebook.get('questions_compressed_lst'):
                    self.meta_codebook['questions_compressed_lst'].pop()
                if self.meta_codebook.get('questions_groups_lst'):
                    self.meta_codebook['questions_groups_lst'].pop()

        if facts_cb is not None:
            if DEBUG_VERBOSE:
                print("----------fact is loaded------")
            self.meta_codebook = merging_codebook(self.meta_codebook, facts_cb, 'facts', self.word_emb, False)
            self._facts_preloaded = True


    def _save_combined_codebook_to(self, out_path):
        """Save post-combine codebook so future runs skip the expensive initial combine."""
        import json as _json
        from pathlib import Path
        out = Path(out_path)
        # Strip non-serialisable runtime caches (top-level _ keys only)
        _keep_keys = {'_dirty_edges', '_dirty_entities', '_combine_round'}
        def _safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_safe(v) for v in obj]
            return obj
        save_data = {k: _safe(v) for k, v in self.meta_codebook.items()
                     if not k.startswith('_') or k in _keep_keys}
        out.parent.mkdir(parents=True, exist_ok=True)
        # Stream JSON to file to avoid building full string in memory
        with open(out, 'w', encoding='utf-8') as f:
            _json.dump(save_data, f, separators=(',', ':'))

    def combine_ents_func(self, mode="auto"):
        if mode == "auto":
            self.meta_codebook = combine_ents_auto(self.meta_codebook,
                    self.min_exp_num,
                    self.max_exp_num,
                    self.include_thinkings,
                    sample_size_prop = self.sample_size_prop,
                    k_grid_size = self.k_grid_size,
                    word_emb = self.word_emb
                    )
        elif mode == "knn":
            self.meta_codebook = combine_ents_ann_knn(self.meta_codebook,sim_threshold = self.combine_ent_sim,word_emb = self.word_emb)
        elif mode == "coarse":
            self.meta_codebook = coarse_combine(self.meta_codebook,sim_threshold = self.combine_ent_sim,word_emb = self.word_emb)

        # clear dirty state and advance round counter
        self.meta_codebook["_dirty_edges"] = set()
        self.meta_codebook["_dirty_entities"] = set()
        self.meta_codebook["_combine_round"] = self.meta_codebook.get("_combine_round", 0) + 1
        # reset remap-done flags so new remap from combine_ents is applied
        self.meta_codebook.pop('_remap_done_facts', None)
        self.meta_codebook.pop('_remap_done_questions', None)

        # invalidate caches — entity/relation IDs were remapped
        self.meta_codebook.pop('_ent_inv_idx_questions', None)
        self.meta_codebook.pop('_ent_inv_idx_facts', None)
        self.meta_codebook.pop('_ent_inv_wm_questions', None)
        self.meta_codebook.pop('_ent_inv_wm_facts', None)
        # invalidate v8_ann caches that depend on entity/relation IDs
        self.meta_codebook.pop('_chunk_ents_questions', None)
        self.meta_codebook.pop('_chunk_ents_facts', None)
        self.meta_codebook.pop('_chunk_rels_questions', None)
        self.meta_codebook.pop('_chunk_rels_facts', None)
        self.meta_codebook.pop('_chunk_er_wm_questions', None)
        self.meta_codebook.pop('_chunk_er_wm_facts', None)
        self.meta_codebook.pop('_norm_r_matrix', None)
        self.meta_codebook.pop('_bridge_ent_inv_questions', None)
        self.meta_codebook.pop('_bridge_ent_inv_facts', None)
        self.meta_codebook.pop('_bridge_ent_inv_wm_questions', None)
        self.meta_codebook.pop('_bridge_ent_inv_wm_facts', None)
        self.meta_codebook.pop('_edge_matrix_np_cache', None)
        # NOTE: TorchDBCache is NOT invalidated here — _combine_ents_remap signal
        # (stored by combine_ents_auto) triggers fast in-place remap in
        # _get_or_build_torch_cache, avoiding expensive full rebuild.


    def load_and_merge_facts(
        self, facts_json_path,
        chunk_tokens=1200, overlap_tokens=100,
        sub_chunk_chars=600, sub_chunk_overlap=50,
        tokenizer_name="gpt-4o-mini",
        subchunk_batch = 500,
        subchunk_mode = 'chars',
        sub_chunk_token_size: int = 300,       # used when subchunk_mode="tokens"
        sub_chunk_token_overlap: int = 50,     # used when subchunk_mode="tokens"
    ):
        if not facts_json_path:
            return None
        if isinstance(facts_json_path, (list, tuple)):
            paths = [p for p in facts_json_path if p]
        else:
            paths = [facts_json_path]

        for p in paths:
            facts_codebook_lst = self.preload_context_json(
                p,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                sub_chunk_chars=sub_chunk_chars,
                sub_chunk_overlap=sub_chunk_overlap,
                tokenizer_name=tokenizer_name,
                subchunk_batch = subchunk_batch,
                subchunk_mode =subchunk_mode,
                sub_chunk_token_size =sub_chunk_token_size,
                sub_chunk_token_overlap = sub_chunk_token_overlap
            )
            for cb in facts_codebook_lst:
                if cb:
                    self.meta_codebook = merging_codebook(
                        self.meta_codebook, cb, 'facts', self.word_emb, False
                    )

                    if DEBUG_VERBOSE:
                        print("len(self.meta_codebook[facts_lst])",len(self.meta_codebook["facts_lst"]))

            # newly adding trying to merge boundaries
            if DEBUG_VERBOSE:
                print('starting to check whether to merge boundaries')
            self.meta_codebook['facts_lst'],self.meta_codebook['facts_lst_embedding']  = merge_chunks_by_boundary(chunks = self.meta_codebook['facts_lst'],
                                                                                                                  embeddings=self.meta_codebook.get('facts_lst_embedding'),
                                                                    codebook_main = self.meta_codebook,
                                                                    sent_emb = self.sentence_emb)
            
            if "facts_lst" in self.meta_codebook and len(self.meta_codebook["facts_lst"]) >= 2:
                if "facts_lst_embedding" in self.meta_codebook:
                    f, e = dedup_facts_and_embeddings(
                        self.meta_codebook["facts_lst"],
                        self.meta_codebook["facts_lst_embedding"],
                    )
                    self.meta_codebook["facts_lst"] = f
                    self.meta_codebook["facts_lst_embedding"] = e
                else:
                    self.meta_codebook["facts_lst"] = remove_duplicate_inner_lists(self.meta_codebook["facts_lst"])


    def run_work_flow(self, q_prompt, rule="Answer questions",
                    facts_json_path: list = None, chunk_chars: int = 800,
                    overlap: int = 120, warm_start = "knn", return_metrics: bool = False, gold_ref: str | None = None,
                    skip_update_meta: bool = False, skip_combine_ents: bool = False):

        if DEBUG_VERBOSE:
            snapshot("run_work_flow START")
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        # prevent dpo change choice but not change includings
        t0 = time.perf_counter()
        self.set_includings()
        t_set_incl = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_json = self.encode_question(q_prompt, rule)
        t_encode = time.perf_counter() - t0

        timings["set_includings"] = t_set_incl
        timings["encode_question (LLM parse)"] = t_encode

        combined_facts_cb = None
        t0 = time.perf_counter()
        if not getattr(self, "_facts_preloaded", False) and facts_json_path:
            self.load_and_merge_facts(facts_json_path, chunk_chars, overlap)
            self._facts_preloaded = True
        timings["load_facts"] = time.perf_counter() - t0

        if self.meta_codebook:
            t0 = time.perf_counter()
            if DEBUG_VERBOSE:
                codebook_mem_breakdown(self.meta_codebook, "before retrieve_new")
                snapshot("before retrieve_new")
            timings["snapshot_before_retrieve"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            all_answers, all_q_indices, all_facts = self.retrieve_new(q_json)
            timings["retrieve_new"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            if DEBUG_VERBOSE:
                snapshot("after retrieve_new")
            timings["snapshot_after_retrieve"] = time.perf_counter() - t0

            retrieval_time = timings["retrieve_new"]
            if DEBUG_VERBOSE:
                print("all_answers", all_answers)
                print("all_q_indices", all_q_indices)
                print("all_facts", all_facts)

            if DEBUG_VERBOSE:
                print(f'answers choice is {self.answers_choice}')
                print(f'thinkings_choice choice is {self.thinkings_choice}')
                print(f'facts choice is {self.facts_choice}')

            t0 = time.perf_counter()
            domain_knowledge_lst = self.find_related_knowledge(all_answers, all_q_indices, all_facts)
            timings["find_related_knowledge"] = time.perf_counter() - t0
            if DEBUG_VERBOSE:
                print("domain_knowledge_lst", domain_knowledge_lst)
                print(f'q_json is {q_json}')

            t0 = time.perf_counter()
            final_merged_json = self.compact_indicies_for_prompt(q_json, domain_knowledge_lst)
            timings["compact_indices"] = time.perf_counter() - t0
        else:
            final_merged_json = combined_facts_cb if combined_facts_cb else q_json.copy()
            retrieval_time = 0
            timings["retrieve_new"] = 0.0
            timings["find_related_knowledge"] = 0.0
            timings["compact_indices"] = 0.0

        if DEBUG_VERBOSE:
            print(f'final_merged_json unsliced {final_merged_json}')
            print('=' * 76)

        t0 = time.perf_counter()
        q_txt, gk_txt, st_txt, ft_txt = select_best_context_by_keys(final_merged_json)
        final_merged_json = slice_for_final_merged_json(final_merged_json, self.use_word)
        timings["context_selection_and_slice"] = time.perf_counter() - t0

        if DEBUG_VERBOSE:
            print(f'final_merged_json sliced {final_merged_json}')
        if gk_txt:
            self.cur_fact_context = ft_txt+gk_txt
        else:
            self.cur_fact_context = ft_txt

        t0 = time.perf_counter()
        new_result, new_json_lst = self.collect_results(final_merged_json, questions=q_prompt, retrieval_time=retrieval_time)
        timings["collect_results_llm"] = time.perf_counter() - t0

        # Clean summary: question → answer (always printed)
        print(f"\n  Q: {q_prompt}")
        print(f"  A: {new_result}\n")

        t0 = time.perf_counter()
        metrics_map = self.llm.last_metrics or {}
        if metrics_map:
            (qk, gen_info) = next(iter(metrics_map.items()))
            try:
                retrieved_count = sum(len(bucket) for bucket in all_answers) if all_answers else 0
            except Exception:
                retrieved_count = 0
            gen_info["retrieved_count"] = int(retrieved_count)
            gen_info["fact_context"] = self.cur_fact_context
            gen_info["total_latency_sec"] = float(
                gen_info.get("latency_sec", 0.0) + gen_info.get("retrieval_latency_sec", 0.0)
            )

            if gold_ref is not None:
                try:
                    smooth = SmoothingFunction().method1
                    bleu = sentence_bleu([gold_ref.split()], str(new_result).split(), smoothing_function=smooth)
                    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                    rs = rouge.score(gold_ref, str(new_result))
                    gen_info["BLEU"]    = float(bleu)
                    gen_info["ROUGE-1"] = float(rs["rouge1"].fmeasure)
                    gen_info["ROUGE-2"] = float(rs["rouge2"].fmeasure)
                    gen_info["ROUGE-L"] = float(rs["rougeL"].fmeasure)
                except Exception:
                    pass
            metrics_map = {qk: gen_info}
        timings["metrics_enrichment"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if not skip_update_meta:
            if DEBUG_VERBOSE:
                snapshot("before update_meta")
            self.update_meta(new_json_lst, facts_cb=combined_facts_cb)
            if DEBUG_VERBOSE:
                snapshot("after update_meta")
        timings["update_meta"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _env_skip_combine = _os.environ.get("SKIP_COMBINE_ENTS", "0") == "1"
        if not skip_combine_ents and not _env_skip_combine:
            self.combine_ents_func(mode=warm_start)
        timings["combine_ents"] = time.perf_counter() - t0

        # Batch dedup: run every 10 questions instead of every question
        self._dedup_counter = getattr(self, '_dedup_counter', 0) + 1
        t0 = time.perf_counter()
        if self._dedup_counter % 10 == 0:
            if DEBUG_VERBOSE:
                print('remove the duplicates ...')
            if "facts_lst" in self.meta_codebook and len(self.meta_codebook["facts_lst"]) >= 2:
                if "facts_lst_embedding" in self.meta_codebook:
                    f, e = dedup_facts_and_embeddings(
                        self.meta_codebook["facts_lst"],
                        self.meta_codebook["facts_lst_embedding"],
                    )
                    self.meta_codebook["facts_lst"] = f
                    self.meta_codebook["facts_lst_embedding"] = e
                else:
                    self.meta_codebook["facts_lst"] = remove_duplicate_inner_lists(self.meta_codebook["facts_lst"])
        timings["dedup_facts"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        _print_timings(timings)

        return (new_result, metrics_map, self.cur_fact_context) if return_metrics else new_result
        
    # dpo version for run work_flow, same process but return the collected metrics from the llm

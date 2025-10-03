import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re
import json, hashlib
from typing import List, Tuple, Dict, Optional,Iterable,Any,Callable, Set, Union
import itertools
from collections import defaultdict
import numpy as np
from gensim.models import KeyedVectors
import numpy as np
import re
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from optimize_combine_ent import combine_ents_auto, combine_ents_ann_knn, coarse_combine
from copy import deepcopy
from textwrap import dedent
from graph_generator.rebel_large import triplet_parser
import time
from sentence_embed_overlap import get_unique_or_overlap_by_sentence_embedded
import gensim.downloader as api
from WordEmb import Word2VecEmbeddings, WordAvgEmbeddings
from functools import partial
import copy
from optimize_combine_storage import ann_feat_combine,ann_merge_questions_answer_gated
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

Triplet = Tuple[str, str, str]

def json_dump_str(obj, indent=0):
    """Return compact JSON string by default; pretty-print if indent>0."""
    if indent:
        return json.dumps(obj, ensure_ascii=False, indent=indent)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

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


def update_the_index(codebook_main, codebook_sub, select_feature):
    items_needs_merged = codebook_sub[select_feature]   # list of strings
    items_main = codebook_main[select_feature]          # list of strings

    index_item_sub  = {val: idx for idx, val in enumerate(items_needs_merged)}
    index_item_main = {val: idx for idx, val in enumerate(items_main)}

    next_idx = len(items_main)  
    new_index_replacement_for_sub = {}
    new_added_items = []

    for item_sub in items_needs_merged:
        if item_sub in index_item_main:
            new_index_replacement_for_sub[index_item_sub[item_sub]] = index_item_main[item_sub]
        else:
            new_index_replacement_for_sub[index_item_sub[item_sub]] = next_idx
            index_item_main[item_sub] = next_idx
            new_added_items.append(item_sub)
            next_idx += 1

    return new_index_replacement_for_sub, index_item_main, new_added_items

def all_chains_no_subchains(edges,use_full_edges = True):
    """
    Generate all simple chains where edges[i].tail == edges[j].head for consecutive edges,
    then remove any chain that appears contiguously inside a longer chain (order matters).
    Returns a list of chains (each chain is a list of edge triples).
    """
    n = len(edges)
    # Build head->indices and adjacency i->list(j) if edges[i].t == edges[j].h
    head_to_idxs = defaultdict(list)
    for i, (h, _, _) in enumerate(edges):
        head_to_idxs[h].append(i)
    adj = [[] for _ in range(n)]
    for i, (_, _, t) in enumerate(edges):
        adj[i] = head_to_idxs.get(t, [])

    # DFS from every edge to enumerate all simple paths (as tuples of indices)
    all_paths = set()
    for s in range(n):
        stack = [(s, (s,))]
        while stack:
            cur, path = stack.pop()
            all_paths.add(path)
            for j in adj[cur]:
                if j not in path:  # no edge reuse
                    stack.append((j, path + (j,)))

    # prune paths that are contiguous subchains of longer ones
    paths = list(all_paths)
    def is_contig_subseq(a, b):
        if len(a) >= len(b): return False
        la = len(a)
        for i in range(len(b)-la+1):
            if b[i:i+la] == a:
                return True
        return False

    keep = []
    for a in paths:
        if not any(is_contig_subseq(a, b) for b in paths if b is not a):
            keep.append(a)

    # map back to triples
    if use_full_edges:
      keep_chains = [[edges[i] for i in tup] for tup in keep]
    else:
      keep_chains = [[i for i in tup] for tup in keep]
    # (optional) sort by length desc then lexicographically by indices for stable output
    keep_chains.sort(key=lambda c: (-len(c), c))
    return keep_chains

def edges_from_triples(
    triples: List[Tuple[str, str, str]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
) -> List[List[int]]:
    g = []
    for h, r, t in triples:
        g.append([ent2id[h], rel2id[r], ent2id[t]])
    return g

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

def _merge_sets(sets: Iterable[Set[Triplet]]) -> Set[Triplet]:
    merged: Set[Triplet] = set()
    for s in sets:
        if s:
            merged |= s
    return merged

def preload_context_json(json_path: str, chunk_chars: int = 1200, overlap: int = 100):
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _chunk_text(text: str, *, chunk_chars: int = 800, overlap: int = 120):
        text = (text or "").strip()
        if not text:
            return []
        chunks, n = [], len(text)
        step = max(1, chunk_chars - overlap)
        i = 0
        while i < n:
            j = min(n, i + chunk_chars)
            chunk = text[i:j].strip()
            if chunk:
                chunks.append(chunk)
            if j == n:
                break
            i += step
        return chunks

    items = data if isinstance(data, list) else [data]

    all_chunks = []
    for item_idx, item in enumerate(items):
        ctx = (item.get("context") or "").strip()
        if ctx:
            item_chunks = _chunk_text(ctx, chunk_chars=chunk_chars, overlap=overlap)
            print(f"Item {item_idx}: original text length = {len(ctx)} chars, split into {len(item_chunks)} chunks")
            for chunk_idx, chunk in enumerate(item_chunks):
                print(f"  Chunk {chunk_idx}: {len(chunk)} chars")
            all_chunks.extend(item_chunks)

    total_chunks = len(all_chunks)
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunk size: {sum(len(chunk) for chunk in all_chunks) / total_chunks:.1f} chars" if total_chunks > 0 else "No chunks created")
    
    if total_chunks == 0:
        return None
    
    combined = None  
    fact_cb = get_code_book(
        all_chunks,
        type='facts',
        rule="Store factual statements.",
        # batch_size=64, 
    )
    if combined is None:
        combined = {
            "e": list(fact_cb.get("e", [])),
            "r": list(fact_cb.get("r", [])),
            "edge_matrix": list(fact_cb.get("edges([e,r,e])", [])),
            "facts(edges[i])": [lst for lst in fact_cb.get("facts(edges[i])", [])],
            "questions_lst": [],
            "answers_lst": [],
            "thinkings_lst": [],
            "rule": fact_cb.get("rule", "Store factual statements."),
            "e_embeddings": [],
            "r_embeddings": [],
        }
    else:
        ent_map, _, new_ents = update_the_index(combined, fact_cb, "e")
        rel_map, _, new_rels = update_the_index(combined, fact_cb, "r")

        edges_remapped = remap_edges_matrix(
            fact_cb["edges([e,r,e])"], ent_map, rel_map
        )

        edge_map, new_edge_matrix = combine_updated_edges(
            combined["edge_matrix"], edges_remapped
        )

        facts_runs = fact_cb["facts(edges[i])"]
        facts_runs_mapped = remap_question_indices(facts_runs, edge_map)
        combined["facts(edges[i])"].extend(facts_runs_mapped)

        combined["e"].extend(new_ents)
        combined["r"].extend(new_rels)
        combined["edge_matrix"] = new_edge_matrix
    print(f"{json_path} chunk num:", total_chunks)
    return combined
    
preload_context_json("/Users/lancelotchu/Desktop/GraphRAG-Benchmark/Datasets/Corpus/medical.json", chunk_chars=1200, overlap=100)

def get_code_book(
    prompt: Union[str, List[str]],
    type: str = 'questions',
    rule: str = "Answer questions.",
    factparser: bool = False,   
    *,
    batch_size: int = 64       
):
    valid_types = {'questions', 'answers', 'thinkings', 'facts'}
    if type not in valid_types:
        raise ValueError(f"type must be one of {valid_types}, got: {type}")

    if isinstance(prompt, str):
        triples_merged: Set[Triplet] = triplet_parser(prompt)  # Set[Triplet]
    else:
        texts: List[str] = [t for t in prompt if isinstance(t, str) and t.strip()]
        if not texts:
            triples_merged = set()
        elif len(texts) <= batch_size:
            parsed = triplet_parser(texts)
            if isinstance(parsed, set):
                triples_merged = parsed
            elif isinstance(parsed, list):
                triples_merged = _merge_sets(parsed)          
            else:
                triples_merged = set(parsed)                 
        else:
            acc: List[Iterable[Triplet]] = []
            for i in range(0, len(texts), batch_size):
                parsed = triplet_parser(texts[i:i+batch_size])
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

    codebook, ent2id, rel2id = build_codebook_from_triples(triples_merged, rule)
    edges = edges_from_triples(triples_merged, ent2id, rel2id)
    feat_name = f"{type}(edges[i])"

    codebook.update({
        "edges([e,r,e])": edges,
        feat_name: all_chains_no_subchains(edges, False),
    })
    codebook.pop('sid', None)
    return codebook
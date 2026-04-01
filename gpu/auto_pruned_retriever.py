import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys.stderr, 'reconfigure'):
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import networkx as nx
import matplotlib.pyplot as plt
import json, hashlib
from typing import List, Tuple, Dict, Optional,Iterable,Any,Callable, Set, Union,Mapping
from collections import defaultdict
import numpy as np
import numpy as np
import re
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from combine_ent_cached_aligned import combine_ents_auto, combine_ents_ann_knn, coarse_combine
from copy import deepcopy
from mem_debug import snapshot, codebook_mem_breakdown, force_gc_and_report
from math import ceil
from textwrap import dedent
try:
    from graph_generator.rebel_large import triplet_parser
except ImportError:
    triplet_parser = None  # REBEL not available; use LLM parser
from graph_generator.llm_parser import triplet_parser_llm, triplet_parser_llm_question, triplet_parser_llm_question_structured, TOKEN_STATS
from graph_generator.llm_parser_concurrent import triplet_parser_llm_concurrent
import time
from sentence_embed_overlap_cached import get_unique_or_overlap_by_sentence_embedded
from functools import partial
import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from test_continous_chunk_cached import embed_triples_as_sentences,segment_by_centroid_sim,merge_chunks_by_boundary
import os as _os
from retrieve_gpu_cached_combined import coarse_filter_torch, _print_timings

DEBUG_VERBOSE = _os.environ.get("DEBUG_VERBOSE", "0") == "1"
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrieve_simple import retrieve_top_m_by_structure_simple as retrieve_top_m_by_structure
print("[AutoPrunedRetriever] Using simple retrieval (coarse_filter_torch only)")



Triplet = Tuple[str, str, str]



word_emb = None  # was loading BAAI/bge-large-en-v1.5 here (~1.3GB wasted, never used in this file)

SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJ_DEPS  = {"dobj", "obj", "attr", "oprd", "dative"}
NEG_DEPS  = {"neg"}



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
def noun_phrase_label(head, include_det=False, use_ents=True):
    # 1) prefer named entities (incl. FAC)
    if use_ents:
        for ent in head.doc.ents:
            if ent.start <= head.i < ent.end and ent.label_ in {
                "PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","FAC"
            }:
                return ent.text

    # 2) noun_chunk (optionally drop determiners)
    chunk = next((nc for nc in head.doc.noun_chunks if nc.root == head), None)
    if chunk is not None:
        toks = [t for t in chunk if include_det or t.dep_ != "det"]
        return " ".join(t.text for t in toks).strip()

    # 3) fallback: compounds/adjectives/numerals + head (+ "of"-PP)
    keep = {"amod", "compound", "nummod", "poss"}
    left = []
    for c in sorted([c for c in head.lefts if c.dep_ in keep], key=lambda x: x.i):
        left.append(c.text if c.dep_ != "poss" else c.text + "'s")
    label = " ".join(left + [head.text]).strip()
    for prep in (c for c in head.children if c.dep_ == "prep" and c.text.lower() == "of"):
        for p in (c for c in prep.children if c.dep_ == "pobj"):
            label += " of " + noun_phrase_label(p, include_det=include_det)
    return label

def verb_label(tok):
    base = tok.lemma_
    prt = []
    prt  = [c.text for c in tok.children if c.dep_ == "prt"]
    return " ".join([base] + prt)

def collect_neg(tok):
    return any(c.dep_ in NEG_DEPS for c in tok.children)

def has_copula(tok):
    return any(c.dep_ == "cop" for c in tok.children)

def is_passive_auxiliary(tok):
    """Check if token is an auxiliary verb in passive construction"""
    return (tok.pos_ == "AUX" and tok.lemma_ == "be" and 
            any(c.dep_ in {"nsubjpass", "csubjpass"} for c in tok.children))

def find_main_verb_in_passive(aux_tok):
    """Find the main verb (participle) in passive construction"""
    # Look for past participle that depends on this auxiliary
    for child in aux_tok.children:
        if child.pos_ == "VERB" and child.tag_ in {"VBN"}:  # past participle
            return child
    
    # Alternative: look in the sentence for past participles
    for tok in aux_tok.doc:
        if (tok.i > aux_tok.i and tok.pos_ == "VERB" and 
            tok.tag_ == "VBN" and tok.head == aux_tok):
            return tok
    
    return None

# -------- robust subject finder --------
def subjects_for(pred):
    # 1) direct dependency
    subs = [c for c in pred.children if c.dep_ in SUBJ_DEPS]
    if subs:
        return subs

    # 2) borrow from coordinated predicate
    if pred.dep_ == "conj" and pred.head.pos_ in {"VERB","ADJ","NOUN"}:
        sh = [c for c in pred.head.children if c.dep_ in SUBJ_DEPS]
        if sh:
            return sh

    # 3) for passive constructions, check if there's an auxiliary with the subject
    if pred.pos_ == "VERB":
        for tok in pred.doc:
            if (tok.pos_ == "AUX" and tok.lemma_ == "be" and
                any(c.dep_ in SUBJ_DEPS for c in tok.children)):
                return [c for c in tok.children if c.dep_ in SUBJ_DEPS]

    # 4) aux-fronted question: noun_chunks between last AUX and predicate
    aux_before = [t for t in pred.doc if t.i < pred.i and t.pos_ == "AUX"]
    if aux_before:
        left_idx = max(a.i for a in aux_before)
        chunks = [nc for nc in pred.doc.noun_chunks if left_idx < nc.end <= pred.i]
        if chunks:
            return [sorted(chunks, key=lambda nc: nc.end)[-1].root]

    # 5) general fallback: rightmost noun_chunk before predicate
    chunks = [nc for nc in pred.doc.noun_chunks if nc.end <= pred.i]
    if chunks:
        return [sorted(chunks, key=lambda nc: nc.end)[-1].root]

    # 6) token fallback
    cands = [t for t in pred.doc if t.i < pred.i and t.pos_ in {"NOUN","PROPN","PRON"}]
    if cands:
        return [cands[-1]]

    return []

def prioritize_semantic_entities(subjects):
    """
    Given multiple potential subjects, prioritize based on linguistic structure.
    Looks for 'of' relationships and compound nouns to find semantic focus.
    """
    semantic_subjects = []

    for subj in subjects:
        # Try to extract semantic entity
        semantic_entity = extract_semantic_subject(subj)
        original_entity = noun_phrase_label(subj if subj.pos_ in {"NOUN", "PROPN"} else subj.head)

        # If we extracted something different, we found a semantic focus
        if semantic_entity != original_entity:
            semantic_subjects.append((subj, semantic_entity, original_entity))
        else:
            semantic_subjects.append((subj, semantic_entity, None))

    return semantic_subjects

def extract_semantic_subject(token, include_det=False):
    """
    Extract semantically meaningful subject from complex noun phrases.
    Promotes 'X of Y' constructions so that Y is treated as the subject.
    """
    # Case 1: "cases of Y" → promote Y
    for prep in token.children:
        if prep.dep_ == "prep" and prep.text.lower() == "of":
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj and pobj.pos_ in {"NOUN", "PROPN"}:
                return noun_phrase_label(pobj, include_det)

    # Case 2: Compounds keep full phrase
    compounds = [c for c in token.children if c.dep_ == "compound"]
    if compounds:
        return noun_phrase_label(token, include_det)

    # Default
    return noun_phrase_label(token, include_det)


def extract_core_noun_types(token, include_det=False):
    """
    Extract core noun types from complex noun phrases.
    For "most common type of skin cancer" -> ["type", "skin cancer"]
    For "the largest city in France" -> ["city"]
    """
    results = []

    # Start with the head noun
    if token.pos_ in {"NOUN", "PROPN"}:
        # Get the basic noun phrase
        full_phrase = noun_phrase_label(token, include_det)

        # Look for "of" prepositional phrases that indicate type relationships
        for prep in token.children:
            if prep.dep_ == "prep" and prep.text.lower() == "of":
                for pobj in prep.children:
                    if pobj.dep_ == "pobj" and pobj.pos_ in {"NOUN", "PROPN"}:
                        # This is likely the core type (e.g., "skin cancer" from "type of skin cancer")
                        core_type = noun_phrase_label(pobj, include_det)
                        results.append(core_type)

        # Also include the head noun itself (e.g., "type")
        head_noun = token.text
        if not any(adj.pos_ == "ADJ" and adj.lemma_ in {"common", "large", "big", "small", "most"}
                   for adj in token.lefts):
            # Only include head if it's not just a superlative modifier
            results.append(head_noun)

        # If no "of" relationship found, use the full phrase but try to clean it
        if not results:
            # Remove superlative modifiers for cleaner semantic relationships
            cleaned = full_phrase
            superlative_patterns = ["most common ", "largest ", "biggest ", "smallest ", "most "]
            for pattern in superlative_patterns:
                if cleaned.lower().startswith(pattern):
                    cleaned = cleaned[len(pattern):]
            results.append(cleaned)

    return results if results else [token.text]

# -------- graph build/plot --------
def build_graph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_node(h); G.add_node(t)
        G.add_edge(h, t, rel=r)
    return G

def plot_graph(G, title=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=2400, font_size=10, font_weight="bold", arrows=True, arrowsize=18)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'rel'), font_size=9)
    if title: plt.title(title)
    plt.tight_layout(); plt.show()


# ---------- ) JSON with ID + Dictionary ----------
def triples_to_id_dictionary(triples,tasks = 'answer the questions'):
    """
    triples: set or list of (head, rel, tail)
    Return:
      {
        "entity_dict": [...],        # index = entity_id
        "relation_dict": [...],      # index = relation_id
        "edges": [[e_id, r_id, e_id], ...],
        "tasks":'answer the questions'
      }
    """
    ent2id, rel2id = {}, {}
    entity_dict, relation_dict = [], []
    edges = []

    def _eid(x):
        if x not in ent2id:
            ent2id[x] = len(entity_dict)
            entity_dict.append(x)
        return ent2id[x]

    def _rid(x):
        if x not in rel2id:
            rel2id[x] = len(relation_dict)
            relation_dict.append(x)
        return rel2id[x]

    for h, r, t in triples:
        h_id = _eid(h)
        r_id = _rid(r)
        t_id = _eid(t)
        edges.append([h_id, r_id, t_id])

    return {"entity_dict": entity_dict, "relation_dict": relation_dict, "edges": edges,"tasks":tasks}


# ---------- Utility ----------
def json_dump_str(obj, indent=0):
    """Return compact JSON string by default; pretty-print if indent>0."""
    if indent:
        return json.dumps(obj, ensure_ascii=False, indent=indent)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))




def all_chains_no_subchains(edges, use_full_edges=True):
    chain = [edges]
    if use_full_edges:
        return chain
    else:
        return [[i for i in range(len(edges))]]

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

def json_dump_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

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



def get_topk_word_embedding_batched_cross_sim(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    top_k: int = 3,
    question_batch_size: int = 1,
    questions_db_batch_size: int = 1,
    w_ent: float = 1.0,
    w_rel: float = 0.3,
    target = 'questions'
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Uses (entities pooled: heads+tails) and relations from each edge_run, scored by:
        score = w_ent * max_{entity pair} cos + w_rel * max_{relation pair} cos
    Returns the same structure as before.
    """

    # 0) ensure embeddings exist in the codebook (same as old path)
    _ensure_embeddings_in_codebook(codebook_main, dim_fallback=64)

    results: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(questions))}

    # Decide DB source (answers vs historical questions), same as before

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

    # Flatten DB runs
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

    # Pre-extract (E,R) for DB once
    db_ER: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [
        _extract_entities_relations_from_run(edge_run, codebook_main)
        for edge_run in db_questions
    ]

    for q_start in range(0, N_total, question_batch_size):
        q_end = min(q_start + question_batch_size, N_total)
        q_batch_idx = list(range(q_start, q_end))

        # Extract (E,R) for this query batch
        q_ER: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [
            _extract_entities_relations_from_run(questions[i], codebook_main)
            for i in q_batch_idx
        ]

        # Keep running top-k per query row as before
        best_scores = [np.array([], dtype=np.float32) for _ in q_batch_idx]
        best_cols   = [np.array([], dtype=np.int32)   for _ in q_batch_idx]

        for db_start in range(0, M_total, questions_db_batch_size):
            db_end = min(db_start + questions_db_batch_size, M_total)

            # Compute a (len(q_batch) x (db_end-db_start)) similarity block
            block_sims = np.empty((len(q_batch_idx), db_end - db_start), dtype=np.float32)

            for i, (Eq, Rq) in enumerate(q_ER):
                # Fill this row against current DB slice
                for j, (Ef, Rf) in enumerate(db_ER[db_start:db_end]):
                    block_sims[i, j] = entrel_maxpair_similarity(
                        Eq, Rq, Ef, Rf, w_ent=w_ent, w_rel=w_rel
                    )

            # Merge into global top-k per row
            # top k chunks
            k_local = min(top_k, db_end - db_start)
            for i in range(len(q_batch_idx)):
                row = block_sims[i]
                # same selection logic as old
                cand_idx = np.argpartition(-row, k_local - 1)[:k_local]
                cand_idx = cand_idx[np.argsort(-row[cand_idx])]
                batch_scores = row[cand_idx]
                batch_cols   = cand_idx + db_start
                merged_scores, merged_cols = _topk_merge(
                    best_scores[i], best_cols[i], batch_scores, batch_cols, top_k
                )
                best_scores[i], best_cols[i] = merged_scores, merged_cols

        # Write results for this batch (same schema)
        for loc_i, gq_idx in enumerate(q_batch_idx):
            cols = best_cols[loc_i]; scs = best_scores[loc_i]
            keep = (cols >= 0)
            cols, scs = cols[keep], scs[keep]
            entries = []
            for col, sc in zip(cols, scs):
                entries.append({
                    "score": float(sc),
                    "questions_index": int(db_qi[col]),
                    "question_index": int(db_qj[col]),
                    "db_source": db_source,
                })
            results[gq_idx] = entries

    return results

# --- tiny utilities for rerank ---
def _l2norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def _triples_words(edge_run: List[int], codebook_main: Dict[str, Any]) -> List[List[str]]:
    """Returns [[h, r, t], ...] using your decoder in 'words' mode."""
    return decode_question(edge_run, codebook_main, fmt='words')

def _embed_lines(lines: List[str], emb) -> np.ndarray:
    vecs = emb.embed_documents(lines)  # HuggingFaceEmbeddings-compatible
    return np.asarray(vecs, dtype=np.float32)


# --- scorer with attention row-weights---

# used cached chunk embeddingsß
def _score_query_vs_chunk_with_bonus_optimized(
    questions,
    coarse_topk,
    chunk_selected,                         # (qi, qj, src)
    q_edges_embeddings_lst_dict,            # i -> (nq_i,d)
    questions_embeddings,                   # (N,d) L2-normalized or None
    codebook_main,
    emb,
    q_triple_attention=None,                # i -> (nq_i,) sums to 1
    top_t: int = 3,
    cov_tau: float = 0.45, cov_weight: float = 0.10,
    pair_tau: float = 0.55, pair_temp: float = 0.10, pair_weight: float = 0.20, pair_norm: str = "sqrt",
    distinct_weight: float = 0.10, distinct_tau: float = 0.50,
    whole_weight: float = 0.10,
    whole_gate_tau: float = 0.55,
    whole_gate_temp: float = 0.15,
    whole_len_norm: str = "sqrt_nc"
) -> float:
    import numpy as np

    def _l2norm_rows(X, eps=1e-12):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    def _cosine_sim_matrix(A, B):
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
        return _l2norm_rows(A) @ _l2norm_rows(B).T

    sel_qi, sel_qj, sel_src = chunk_selected

    # which questions nominated this chunk
    voters = []
    for i, _ in enumerate(questions):
        for it in coarse_topk.get(i, []):
            key = (int(it["questions_index"]), int(it["question_index"]), it.get("db_source", "questions"))
            if key == chunk_selected:
                voters.append(i); break
    if not voters:
        return 0.0

    # chunk edges by source
    # using the cache embeddings
    if sel_src == "questions":
        C_full = codebook_main['questions_lst_embedding'][sel_qi][sel_qj]
        chunk_edges = codebook_main['questions_lst'][sel_qi][sel_qj]
    elif sel_src == "answers":
        C_full = codebook_main['answers_lst_embedding'][sel_qi][sel_qj]
        chunk_edges = codebook_main['answers_lst'][sel_qi][sel_qj]
    else:
        C_full = codebook_main['facts_lst_embedding'][sel_qi][sel_qj]
        chunk_edges = codebook_main['facts_lst'][sel_qi][sel_qj]

    # using the cache embeddings
    C = np.array([codebook_main['edge_matrix_embedding'][edge_i] for edge_i in chunk_edges], dtype=np.float32)

    c_triples = _triples_words(chunk_edges, codebook_main)

    S_list, A_list = [], []
    for i in voters:
        Q = np.asarray(q_edges_embeddings_lst_dict.get(i, np.zeros((0, C.shape[1]), dtype=np.float32)), dtype=np.float32)
        if Q.size == 0:
            continue
        S = _cosine_sim_matrix(Q, C)     # (nq_i, nc)
        S_list.append(S)

        # attention vector per question i (aligns to rows of Q)
        if q_triple_attention is not None and i in q_triple_attention and q_triple_attention[i].size:
            a = np.asarray(q_triple_attention[i], dtype=np.float32).ravel()
            if a.size != S.shape[0]:
                # trim or pad uniformly
                if a.size > S.shape[0]:
                    a = a[:S.shape[0]]
                else:
                    pad = np.full((S.shape[0] - a.size,), 1.0 / S.shape[0], dtype=np.float32)
                    a = np.concatenate([a, pad])
        else:
            a = np.full((S.shape[0],), 1.0 / max(1, S.shape[0]), dtype=np.float32)

        a = a / (a.sum() + 1e-12)
        A_list.append(a)

    if not S_list:
        return 0.0

    def score_from_S(S, a):
        nq, nc = S.shape
        if nq == 0 or nc == 0:
            return 0.0

        S_attn = (a[:, None] * S).astype(np.float32)

        # adaptive top-t mean (weighted matrix)
        t_pairs = max(1, min(top_t, nq * nc))
        flat = S_attn.ravel()
        idx = np.argpartition(flat, -t_pairs)[-t_pairs:]
        rel = float(flat[idx].mean())

        # attention-weighted coverage (threshold on true best row sims)
        best_per_q = S.max(axis=1)
        covered = (best_per_q >= cov_tau).astype(np.float32)
        coverage = float((covered * a).sum())

        # many-to-many soft count
        soft_hits = 1.0 / (1.0 + np.exp(-(S_attn - pair_tau) / max(1e-6, pair_temp)))
        good_pairs_soft = float(soft_hits.sum())
        if pair_norm == "sqrt":
            norm = np.sqrt(max(1, nq * nc))
        elif pair_norm == "log":
            norm = np.log1p(max(1, nq * nc))
        else:
            norm = 1.0
        good_pairs_bonus = np.log1p(good_pairs_soft / (norm + 1e-12))

        # distinct greedy on weighted matrix
        distinct_bonus = 0.0
        if distinct_weight > 0.0:
            S_work = S_attn.copy()
            taken = 0
            for _ in range(min(nq, nc)):
                r, c = divmod(int(S_work.argmax()), S_work.shape[1])
                val = S_work[r, c]
                if val < distinct_tau:
                    break
                distinct_bonus += float(val)
                S_work[r, :] = -np.inf
                S_work[:, c] = -np.inf
                taken += 1
            if taken > 0:
                distinct_bonus /= np.sqrt(taken)

        return rel + cov_weight * coverage + pair_weight * good_pairs_bonus + distinct_weight * distinct_bonus

    score = sum(score_from_S(S, a) for S, a in zip(S_list, A_list))

    # whole-question vs whole-chunk bonus (averaged over voters)
    if questions_embeddings is not None and whole_weight > 0.0:
        from numpy import mean, sqrt, log1p
        vc = _l2norm_rows(C_full)
        vq_all = _l2norm_rows(questions_embeddings)
        sims = [float(vq_all[i:i+1, :] @ vc.T) for i in voters]
        if sims:
            s_whole = float(mean(sims))
            nc = len(c_triples)
            if whole_len_norm == "sqrt_nc":
                len_factor = np.sqrt(nc) + 1e-12
            elif whole_len_norm == "log_nc":
                len_factor = np.log1p(nc) + 1e-12
            else:
                len_factor = 1.0
            s_whole_adj = s_whole / len_factor
            whole_gate = 1.0 / (1.0 + np.exp(-(s_whole - whole_gate_tau) / max(1e-6, whole_gate_temp)))
            score += whole_weight * (whole_gate * s_whole_adj)

    return float(score)


# --- reranker that RETURNS the dict-of-lists your helpers expect ---
def rerank_with_sentence_embeddings_score_with_coverage_optimized(
    questions,
    codebook_main,
    coarse_topk,
    emb,
    top_m: int = 1,
    use_attention: bool = False
):
    import numpy as np

    # build one whole-question vector per question (L2-normalized)
    def _build_whole_question_embeddings(questions, codebook_main, emb, sep=" <SEP> "):
        rows = []
        d = None
        for q_edges in questions:
            q_triples = _triples_words(q_edges, codebook_main)
            if not q_triples:
                rows.append(np.zeros((1, 1), dtype=np.float32)); continue
            txt = sep.join([f"[H]{h} [R]{r} [T]{t}" for h, r, t in q_triples])
            v = _embed_lines([txt], emb)  # (1,d)
            d = v.shape[1] if d is None else d
            rows.append(v)
        if d is None:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [r if r.shape[1] == d else np.zeros((1, d), dtype=np.float32) for r in rows]
        V = np.vstack(rows).astype(np.float32)
        # L2 norm rows
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    questions_embeddings = _build_whole_question_embeddings(questions, codebook_main, emb)  # (N,d)

    # per-question per-triple embeddings + attention
    q_edges_embeddings_lst_dict = {}
    q_triple_attention = {}

    for i, q_edges in enumerate(questions):
        q_triples = _triples_words(q_edges, codebook_main)
        q_lines = [f"{h} {r} {t}" for h, r, t in q_triples]
        if q_lines:
            Q = _embed_lines(q_lines, emb)
        else:
            Q = np.zeros((0, questions_embeddings.shape[1] if questions_embeddings.size else 0), dtype=np.float32)
        q_edges_embeddings_lst_dict[i] = Q

        if use_attention and Q.size > 0 and questions_embeddings.size > 0:
            vqi = questions_embeddings[i:i+1, :]           # (1,d)
            VQ  = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            sims = (VQ @ vqi.T).ravel().astype(np.float32)  # (nq_i,)
            # softmax
            s = sims - sims.max()
            w = np.exp(s, dtype=np.float64); w = (w / (w.sum() + 1e-12)).astype(np.float32)
            q_triple_attention[i] = w
        else:
            if Q.shape[0] > 0:
                q_triple_attention[i] = np.full((Q.shape[0],), 1.0 / Q.shape[0], dtype=np.float32)
            else:
                q_triple_attention[i] = np.zeros((0,), dtype=np.float32)

    # score all unique candidates (GLOBAL top-m) -> dict of lists
    all_scored = {
        "score": [],
        "questions_index": [],
        "question_index": [],
        "db_source": [],
        "index_combo": []
    }

    # Build a unique candidate set globally
    seen_global = set()
    for i, _ in enumerate(questions):
        for it in coarse_topk.get(i, []):
            key = (int(it["questions_index"]), int(it["question_index"]), it.get("db_source", "questions"))
            seen_global.add(key)

    for chunk_selected in seen_global:
        sc = _score_query_vs_chunk_with_bonus_optimized(
            questions=questions,
            coarse_topk=coarse_topk,
            chunk_selected=chunk_selected,
            q_edges_embeddings_lst_dict=q_edges_embeddings_lst_dict,
            questions_embeddings=questions_embeddings if questions_embeddings.size else None,
            codebook_main=codebook_main,
            emb=emb,
            q_triple_attention=q_triple_attention
        )
        qi, qj, src = chunk_selected
        all_scored["score"].append(float(sc))
        all_scored["questions_index"].append(qi)
        all_scored["question_index"].append(qj)
        all_scored["db_source"].append(src)
        all_scored["index_combo"].append([qi, qj])

    # take global top-m & keep the dict-of-lists SHAPE
    if all_scored["score"]:
        order = np.argsort(all_scored["score"])[::-1][:min(top_m, len(all_scored["score"]))]
        for k in list(all_scored.keys()):
            all_scored[k] = [all_scored[k][idx] for idx in order]

    return all_scored

def coarse_filter_optimized(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    sentence_emb: HuggingFaceEmbeddings,        # ← move before defaults
    top_k: int = 3,                             # word-embedding candidates
    question_batch_size: int = 1,               # query batch size
    questions_db_batch_size: int = 1,           # DB batch size
    top_m: int = 1,                             # sentence-embedding rerank
    custom_linearizer: Optional[Callable[[List[List[str]]], str]] = None,
    target = 'questions',
    w_ent: float = 1.0,
    w_rel: float = 0.3,):

    # doing the word embedding pre-filter 


    top_k_time = time.time()

    coarse_top_k = get_topk_word_embedding_batched_cross_sim(
    questions,
    codebook_main,
    top_k,
    question_batch_size,         # number of query questions processed per time
    questions_db_batch_size,     # number of db questions processed per time
    w_ent,
    w_rel,
    target
    )

    top_k_time_end = time.time() - top_k_time

    # doing the sentence embedding filter 

    top_m_time = time.time()

    top_m_results = rerank_with_sentence_embeddings_score_with_coverage_optimized(
    questions,
    codebook_main,
    coarse_top_k,
    sentence_emb,
    top_m)

    top_m_time_end = time.time() - top_m_time

    if DEBUG_VERBOSE:
        print('top_m_results',top_m_results)
        print('top_k_time_end',top_k_time_end)
        print('top_m_time_end',top_m_time_end)


    return top_m_results



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

def _build_ent_inv_index(codebook_main, target):
    """
    Incrementally maintained entity → chunk inverted index.

    Only processes groups added since the last call.  O(new_edges) per call
    instead of O(all_edges).

    For target='questions': entity_id → set of qi  (answer group indices)
    For target='facts':     entity_id → set of (qi, qj) (fact chunk indices)
    """
    cache_key = f'_ent_inv_idx_{target}'
    watermark_key = f'_ent_inv_wm_{target}'   # how many groups were indexed

    edge_matrix = codebook_main['edge_matrix']

    if target == 'questions':
        lst = codebook_main.get('answers_lst', [])
    else:
        lst = codebook_main.get('facts_lst', [])

    prev_wm = codebook_main.get(watermark_key, 0)
    inv = codebook_main.get(cache_key)

    if inv is None or prev_wm > len(lst):
        # first call or list shrank (combine_ents) → full rebuild
        inv = defaultdict(set)
        prev_wm = 0

    # only process groups from prev_wm onward
    for qi in range(prev_wm, len(lst)):
        group = lst[qi]
        if target == 'questions':
            for chunk in group:
                for eid in chunk:
                    h, _r, t = edge_matrix[int(eid)]
                    inv[h].add(qi)
                    inv[t].add(qi)
        else:
            for qj, chunk in enumerate(group):
                for eid in chunk:
                    h, _r, t = edge_matrix[int(eid)]
                    inv[h].add((qi, qj))
                    inv[t].add((qi, qj))

    codebook_main[cache_key] = inv
    codebook_main[watermark_key] = len(lst)
    return inv


def _classify_question_edges(question_edge_runs, edge_matrix, n_ents_before):
    """
    Classify question edges into dependency components.

    Grounding rule (parameter-free):
      entity_id < n_ents_before  →  GROUNDED  (existed before this question)
      entity_id >= n_ents_before →  UNGROUNDED (new from this question's LLM parse)

    Returns list of components, each:
      {
        'type':        'parallel' | 'chain',
        'run_indices': [int, ...],
        'anchor_ents': set of int,          # grounded entity ids
        'bridge_ents': set of int,          # ungrounded entity ids
      }
    """
    # Classify at per-EDGE level (not per-run) so independent edges
    # in the same run stay as separate components when they don't share
    # ungrounded entities (e.g. comparison questions with 2 films).
    run_ents = []
    for run in question_edge_runs:
        for eid in run:
            grounded = set()
            ungrounded = set()
            h, _r, t = edge_matrix[int(eid)]
            for e in (h, t):
                if e < n_ents_before:
                    grounded.add(e)
                else:
                    ungrounded.add(e)
            run_ents.append((grounded, ungrounded))

    n = len(run_ents)
    if n == 0:
        return []

    # two runs are DEPENDENT if they share an ungrounded entity
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if run_ents[i][1] & run_ents[j][1]:
                adj[i].add(j)
                adj[j].add(i)

    # connected components (BFS)
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        comp = []
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        anchors = set()
        bridges = set()
        for idx in comp:
            anchors |= run_ents[idx][0]
            bridges |= run_ents[idx][1]

        has_chain = len(bridges) > 0
        components.append({
            'type': 'chain' if has_chain else 'parallel',
            'run_indices': comp,
            'anchor_ents': anchors,
            'bridge_ents': bridges,
        })

    return components


def adaptive_graph_expand(codebook_main, top_m_results, target,
                          question_edge_runs, n_ents_before, max_hops=3):
    """
    Adaptive retriever: structural graph walk for chains, embedding pass-through
    for parallel.

    For PARALLEL components: coarse_filter_torch already found them via embedding
    similarity → keep those results as-is.

    For CHAIN components: coarse_filter_torch returns empty (the descriptive
    subject entity like "mother of the director" matches nothing).  Instead,
    use the ANCHOR entities (grounded, named entities like "Polish-Russian War")
    directly via inverted index to seed hop-1, then walk the graph for hop-2+.

    Completely parameter-free — no weights, thresholds, or temperatures.
    """
    edge_matrix = codebook_main['edge_matrix']

    # ---- step 1: classify question structure ----
    components = _classify_question_edges(
        question_edge_runs, edge_matrix, n_ents_before)

    has_chain = any(c['type'] == 'chain' for c in components)
    if not has_chain:
        return top_m_results

    # ---- step 2: build entity inverted index (cached, incremental) ----
    ent_inv = _build_ent_inv_index(codebook_main, target)

    if target == 'questions':
        answers_lst = codebook_main.get('answers_lst', [])
        questions_lst = codebook_main.get('questions_lst', [])
        cur_q_idx = len(questions_lst) - 1
    else:
        facts_lst = codebook_main.get('facts_lst', [])

    def _collect_entities(result_qis, result_chunks):
        ents = set()
        if target == 'questions':
            for qi in result_qis:
                if qi < len(answers_lst):
                    for chunk in answers_lst[qi]:
                        for eid in chunk:
                            h, _r, t = edge_matrix[int(eid)]
                            ents.add(h)
                            ents.add(t)
        else:
            for qi, qj in result_chunks:
                if qi < len(facts_lst) and qj < len(facts_lst[qi]):
                    for eid in facts_lst[qi][qj]:
                        h, _r, t = edge_matrix[int(eid)]
                        ents.add(h)
                        ents.add(t)
        return ents

    # ---- step 3: seed hop-1 from ANCHOR entities of chain components ----
    #   This replaces coarse_filter_torch for chains — direct entity ID lookup,
    #   no embeddings needed.
    anchor_ents = set()
    for comp in components:
        if comp['type'] == 'chain':
            anchor_ents |= comp['anchor_ents']

    # Inverted index lookup: anchor entities → chunks
    seed_qis = set()
    seed_chunks = set()
    seed_candidates = defaultdict(int)  # key → overlap count with anchors

    for ent_id in anchor_ents:
        hits = ent_inv.get(ent_id, set())
        if target == 'questions':
            for qi in hits:
                if qi < cur_q_idx:
                    seed_candidates[qi] += 1
        else:
            for key in hits:
                seed_candidates[key] += 1

    # budget: use the configured top_m equivalent from coarse results, or default 20
    budget_per_hop = max(len(top_m_results.get('score', [])), 5, 20)
    total_budget = budget_per_hop * max_hops

    # rank seed by overlap with anchor entities, take top budget
    seed_ranked = sorted(seed_candidates.items(), key=lambda x: -x[1])[:budget_per_hop]

    for key, _cnt in seed_ranked:
        if target == 'questions':
            seed_qis.add(key)
        else:
            seed_chunks.add(key)

    n_seed = len(seed_qis) + len(seed_chunks)
    if n_seed > 0:
        print(f"[adaptive_expand] {target}: seeded {n_seed} hop-1 chunks "
              f"from {len(anchor_ents)} anchor entities (inverted index)")

    # ---- step 4: merge seed with coarse results (coarse may have found parallel hits) ----
    coarse_qis = set()
    coarse_chunks = set()
    for combo in top_m_results.get('index_combo', []):
        qi, qj = int(combo[0]), int(combo[1])
        coarse_qis.add(qi)
        coarse_chunks.add((qi, qj))

    all_qis = coarse_qis | seed_qis
    all_chunks = coarse_chunks | seed_chunks

    # ---- step 5: iterative hop expansion from merged seed ----
    query_ents = set()
    for run in question_edge_runs:
        for eid in run:
            h, _r, t = edge_matrix[int(eid)]
            query_ents.add(h)
            query_ents.add(t)

    seen_ents = set(query_ents)
    frontier_qis = set(all_qis)
    frontier_chunks = set(all_chunks)
    expansion_ranked = []

    for hop in range(max_hops):
        new_ents = _collect_entities(frontier_qis, frontier_chunks)
        bridge_ents = new_ents - seen_ents
        seen_ents |= bridge_ents

        if not bridge_ents:
            break

        # breadth guard
        n_total_ents = len(codebook_main.get('e', []))
        if n_total_ents > 0 and len(bridge_ents) > n_total_ents * 0.2:
            print(f"[adaptive_expand] hop {hop+1}: {len(bridge_ents)} bridge entities "
                  f"= {len(bridge_ents)/n_total_ents:.0%} of codebook -- skipping (too broad)")
            break

        candidates = defaultdict(float)
        import math
        for ent_id in bridge_ents:
            hits = ent_inv.get(ent_id, set())
            # IDF weight: specific entities (fewer inv entries) rank higher
            idf = 1.0 / math.log2(2 + len(hits))
            if target == 'questions':
                for qi in hits:
                    if qi not in all_qis and qi < cur_q_idx:
                        candidates[qi] += idf
            else:
                for key in hits:
                    if key not in all_chunks:
                        candidates[key] += idf

        if not candidates:
            break

        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:budget_per_hop]

        next_frontier_qis = set()
        next_frontier_chunks = set()
        for key, cnt in ranked:
            expansion_ranked.append((key, cnt))
            if target == 'questions':
                all_qis.add(key)
                next_frontier_qis.add(key)
            else:
                all_chunks.add(key)
                next_frontier_chunks.add(key)

        frontier_qis = next_frontier_qis
        frontier_chunks = next_frontier_chunks

        if not frontier_qis and not frontier_chunks:
            break
        if len(expansion_ranked) >= total_budget:
            break

        print(f"[adaptive_expand] hop {hop+1}: {len(bridge_ents)} bridge entities "
              f"-> {len(ranked)} new chunks (top overlap={ranked[0][1] if ranked else 0})")

    # ---- step 6: build final result dict ----
    # Start from coarse results (parallel hits), append seed + expansion
    result = {k: list(v) for k, v in top_m_results.items()}
    db_src = target
    if result.get('db_source'):
        db_src = result['db_source'][0]

    # collect what coarse already has, to avoid duplicates
    existing = set()
    for combo in result.get('index_combo', []):
        existing.add((int(combo[0]), int(combo[1])))

    all_to_add = []
    # seed chunks (hop-1 from anchor entities)
    for key, cnt in seed_ranked:
        if target == 'questions':
            entry = (key, 0)
        else:
            entry = key
        if entry not in existing:
            all_to_add.append(entry)
            existing.add(entry)
    # expansion chunks (hop-2+)
    for key, cnt in expansion_ranked:
        if target == 'questions':
            entry = (key, 0)
        else:
            entry = key
        if entry not in existing:
            all_to_add.append(entry)
            existing.add(entry)

    for entry in all_to_add[:total_budget]:
        qi, qj = entry if isinstance(entry, tuple) and len(entry) == 2 else (entry, 0)
        result['score'].append(0.0)
        result['questions_index'].append(qi)
        result['question_index'].append(qj)
        result['db_source'].append(db_src)
        result['index_combo'].append([qi, qj])

    n_added = len(all_to_add[:total_budget])
    chain_comps = [c for c in components if c['type'] == 'chain']
    par_comps = [c for c in components if c['type'] == 'parallel']
    print(f"[adaptive_expand] {target}: "
          f"{len(par_comps)} parallel + {len(chain_comps)} chain components, "
          f"coarse={len(top_m_results.get('score',[]))}, "
          f"seed={len(seed_ranked)}, hop-expand={len(expansion_ranked)}, "
          f"added={n_added}")

    return result


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

def _all_contiguous_subseqs(seq, min_len=2):
    """Generate all contiguous subsequences of seq with length >= min_len."""
    n = len(seq)
    for i in range(n):
        for j in range(i + min_len, n + 1):
            yield tuple(seq[i:j])

def _is_subrun(a, b):
    """Check if subsequence a is fully inside b (both are tuples)."""
    return len(a) < len(b) and any(
        b[i:i+len(a)] == a for i in range(len(b) - len(a) + 1)
    )

def common_contiguous_overlaps_naive(answers_lst, min_len=2):
    """
    Naïve version: find all maximal common contiguous subsequences.
    """
    if not answers_lst:
        return []

    candidates = set(_all_contiguous_subseqs(answers_lst[0], min_len=min_len))

    for lst in answers_lst[1:]:
        runs_here = set(_all_contiguous_subseqs(lst, min_len=min_len))
        candidates &= runs_here
        if not candidates:
            return []

    maximal = set(candidates)
    for a in list(candidates):
        for b in candidates:
            if a != b and _is_subrun(a, b):
                maximal.discard(a)
                break

    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]

# Rolling-hash-based approach
def common_contiguous_overlaps_hash(lists, min_len=2):
    """
    Corrected hash-based implementation that matches naive semantics.
    Finds maximal common contiguous subsequences in all lists.
    """
    if not lists:
        return []
    if len(lists) == 1:
        return [lists[0]]

    # Collect all candidates from the first list
    first = lists[0]
    candidates = set()
    for i in range(len(first)):
        for j in range(i + min_len, len(first) + 1):
            candidates.add(tuple(first[i:j]))

    # Intersect with candidates from the rest
    for lst in lists[1:]:
        runs_here = set()
        for i in range(len(lst)):
            for j in range(i + min_len, len(lst) + 1):
                runs_here.add(tuple(lst[i:j]))
        candidates &= runs_here
        if not candidates:
            return []

    # Keep only maximal runs
    maximal = set(candidates)
    for a in list(candidates):
        for b in candidates:
            if a != b and len(a) < len(b) and any(
                tuple(b[i:i+len(a)]) == a for i in range(len(b) - len(a) + 1)
            ):
                maximal.discard(a)
                break

    # Sort by length desc, then lex for determinism
    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]

from collections import defaultdict

def common_contiguous_overlaps_advanced(lists, min_len=2, min_support_ratio=0.7):
    """
    Find maximal contiguous subsequences that appear in at least `min_support` lists.
    """
    if not lists:
        return []

    min_support = max(1, round(len(lists) * min_support_ratio))
    if DEBUG_VERBOSE:
        print("min_support: ", min_support)

    # Collect all candidates from all lists
    candidate_counts = defaultdict(set)  # subseq -> set of list indices
    for idx, lst in enumerate(lists):
        for i in range(len(lst)):
            for j in range(i + min_len, len(lst) + 1):
                subseq = tuple(lst[i:j])
                candidate_counts[subseq].add(idx)

    # Keep only those with enough support
    frequent = {subseq for subseq, idxs in candidate_counts.items() if len(idxs) >= min_support}

    # Filter to maximal
    maximal = set(frequent)
    for a in list(frequent):
        for b in frequent:
            if a != b and len(a) < len(b):
                if any(tuple(b[i:i+len(a)]) == a for i in range(len(b)-len(a)+1)):
                    maximal.discard(a)
                    break

    # Return sorted list
    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]

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

def get_json_with_given_knowledge_and_thinkings(flat_answers_lsts,flat_thinkings_lsts,codebook_main,codebook_sub_q,decode = True):
    # used flat here since trying to flat answers for each answers trunk to get longer overlapp
    # if change the answers here also change the format for other func related

    # get all unique edges

    all_unique_edges_mat_indexes = list(set([x for sublist in flat_answers_lsts for x in sublist]+[x for sublist in flat_thinkings_lsts for x in sublist]))

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
    flat_thinkings_lsts = [[edge_mat_index_dict.get(x, x) for x in sublist] for sublist in flat_thinkings_lsts]

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
    edge_matrix_sub_len = len(edge_matrix_sub)
    entitie_index_dict_q = {}
    for ent_pos, ent in enumerate(codebook_sub_q['e']):
        if ent in entitie_set:
            new_ent_pos = entitie_set.index(ent)
        else:
            new_ent_pos = len(entitie_set)   # ← 用当前长度作为新索引
            entitie_set.append(ent)          # ← 再 append
        entitie_index_dict_q[ent_pos] = new_ent_pos

    # update relation index
    r_index_dict_q = {}
    for r_pos, r in enumerate(codebook_sub_q['r']):
        if r in r_set:
            new_r_pos = r_set.index(r)
        else:
            new_r_pos = len(r_set)
            r_set.append(r)
        r_index_dict_q[r_pos] = new_r_pos

    # map the q edge matrix
    edge_mat_for_q_sub = remap_edges(codebook_sub_q['edges([e,r,e])'], entitie_index_dict_q, r_index_dict_q)

    # update the edges
    edge_matrix_sub_len = len(edge_matrix_sub)
    edge_pos = 0
    edge_mat_for_q_sub_dict = {}

    edge_matrix_sub_len = len(edge_matrix_sub)
    for edge in edge_mat_for_q_sub:
        if edge in edge_matrix_sub:
            new_edge_pos = edge_matrix_sub.index(edge)
        else:
            new_edge_pos = edge_matrix_sub_len    # ✅ 先用当前长度作为新索引
            edge_matrix_sub.append(edge)          # ✅ 再 append
            edge_matrix_sub_len += 1              # ✅ 最后长度+1
        edge_mat_for_q_sub_dict[edge_pos] = new_edge_pos
        edge_pos += 1


    # update the questions
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
        'start thinking with(edges[i])':flat_thinkings_lsts,
        'rule':codebook_sub_q['rule']
    }

    if decode:
        final_merged_json = {
            'e':entitie_set,
            'r':r_set,
            'edge_matrix':edge_matrix_sub,
            'questions([[e,r,e], ...])':decode_questions(questions, final_merged_json, 'edges'),
            'given knowledge([[e,r,e], ...])': decode_questions(flat_answers_lsts, final_merged_json, 'edges'),
            'start thinking with([[e,r,e], ...])':decode_questions(flat_thinkings_lsts,final_merged_json,'edges'),
            'rule':codebook_sub_q['rule']

        }


    return final_merged_json


def get_json_with_given_thinkings(flat_thinkings_lsts,codebook_main,codebook_sub_q,decode = True):
    # used flat here since trying to flat answers for each answers trunk to get longer overlapp
    # if change the answers here also change the format for other func related

    # get all unique edges

    all_unique_edges_mat_indexes = list(set([x for sublist in flat_thinkings_lsts for x in sublist]))

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

    # update edge index in flat_thinkings_lsts
    flat_thinkings_lsts = [[edge_mat_index_dict.get(x, x) for x in sublist] for sublist in flat_thinkings_lsts]

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
            entitie_set_len += 1  ## edge_matrix_sub_len += 1

        entitie_index_dict_q[ent_pos] = new_ent_pos

        ent_pos+=1

    # update relation index
    r_pos = 0
    for r in codebook_sub_q['r']:
        if r in r_set:
            new_r_pos = r_set.index(r)
        else:
            r_set_len+=1
            new_r_pos = r_set_len
            r_set.append(r)

        r_index_dict_q[r_pos] = new_r_pos

        r_pos+=1

    # map the q edge matrix
    edge_mat_for_q_sub = remap_edges(codebook_sub_q['edges([e,r,e])'], entitie_index_dict_q, r_index_dict_q)

    # update the edges
    edge_matrix_sub_len = len(edge_matrix_sub)
    edge_pos = 0
    edge_mat_for_q_sub_dict = {}
    for edge_pos, edge in enumerate(edge_mat_for_q_sub):
        if edge in edge_matrix_sub:
            new_edge_pos = edge_matrix_sub.index(edge)
        else:
            new_edge_pos = edge_matrix_sub_len
            edge_matrix_sub.append(edge)
            edge_matrix_sub_len += 1
        edge_mat_for_q_sub_dict[edge_pos] = new_edge_pos


    # update the questions
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
        'start thinking with(edges[i])':flat_thinkings_lsts,
        'rule':codebook_sub_q['rule']
    }

    if decode:
        final_merged_json = {
            'e':entitie_set,
            'r':r_set,
            'edge_matrix':edge_matrix_sub,
            'questions([[e,r,e], ...])':decode_questions(questions, final_merged_json, 'edges'),
            'start thinking with([[e,r,e], ...])':decode_questions(flat_thinkings_lsts,final_merged_json,'edges'),
            'rule':codebook_sub_q['rule']

        }


    return final_merged_json

def combine_ents(codebook_main: Dict[str, Any],
                 min_exp_num: int = 2,   # 每个簇期望最少候选数
                 max_exp_num: int = 20,  # 每个簇期望最多候选数
                 use_thinking: bool = True,
                 random_state: int = 0) -> Dict[str, Any]:

    E = list(codebook_main.get('e', []))
    X = np.asarray(codebook_main.get('e_embeddings', []), dtype=np.float32)

    n = X.shape[0]
    # 没有可并的情况
    if n <= 2:
        # 防止类型跑偏：统一成 list
        codebook_main['e'] = list(E)
        codebook_main['e_embeddings'] = [np.asarray(v, dtype=np.float32) for v in X]
        codebook_main['edge_matrix'] = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
        return codebook_main

    # L2 归一化（与 KMeans 质心空间一致）
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # 选择聚类数 k（合理边界+silhouette 优先、elbow 次之）
    k_low  = max(2, int(np.ceil(n / max_exp_num)))
    k_high = max(2, min(n - 1, int(np.floor(n / min_exp_num))))
    if k_low > k_high:  # 极端情况下兜底
        k_low, k_high = 2, max(2, min(n - 1, 5))
    cand_ks = list(range(k_low, k_high + 1))

    best_k, best_sil, inertia_by_k = None, -1.0, {}
    for k in cand_ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_norm)
        sil = silhouette_score(X_norm, labels, metric='euclidean')
        inertia_by_k[k] = km.inertia_
        if (sil > best_sil) or (np.isclose(sil, best_sil) and inertia_by_k[k] < inertia_by_k.get(best_k, np.inf)):
            best_sil, best_k = sil, k

    # 最终聚类
    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_norm)
    centroids = km.cluster_centers_  # 与 X_norm 一致的空间

    # 为每个簇选一个代表（与质心最近，使用归一化空间）
    rep_set = set()
    old_to_rep: Dict[int, int] = {}
    for c in range(best_k):
        idxs = np.where(labels == c)[0]
        pts  = X_norm[idxs]
        d    = np.linalg.norm(pts - centroids[c], axis=1)
        rep  = idxs[int(np.argmin(d))]
        rep_set.add(rep)
        for i in idxs:
            old_to_rep[i] = rep

    # 新实体下标重排（仅保留代表）
    kept_indices = sorted(rep_set)
    rep_to_new: Dict[int, int] = {old: new for new, old in enumerate(kept_indices)}
    # 每个旧实体映射到新实体下标
    old_ent_to_new: Dict[int, int] = {i: rep_to_new[old_to_rep[i]] for i in range(n)}

    # 生成新的实体与向量（保持 list 类型）
    new_e = [E[i] for i in kept_indices]
    new_e_emb = [np.asarray(codebook_main['e_embeddings'][i], dtype=np.float32) for i in kept_indices]

    # 处理边：按新实体映射，并去重，同时记录 旧边idx→新边idx 的映射
    old_edges = [list(map(int, e)) for e in codebook_main.get('edge_matrix', [])]
    tuple_to_new_edge_idx: Dict[Tuple[int,int,int], int] = {}
    new_edges: List[List[int]] = []
    old_edge_to_new_edge: Dict[int, int] = {}

    for old_idx, (e1, r, e2) in enumerate(old_edges):
        ne1 = old_ent_to_new.get(e1, e1)
        ne2 = old_ent_to_new.get(e2, e2)
        tup = (ne1, int(r), ne2)
        if tup not in tuple_to_new_edge_idx:
            tuple_to_new_edge_idx[tup] = len(new_edges)
            new_edges.append([ne1, int(r), ne2])
        old_edge_to_new_edge[old_idx] = tuple_to_new_edge_idx[tup]

    # 重写 questions/answers/thinkings 的边索引
    def remap_edge_indices(struct):
        if isinstance(struct, list):
            return [remap_edge_indices(x) for x in struct]
        # 叶子：认为是 int 的旧边下标
        try:
            return old_edge_to_new_edge.get(int(struct), int(struct))
        except (ValueError, TypeError):
            return struct

    if codebook_main.get('questions_lst') is not None:
        codebook_main['questions_lst'] = remap_edge_indices(codebook_main['questions_lst'])
    if codebook_main.get('answers_lst') is not None:
        codebook_main['answers_lst'] = remap_edge_indices(codebook_main['answers_lst'])
    if use_thinking and codebook_main.get('thinkings_lst') is not None:
        codebook_main['thinkings_lst'] = remap_edge_indices(codebook_main['thinkings_lst'])

    # 回写（统一成 list）
    codebook_main['e'] = list(new_e)
    codebook_main['e_embeddings'] = list(new_e_emb)
    codebook_main['edge_matrix'] = [list(map(int, e)) for e in new_edges]

    return codebook_main

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


def skip_func(a,b):
    return None

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
    def collect_results_dpo(self, final_merged_json, questions, retrieval_time: float = 0.0):
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
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')
            if is_no_answer_text(a_new):
                a_new_json["edges([e,r,e])"] = []
            new_json_lst.append(a_new_json)

        metrics_from_llm = llm.last_metrics

        return new_result,new_json_lst,metrics_from_llm
    
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
    def run_work_flow_for_dpo(self, q_prompt, rule="Answer questions", facts_json_path: str = None, chunk_chars: int = 1024, overlap: int = 0, warm_start = "knn"): #coarse

        #prevent dpo change choice but not change includings
        self.set_includings()
        q_json = self.encode_question(q_prompt, rule)
  
        combined_facts_cb = None
        if not getattr(self, "_facts_preloaded", False) and facts_json_path:
            self.load_and_merge_facts(facts_json_path, chunk_chars, overlap)
            self._facts_preloaded = True

        if self.meta_codebook:
            t0 = time.perf_counter()
            all_answers, all_q_indices, all_facts = self.retrieve_new(q_json)
            retrieval_time = time.perf_counter() - t0
            if DEBUG_VERBOSE:
                print("all_answers", all_answers)
                print("all_q_indices", all_q_indices)
                print("all_facts", all_facts)
            domain_knowledge_lst = self.find_related_knowledge(all_answers, all_q_indices, all_facts)
            if DEBUG_VERBOSE:
                print("domain_knowledge_lst", domain_knowledge_lst)
                print(f'q_json is {q_json}')
            final_merged_json = self.compact_indicies_for_prompt(q_json, domain_knowledge_lst)
        else:
            final_merged_json = combined_facts_cb if combined_facts_cb else q_json.copy()
            retrieval_time = 0

        if DEBUG_VERBOSE:
            print(f'final_merged_json unsliced{final_merged_json}')

        q_txt, gk_txt, st_txt, ft_txt = select_best_context_by_keys(final_merged_json)
        
        final_merged_json = slice_for_final_merged_json(final_merged_json,self.use_word)

        if gk_txt:
            self.cur_fact_context = ft_txt+gk_txt
        else:
            self.cur_fact_context = ft_txt

        if DEBUG_VERBOSE:
            print(f'final_merged_json sliced{final_merged_json}')
        new_result, new_json_lst,metrics_from_llm = self.collect_results_dpo(final_merged_json, questions=q_prompt, retrieval_time=retrieval_time)

        self.update_meta(new_json_lst, facts_cb=combined_facts_cb)

        # self.combine_ents_func(mode=warm_start)

        # if "facts_lst" in self.meta_codebook and len(self.meta_codebook["facts_lst"]) >= 2:
        #     if "facts_lst_embedding" in self.meta_codebook:
        #         f, e = dedup_facts_and_embeddings(
        #             self.meta_codebook["facts_lst"],
        #             self.meta_codebook["facts_lst_embedding"],
        #         )
        #         self.meta_codebook["facts_lst"] = f
        #         self.meta_codebook["facts_lst_embedding"] = e
        #     else:
        #         self.meta_codebook["facts_lst"] = remove_duplicate_inner_lists(self.meta_codebook["facts_lst"])

        return new_result,metrics_from_llm,ft_txt
    
    def record_labeled_q_and_a(self, questions, answers):
        """Record labeled questions and answers into the meta_codebook."""

        if len(questions) != len(answers):
            raise ValueError("Number of questions and answers must match.")

        # Process answers
        for answer in answers:
            codebook_sub = get_code_book(answer, type="answers")
            self.meta_codebook = merging_codebook(
                self.meta_codebook, codebook_sub, "answers", self.word_emb, True
            )

        # Process questions
        for question in questions:
            codebook_sub = get_code_book(question, type="questions")
            self.meta_codebook = merging_codebook(
                self.meta_codebook, codebook_sub, "questions", self.word_emb, True
            )

    def record_labeled_thinkings(self, thinkings):
        """Record labeled thinkings into the meta_codebook."""
        # Process thinkings
        for t in thinkings:
            codebook_sub_t = get_code_book(t, type='thinkings')
            self.meta_codebook = merging_codebook(
                self.meta_codebook, codebook_sub_t, "thinkings", self.word_emb, True
            )




    



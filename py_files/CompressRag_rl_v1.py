import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re
import json, hashlib
from typing import List, Tuple, Dict, Optional,Iterable,Any,Callable
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
from optimize_combine_ent import combine_ents_auto

nlp = spacy.load("en_core_web_sm")

SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJ_DEPS  = {"dobj", "obj", "attr", "oprd", "dative"}
NEG_DEPS  = {"neg"}


def get_json_with_given_facts(facts_cb, codebook_sub_q, decode: bool = True):

    fact_runs = facts_cb.get('facts(edges[i])', [])  # List[List[int]]
    flat_facts = [[x for run in (r if isinstance(r, (list, tuple)) else [r]) for x in run] for r in fact_runs]
    all_idx = sorted(set(x for sub in flat_facts for x in sub))

    edge_mat_src = facts_cb['edge_matrix']
    e_src = facts_cb['e']
    r_src = facts_cb['r']

    uniq_e_idx, uniq_r_idx = set(), set()
    edges_sub = []
    for old_i in all_idx:
        h, r, t = edge_mat_src[old_i]
        uniq_e_idx.update([h, t])
        uniq_r_idx.add(r)
        edges_sub.append([h, r, t])

    e_old2new = {old: i for i, old in enumerate(sorted(uniq_e_idx))}
    r_old2new = {old: i for i, old in enumerate(sorted(uniq_r_idx))}
    def _remap_edges(edges):
        out = []
        for h, r, t in edges:
            out.append([e_old2new[h], r_old2new[r], e_old2new[t]])
        return out

    edge_matrix_sub = _remap_edges(edges_sub)
    e_list = [e_src[old] for old in sorted(uniq_e_idx)]
    r_list = [r_src[old] for old in sorted(uniq_r_idx)]

    old_edge_to_new = {}
    for new_idx, (h, r, t) in enumerate(edge_matrix_sub):
        old_edge = edges_sub[new_idx]  
        old_edge_to_new[tuple(old_edge)] = new_idx

    def _remap_facts_runs(runs):
        out = []
        for run in runs:
            tmp = []
            for old_edge_idx in run:
                h, r, t = edge_mat_src[old_edge_idx]
                new_idx = old_edge_to_new[(h, r, t)]
                tmp.append(new_idx)
            out.append(tmp)
        return out

    facts_runs_new = _remap_facts_runs(flat_facts)

    ent2idx_q = {}
    rel2idx_q = {}

    for i, name in enumerate(e_list):
        ent2idx_q[name] = i
    for i, name in enumerate(r_list):
        rel2idx_q[name] = i

    def _ensure_ent(name):
        if name in ent2idx_q:
            return ent2idx_q[name]
        ent2idx_q[name] = len(e_list)
        e_list.append(name)
        return ent2idx_q[name]
    def _ensure_rel(name):
        if name in rel2idx_q:
            return rel2idx_q[name]
        rel2idx_q[name] = len(r_list)
        r_list.append(name)
        return rel2idx_q[name]

    edges_q_src = codebook_sub_q['edges([e,r,e])']
    e_q = codebook_sub_q['e']
    r_q = codebook_sub_q['r']


    tuple2idx = {tuple(edge): idx for idx, edge in enumerate(edge_matrix_sub)}
    def _ensure_edge(tup):
        if tup in tuple2idx:
            return tuple2idx[tup]
        idx = len(tuple2idx)
        tuple2idx[tup] = idx
        return idx

    new_edges_list = [None] * len(tuple2idx)
    for tup, idx in tuple2idx.items():
        new_edges_list[idx] = list(tup)

    def _remap_q_edge(eh, rr, et):
        h_name, r_name, t_name = e_q[eh], r_q[rr], e_q[et]
        nh = _ensure_ent(h_name)
        nr = _ensure_rel(r_name)
        nt = _ensure_ent(t_name)
        return _ensure_edge((nh, nr, nt))

    q_runs = codebook_sub_q['questions(edges[i])']
    q_runs_new = []
    for run in q_runs:
        new_run = []
        for (eh, rr, et) in [edges_q_src[i] for i in run]:
            new_idx = _remap_q_edge(eh, rr, et)
            new_run.append(new_idx)
        q_runs_new.append(new_run)

    edge_matrix = []
    for tup, idx in sorted(tuple2idx.items(), key=lambda x: x[1]):
        edge_matrix.append(list(tup))

    fin = {
        'e': e_list,
        'r': r_list,
        'edge_matrix': edge_matrix,
        'questions(edges[i])': q_runs_new,
        'given knowledge(edges[i])': facts_runs_new,
        'rule': codebook_sub_q.get('rule', 'Answer questions'),
    }
    if decode:
        fin = {
            'e': e_list,
            'r': r_list,
            'edge_matrix': edge_matrix,
            'questions([[e,r,e], ...])': decode_questions(q_runs_new, fin, 'edges'),
            'given knowledge([[e,r,e], ...])': decode_questions(facts_runs_new, fin, 'edges'),
            'rule': codebook_sub_q.get('rule', 'Answer questions'),
        }
    return fin

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
            edges = cb["edges([e,r,e])"]
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
                words = _decode_block(groups[0], final_merged_json)
                return _linearize_triples_block(words)
        return "None."

    q_txt  = _extract_txt(["questions([[e,r,e], ...])", "questions(edges[i])"])
    gk_txt = _extract_txt(["given knowledge([[e,r,e], ...])", "given knowledge(edges[i])"])
    st_txt = _extract_txt(["start thinking with([[e,r,e], ...])", "start thinking with(edges[i])"])
    ft_txt = _extract_txt(["facts([[e,r,e], ...])", "facts(edges[i])"])  

    return q_txt, gk_txt, st_txt, ft_txt

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

# -------- extraction --------
def sentence_relations(sentence, include_det=False):
    doc = nlp(sentence)
    triples = set()

    # Track processed auxiliaries to avoid double-processing
    processed_aux = set()

    for i, tok in enumerate(doc):
        # print(tok, " - ", tok.pos_, " - ", tok.dep_)
        # --- Special fix for "do/does/did" questions mis-parsed ---
        if tok.lemma_ in {"do", "does", "did"} and tok.dep_ == "ROOT":
            for child in tok.children:
                if child.dep_ == "nsubj" and child.pos_ in {"NOUN", "PROPN"}:
                    # This is actually the subject of the *next word*, not the main predicate
                    subject_np = noun_phrase_label(child, include_det)

                if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                    # Check if this "NOUN" is actually the verb predicate (e.g., span mis-tagged)
                    if child.text.lower() not in {"wall", "miles"}:  # crude filter
                        child.pos_ = "VERB"
                        child.dep_ = "ROOT"
                        tok.dep_ = "aux"

                        # steal tok’s dependents (prep phrases) and attach to child
                        for dep in list(tok.children):
                            if dep.dep_ == "prep":
                                dep.head = child

                        break

        # Special case for "be + adjective + preposition" constructions
        if (len(doc) > 3 and
                doc[0].text.lower() == "is" and
                doc[0].pos_ == "AUX" and
                any(t.pos_ == "ADJ" and t.dep_ == "acomp" for t in doc)):

            # Get the adjective (e.g., "visible")
            adj = next(t for t in doc if t.pos_ == "ADJ" and t.dep_ == "acomp")

            # Get the subject (e.g., "the Great Wall")
            subject = next((t for t in doc if t.dep_ in SUBJ_DEPS), None)

            if subject and adj:
                subj_text = noun_phrase_label(subject, include_det)
                adj_text = adj.text

                # Handle prepositional phrases attached to the adjective
                for prep in [c for c in adj.children if c.dep_ == "prep"]:
                    pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                    if pobj:
                        loc_text = noun_phrase_label(pobj, include_det)
                        triples.add((subj_text, "property", f"{adj_text} {prep.text} {loc_text}"))
                        return triples

                # Fallback if no preposition found
                triples.add((subj_text, "property", adj_text))
                return triples

        # Case F (NEW): Handle AUX as ROOT for IsA questions like "Is X Y?"
        if tok.i == tok.head.i and tok.pos_ == "AUX" and tok.lemma_ == "be":
            subjects = [c for c in tok.children if c.dep_ in SUBJ_DEPS]
            attrs = [c for c in tok.children if c.dep_ in {"attr", "acomp"}]
            if subjects and attrs:
                subj = noun_phrase_label(subjects[0])
                pred = noun_phrase_label(attrs[0])
                triples.add((subj, "isa", pred))

        # Case A: Handle passive voice constructions first
        if tok.pos_ == "AUX" and is_passive_auxiliary(tok) and tok.i not in processed_aux:
            main_verb = find_main_verb_in_passive(tok)
            if main_verb:
                processed_aux.add(tok.i)
                
                v = verb_label(main_verb)
                if collect_neg(tok) or collect_neg(main_verb):
                    v = f"not {v}"

                # Get subjects from the auxiliary (passive subjects)
                subs = [c for c in tok.children if c.dep_ in SUBJ_DEPS]
                for s in subs:
                    subj = noun_phrase_label(s if s.pos_ in {"NOUN","PROPN"} else s.head, include_det)
                    triples.add((subj, "subj", v))

                # Handle prepositional phrases attached to main verb
                for prep in (c for c in main_verb.children if c.dep_ == "prep"):
                    for p in (c for c in prep.children if c.dep_ == "pobj"):
                        tail = noun_phrase_label(p, include_det) if p.pos_ in {"NOUN","PROPN"} else p.text
                        triples.add((v, f"prep_{prep.text.lower()}", tail))

        # Case B: Regular VERB predicates (non-passive)
        elif tok.pos_ == "VERB" and tok.i not in processed_aux:
            # Skip passive participles
            if any(aux.pos_ == "AUX" and aux.lemma_ == "be" and
                   find_main_verb_in_passive(aux) == tok for aux in doc):
                continue

            # Skip if this is a support-verb like "do/does/did"
            if tok.lemma_ == "do" and any(c.pos_ == "VERB" for c in tok.children):
                continue

            v = verb_label(tok)
            if collect_neg(tok):
                v = f"not {v}"

            subs = subjects_for(tok)
            for s in subs:
                subj = noun_phrase_label(s if s.pos_ in {"NOUN", "PROPN"} else s.head, include_det)
                triples.add((subj, "subj", v))

            # objects
            for o in (c for c in tok.children if c.dep_ in OBJ_DEPS):
                tail = noun_phrase_label(o, include_det) if o.pos_ in {"NOUN", "PROPN"} else o.text
                triples.add((v, "obj", tail))

            # preps
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                for p in (c for c in prep.children if c.dep_ == "pobj"):
                    tail = noun_phrase_label(p, include_det) if p.pos_ in {"NOUN", "PROPN"} else p.text
                    triples.add((v, f"prep_{prep.text.lower()}", tail))

        # Case C: Handle mis-tagged "NOUN" roots after auxiliaries
        elif tok.pos_ == "NOUN" and tok.dep_ == "ROOT":
            aux_before = [a for a in tok.lefts if a.pos_ == "AUX"]
            if aux_before:
                # Case 1: nominal predicate with copula ("X is a Y") → isa relation
                if any(a.lemma_ == "be" for a in aux_before):
                    subs = subjects_for(tok)
                    for s in subs:
                        subj = noun_phrase_label(s, include_det)
                        pred = noun_phrase_label(tok, include_det)
                        triples.add((subj, "isa", pred))

                # Case 2: aux + verb mis-tagged as NOUN ("Does ... stretch")
                else:
                    v = tok.text.lower()
                    if collect_neg(tok):
                        v = f"not {v}"
                    subs = subjects_for(tok)
                    for s in subs:
                        triples.add((noun_phrase_label(s, include_det), "subj", v))
                    for prep in (c for c in tok.children if c.dep_ == "prep"):
                        for p in (c for c in prep.children if c.dep_ == "pobj"):
                            tail = noun_phrase_label(p, include_det)
                            triples.add((v, f"prep_{prep.text.lower()}", tail))

        # Case D: Copular nominal predicates: "X is a Y" → (X, isa, Y)
        elif tok.pos_ == "NOUN" and has_copula(tok):
            subs = subjects_for(tok)
            for s in subs:
                subj = noun_phrase_label(s if s.pos_ in {"NOUN","PROPN"} else s.head, include_det)
                pred = noun_phrase_label(tok, include_det)
                triples.add((subj, "isa", pred))

        # Case E: Copular adjectival predicates: "X is located …"
        elif tok.pos_ == "ADJ" and has_copula(tok):
            v = tok.lemma_
            subs = subjects_for(tok)
            for s in subs:
                subj = noun_phrase_label(s if s.pos_ in {"NOUN","PROPN"} else s.head, include_det)
                triples.add((subj, "subj", v))
            # keep prepositional modifiers anchored to the adjective
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                for p in (c for c in prep.children if c.dep_ == "pobj"):
                    tail = noun_phrase_label(p, include_det) if p.pos_ in {"NOUN","PROPN"} else p.text
                    triples.add((v, f"prep_{prep.text.lower()}", tail))

    return triples

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



# ---------- ) Codebook ----------

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

# ---------- Message builders ----------
def make_codebook_message(codebook: dict) -> str:
    # Send once at session start (or when the codebook changes)
    return json_dump_str(codebook)

def make_edges_message(sid: str, edges: List[List[int]],use_full_edges:bool = True) -> str:
    # Send repeatedly; tiny payload (ids only)
    if use_full_edges:
        json_msg = json_dump_str({"sid": sid,'questions([[e,r,e], ...])':all_chains_no_subchains(edges,use_full_edges)})
    else:
        json_msg = json_dump_str({"sid": sid, "edges([e,r,e])": edges,'questions(edges[i])':all_chains_no_subchains(edges,use_full_edges)})

    return json_msg


def get_js_msgs_use_triples(question_prompt):
    triples = sentence_relations(question_prompt, include_det=False)
    codebook, ent2id, rel2id = build_codebook_from_triples(triples)
    msg1 = make_codebook_message(codebook)  # send once

    edges = edges_from_triples(triples, ent2id, rel2id)
    msg2 = make_edges_message(codebook["sid"], edges)  # send many times or once

    return msg1,msg2


def get_merged_message(question_prompt,use_full_edges = True):
    triples = sentence_relations(question_prompt, include_det=False)

    codebook, ent2id, rel2id = build_codebook_from_triples(triples)

    edges = edges_from_triples(triples, ent2id, rel2id)

    if use_full_edges:
        dict_2 =  {'questions([[e,r,e], ...])':all_chains_no_subchains(edges,use_full_edges)}
    else:
        dict_2 = {"edges([e,r,e])": edges,'questions(edges[i])':all_chains_no_subchains(edges,use_full_edges)}

    codebook.update(dict_2)

    codebook.pop('sid') 

    return codebook



#### add answers to code book



########### Aug 30-31,2025: merging code book method


class WordAvgEmbeddings(Embeddings):
    """
    A simple word-averaging embedding for LangChain.
    - 输入文本会被正则分词（只保留 a-zA-Z），并转小写
    - 每个词用 KeyedVectors 查词向量，取均值
    - OOV 时返回全零向量
    - 支持可选 L2 归一化，返回 Python list（FAISS 友好）
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        kv: Optional[KeyedVectors] = None,
        l2_normalize: bool = True,
        token_pattern: str = r"[A-Za-z]+"
    ):
        """
        Args:
            model_path: 本地 KeyedVectors 路径（.kv / .bin / word2vec 格式）。
                        例如：'gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model'
            kv:         也可以直接传入已加载的 KeyedVectors（与 model_path 二选一）
            l2_normalize: 是否对平均向量做 L2 归一化
            token_pattern: 分词正则
        """
        if kv is not None:
            self.kv = kv
        elif model_path:
            # 尝试用 KeyedVectors.load 加载；失败则回退到 word2vec 格式加载
            try:
                self.kv = KeyedVectors.load(model_path, mmap='r')
            except Exception:
                # 如果是 word2vec / text 格式
                self.kv = KeyedVectors.load_word2vec_format(model_path, binary=False)
        else:
            raise ValueError("Provide either `model_path` or `kv`.")

        self.dim = self.kv.vector_size
        self.l2_normalize = l2_normalize
        self.token_pat = re.compile(token_pattern)

    # ---- 内部：单条文本向量化 ----
    def _embed_text(self, text: str) -> np.ndarray:
        toks = [t.lower() for t in self.token_pat.findall(text)]
        vecs = [self.kv[w] for w in toks if w in self.kv]
        if not vecs:
            v = np.zeros(self.dim, dtype=np.float32)
        else:
            v = np.mean(vecs, axis=0).astype(np.float32)
        if self.l2_normalize:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    # ---- LangChain 抽象方法：多文档 ----
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain 期望返回 List[List[float]]
        out = [self._embed_text(t).tolist() for t in texts]
        return out

    # ---- LangChain 抽象方法：单查询 ----
    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text).tolist()
    
# change back to your own path

word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")


def get_word_embeddings(list_of_text,word_emb):
    """
    list_of_text: ['str1 str2 ...',]
    word_emb: embedding model

    list_of_text_embeddings:  [embedding_vals,...]
    """

    list_of_text_embeddings = [word_emb._embed_text(text) for text in list_of_text]


    return list_of_text_embeddings


### edit codebook to also take the answers
def get_code_book(prompt, type='questions', rule="Answer questions."):
    """
    prompt : str
    type   : one of {'questions','answers','thinkings','facts'}
    """
    valid_types = {'questions', 'answers', 'thinkings', 'facts'}
    if type not in valid_types:
        raise ValueError(f"type must be one of {valid_types}, got: {type}")

    triples = sentence_relations(prompt, include_det=False)

    codebook, ent2id, rel2id = build_codebook_from_triples(triples, rule)
    edges = edges_from_triples(triples, ent2id, rel2id)

    feat_name = f"{type}(edges[i])"  

    dict_2 = {
        "edges([e,r,e])": edges,
        feat_name: all_chains_no_subchains(edges, False)
    }

    codebook.update(dict_2)
    codebook.pop('sid')
    return codebook


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



#### making the merging codebook also able to merge the answer code book
def merging_codebook(codebook_main, codebook_sub, type='questions', word_emb=word_emb, use_thinkings=False):
    if type == 'fact':
        type = 'facts'

    feat_name = type + '(edges[i])'

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

    if codebook_main:
        codebook_main.setdefault('answers_lst', [])
        codebook_main.setdefault('thinkings_lst', [])
        codebook_main.setdefault('questions_lst', [])
        codebook_main.setdefault('facts_lst', [])
        codebook_main.setdefault('questions_to_thinkings', {})

        questions_needs_merged = codebook_sub[feat_name]
        lst_questions_main = codebook_main[main_feat_name]

        edge_mat_needs_merged = codebook_sub.get('edges([e,r,e])', codebook_sub.get('edge_matrix'))
        edge_mat_main = codebook_main['edge_matrix']

        new_index_replacement_for_ent_sub, index_ent_main, new_added_ents = update_the_index(codebook_main, codebook_sub, 'e')
        new_index_replacement_for_r_sub, index_r_main, new_added_rs = update_the_index(codebook_main, codebook_sub, 'r')

        new_ent_embeds = get_word_embeddings(new_added_ents, word_emb)
        new_r_embeds = get_word_embeddings(new_added_rs, word_emb)

        edge_mat_needs_merged_remapped = remap_edges_matrix(edge_mat_needs_merged, new_index_replacement_for_ent_sub, new_index_replacement_for_r_sub)

        new_index_replacement_for_edges_sub, index_edges_main = combine_updated_edges(edge_mat_main, edge_mat_needs_merged_remapped)

        updated_questions_sub = remap_question_indices(questions_needs_merged, new_index_replacement_for_edges_sub)

        lst_questions_main.append(updated_questions_sub)

        ### add the knowledge graph and it's related index
        codebook_main["e"].extend(new_added_ents)
        codebook_main["r"].extend(new_added_rs)
        codebook_main["edge_matrix"] = index_edges_main
        codebook_main[main_feat_name] = lst_questions_main
        codebook_main["e_embeddings"] = codebook_main['e_embeddings'] + new_ent_embeds
        codebook_main["r_embeddings"] = codebook_main['r_embeddings'] + new_r_embeds

        if type == 'thinkings':
            codebook_main['questions_to_thinkings'][len(codebook_main['questions_lst']) - 1] = len(codebook_main[main_feat_name]) - 1

    else:
        # main codebook is empty
        codebook_main = {
            "e": codebook_sub['e'],
            "r": codebook_sub['r'],
            'edge_matrix': codebook_sub.get('edges([e,r,e])', codebook_sub.get('edge_matrix')),  # ← 同样兜底
            main_feat_name: [codebook_sub[feat_name]],
            unupdated_feat_name1: [],
            "rule": codebook_sub['rule'],
            "e_embeddings": get_word_embeddings(codebook_sub['e'], word_emb),
            "r_embeddings": get_word_embeddings(codebook_sub['r'], word_emb),
        }

        if use_thinkings:
            codebook_main[unupdated_feat_name2] = []
            codebook_main['questions_to_thinkings'] = {}

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

    # --- entities ---
    if "e_embeddings" not in codebook_main or not codebook_main["e_embeddings"]:
        if "e" not in codebook_main:
            raise ValueError("codebook_main missing key 'e' to compute e_embeddings.")
        try:
            # try your word_emb pipeline
            codebook_main["e_embeddings"] = get_word_embeddings(codebook_main["e"], word_emb)
        except Exception:
            # fallback stable random
            codebook_main["e_embeddings"] = _hash_embed(codebook_main["e"], dim=dim_fallback)

    # --- relations ---
    if "r_embeddings" not in codebook_main or not codebook_main["r_embeddings"]:
        if "r" not in codebook_main:
            raise ValueError("codebook_main missing key 'r' to compute r_embeddings.")
        try:
            codebook_main["r_embeddings"] = get_word_embeddings(codebook_main["r"], word_emb)
        except Exception:
            codebook_main["r_embeddings"] = _hash_embed(codebook_main["r"], dim=dim_fallback)

def get_topk_word_embedding_batched(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    top_k: int = 3,
    question_batch_size: int = 1,
    questions_db_batch_size: int = 1,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Top-k similarity with **decode_question(..., fmt='embeddings')** in two-way batches.

    Now auto-build e/r embeddings if missing in codebook_main.
    """
    # 0) ensure embeddings are present (compute on-the-fly if absent)

    # 1) infer embedding dim
    if "e_embeddings" in codebook_main and len(codebook_main["e_embeddings"]) > 0:
        dim = len(codebook_main["e_embeddings"][0])
    elif "r_embeddings" in codebook_main and len(codebook_main["r_embeddings"]) > 0:
        dim = len(codebook_main["r_embeddings"][0])
    else:
        _ensure_embeddings_in_codebook(codebook_main, dim_fallback=64)

    # 2) get DB questions
    if "questions_lst" not in codebook_main:
        raise ValueError("codebook_main missing 'questions_lst' for retrieval DB.")
    
    # don't search the last one, the last one is being updated as the current question
    questions_lst = codebook_main["questions_lst"][:-1]
    db_questions: List[List[int]] = []
    db_qi: List[int] = []
    db_qj: List[int] = []
    for qi, group in enumerate(questions_lst):
        for qj, q_edges in enumerate(group):
            db_questions.append(q_edges)
            db_qi.append(qi)
            db_qj.append(qj)

    N_total = len(questions)
    M_total = len(db_questions)
    results: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(N_total)}
    if N_total == 0 or M_total == 0:
        return results

    db_qi = np.asarray(db_qi, dtype=np.int32)
    db_qj = np.asarray(db_qj, dtype=np.int32)

    # 3) process query batches
    for q_start in range(0, N_total, question_batch_size):
        q_end = min(q_start + question_batch_size, N_total)
        q_batch_idx = list(range(q_start, q_end))
        q_batch_lists = [questions[i] for i in q_batch_idx]

        # embed query batch
        q_mat = _embed_questions_with_decode(q_batch_lists, codebook_main, dim)  # (Qb, d)
        Qb = q_mat.shape[0]

        best_scores = [np.array([], dtype=np.float32) for _ in range(Qb)]
        best_cols   = [np.array([], dtype=np.int32)   for _ in range(Qb)]

        # 4) stream DB in batches
        for db_start in range(0, M_total, questions_db_batch_size):
            db_end = min(db_start + questions_db_batch_size, M_total)
            db_batch_lists = db_questions[db_start:db_end]

            db_mat = _embed_questions_with_decode(db_batch_lists, codebook_main, dim)  # (Db, d)
            Db = db_mat.shape[0]
            if Db == 0:
                continue

            sims = _cosine_sim(q_mat, db_mat)  # (Qb, Db)
            k_local = min(top_k, Db)

            for i in range(Qb):
                row = sims[i]
                cand_idx = np.argpartition(-row, k_local - 1)[:k_local]
                cand_idx = cand_idx[np.argsort(-row[cand_idx])]
                batch_scores = row[cand_idx]
                batch_cols   = cand_idx + db_start
                merged_scores, merged_cols = _topk_merge(
                    best_scores[i], best_cols[i], batch_scores, batch_cols, top_k
                )
                best_scores[i] = merged_scores
                best_cols[i]   = merged_cols

        # 5) collect results
        for local_i, global_q_idx in enumerate(q_batch_idx):
            cols = best_cols[local_i]
            scs  = best_scores[local_i]
            keep = (cols >= 0)
            cols, scs = cols[keep], scs[keep]
            entries = []
            for col, sc in zip(cols, scs):
                entries.append({
                    "score": float(sc),
                    "questions_index": int(db_qi[col]),
                    "question_index": int(db_qj[col]),
                })
            results[global_q_idx] = entries

    return results

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


def rerank_with_sentence_embeddings(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    coarse_topk: Dict[int, List[Dict[str, Any]]],
    emb: HuggingFaceEmbeddings,
    top_m: int = 1,
    custom_linearizer: Optional[Callable[[List[List[str]]], str]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    For each query i:
      - Build a small FAISS index over its coarse_topk candidates (converted to text)
      - Embed the query as text and retrieve best `top_m`
      - Return {"score": <faiss_score>, "questions_index": qi, "question_index": qj, "text": candidate_text}

    Notes:
      * FAISS scores from LangChain are distances (smaller is better).
      * `top_m` can be <= len(coarse_topk[i]); if bigger, it's clipped.
    """
    # Flatten DB pointers for convenience
    questions_lst = codebook_main["questions_lst"]

    results: Dict[int, List[Dict[str, Any]]] = {}
    for i, q_edges in enumerate(questions):
        cand = coarse_topk.get(i, [])
        if not cand:
            results[i] = []
            continue

        # Deduplicate (qi, qj) while preserving order
        seen = set()
        kept_meta = []
        kept_texts = []
        for item in cand:
            qi = int(item["questions_index"])
            qj = int(item["question_index"])
            key = (qi, qj)
            if key in seen:
                continue
            seen.add(key)
            db_edges = questions_lst[qi][qj]
            text = make_question_text(db_edges, codebook_main, custom_linearizer)
            kept_meta.append({"questions_index": qi, "question_index": qj, "text": text})
            kept_texts.append(text)

        if not kept_texts:
            results[i] = []
            continue

        # Build a per-query FAISS index over the candidate texts
        vs = FAISS.from_texts(kept_texts, embedding=emb, metadatas=kept_meta)

        # Query text (use same linearization as candidates)
        query_text = make_question_text(q_edges, codebook_main, custom_linearizer)

        # Retrieve top_m (clip to number of candidates)
        m = min(top_m, len(kept_texts))
        docs_scores = vs.similarity_search_with_score(query_text, k=m)

        # Convert to output format (note: score is distance; smaller = better)
        ranked = []
        for doc, score in docs_scores:
            md = doc.metadata
            ranked.append({
                "score": float(score),
                "questions_index": int(md["questions_index"]),
                "question_index": int(md["question_index"]),
                "text": md["text"],
            })
        results[i] = ranked

    return results


def coarse_filter(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    sentence_emb: HuggingFaceEmbeddings,        # ← move before defaults
    top_k: int = 3,                             # word-embedding candidates
    question_batch_size: int = 1,               # query batch size
    questions_db_batch_size: int = 1,           # DB batch size
    top_m: int = 1,                             # sentence-embedding rerank
    custom_linearizer: Optional[Callable[[List[List[str]]], str]] = None):

    # doing the word embedding pre-filter 

    coarse_top_k = get_topk_word_embedding_batched(
    questions,
    codebook_main,
    top_k,
    question_batch_size,         # number of query questions processed per time
    questions_db_batch_size,     # number of db questions processed per time
    )

    # doing the sentence embedding filter 

    top_m_results = rerank_with_sentence_embeddings(
    questions,
    codebook_main,
    coarse_top_k,
    sentence_emb,
    top_m,
    custom_linearizer)


    return top_m_results



def add_answers_to_filtered_lst(top_m_results,codebook_main):

    result = {}
    for qid, matches in top_m_results.items():
        result[qid] = []
        for m in matches:
            q_idx = m["questions_index"]
            m_with_feat = m.copy()
            m_with_feat['answers(edges[i])'] = codebook_main['answers_lst'][q_idx]
            result[qid].append(m_with_feat)

    return result

def get_answers_lst_from_results(result):
    # this will return the answers list and it's relative indicies
    all_q_indices = [d['questions_index'] for v in result.values() for d in v]
    all_answers = [d['answers(edges[i])'] for v in result.values() for d in v]

    return all_answers,all_q_indices



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

def common_contiguous_overlaps(answers_lst, min_len=2):
    """
    Return all maximal common contiguous subsequences (runs) that appear in EVERY list.
    Each run has length >= min_len.
    """
    if not answers_lst:
        return []

    # Candidate runs = all runs from the first list
    candidates = set(_all_contiguous_subseqs(answers_lst[0], min_len=min_len))

    # Intersect with runs from all the other lists
    for lst in answers_lst[1:]:
        runs_here = set(_all_contiguous_subseqs(lst, min_len=min_len))
        candidates &= runs_here
        if not candidates:
            return []

    # Keep only maximal runs (remove those that are subruns of longer runs)
    maximal = set(candidates)
    for a in list(candidates):
        for b in candidates:
            if a != b and _is_subrun(a, b):
                maximal.discard(a)
                break

    # Sort by length desc, then lexicographically for determinism
    return [list(t) for t in sorted(maximal, key=lambda t: (-len(t), t))]



def get_flat_answers_lsts(answers_lsts):    
    return [[x for group in bucket for x in (group if isinstance(group, (list, tuple)) else [group])] for bucket in answers_lsts]


def find_overlapped_answers(answers_lsts):
    flat_answers_lsts = get_flat_answers_lsts(answers_lsts)
    # default 2 for the overlap
    final_flat_answers_lsts = common_contiguous_overlaps(flat_answers_lsts,2)
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
        final_flat_thinkings_lsts = common_contiguous_overlaps(flat_thinkings_lsts,2)
    else:
        final_flat_thinkings_lsts = selected_thinkings_lsts

    return final_flat_thinkings_lsts

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

def get_unique_knowledge(overlapped_answers,flat_answers_lsts):
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
        overlapped_runs = common_contiguous_overlaps(flat_thinkings_lsts, 2)  # list
        unique_dict = get_unique_knowledge({'overlaps': overlapped_runs},     # ✅ wrap
                                           flat_thinkings_lsts)
        uniqie_thinkings = unique_dict['out_answers']
    else:
        uniqie_thinkings = selected_thinkings_lsts

    return uniqie_thinkings


# add find unique answers

def find_unique_answers(answers_lsts):
    flat_answers_lsts = get_flat_answers_lsts(answers_lsts)
    overlapped_runs = common_contiguous_overlaps(flat_answers_lsts, 2)   # list[list[int]]
    unique_dict = get_unique_knowledge({'overlaps': overlapped_runs},    # ✅ wrap
                                       flat_answers_lsts)
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
            'start thinking with(edges[i])':decode_questions(flat_thinkings_lsts,final_merged_json,'edges'),
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
            'start thinking with(edges[i])':decode_questions(flat_thinkings_lsts,final_merged_json,'edges'),
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

class CompressRag:
    def __init__(
        self,
        ini_meta_codebook = {},
        sentence_emb: Optional[Embeddings] = None,
        word_emb: Optional[Embeddings] = None,
        include_thinkings = True,
        llm = None,
    ):

        # meta
        # start with empty codebook
        self.meta_codebook = ini_meta_codebook
        self.include_thinkings = include_thinkings
        self.llm = llm

        # Embeddings
        self.sentence_emb = sentence_emb 
        self.word_emb = word_emb 

        #coarse filter params
        self.top_k = 10
        self.top_m = 2
        self.question_batch_size = 1
        self.questions_db_batch_size = 1
        self.custom_linearizer = None


        # combine ents
        self.min_exp_num =2
        self.max_exp_num = 10


    def encode_question(self,q_prompt,rule):

        return get_code_book(q_prompt,'questions',rule)
    
    def retrieve(self,q_json):
        # needs to be fixed

        # questions queries: list of edge indices

        # change question edges index to the edges in codebook main first

        # now the problem is that q_json's 'edges([e,r,e])' is the from q_json's (e, r, e) index

        # merge with meta code book first

        self.meta_codebook = merging_codebook(self.meta_codebook,q_json,'questions',self.word_emb,True)

        # take the last one 

        questions_edges_index = self.meta_codebook['questions_lst'][-1]

        top_m_results = coarse_filter(
                        questions_edges_index,
                        self.meta_codebook,
                        self.sentence_emb,                 # ← move before defaults
                        self.top_k,                             # word-embedding candidates
                        self.question_batch_size,               # query batch size
                        self.questions_db_batch_size,           # DB batch size
                        self.top_m,                             # sentence-embedding rerank
                        self.custom_linearizer)
        
        result = add_answers_to_filtered_lst(top_m_results,self.meta_codebook)

        all_answers,all_q_indices = get_answers_lst_from_results(result)

        return all_answers,all_q_indices
    

    def find_related_knowledge(self,all_answers,all_q_indices):

        domain_knowledge_lst = []

        # this will automatically flatten answers
        overlapped_answers = find_overlapped_answers(all_answers)

        flat_answers = get_flat_answers_lsts(all_answers)

        overlapped_answers_dict = {'overlaps': overlapped_answers}

        unique_knowledge_dict = get_unique_knowledge(overlapped_answers_dict,flat_answers)

        unique_knowledge = unique_knowledge_dict['out_answers']

        domain_knowledge_lst.append(unique_knowledge)

        if self.include_thinkings:
            final_flat_thinkings_lsts =find_overlapped_thinkings(all_q_indices,self.meta_codebook)
            domain_knowledge_lst.append(final_flat_thinkings_lsts)



        return domain_knowledge_lst
    

    def compact_indicies_for_prompt(self,codebook_sub_q,domain_knowledge_lst):

        if self.include_thinkings:
            flat_answers_lsts,flat_thinkings_lsts = domain_knowledge_lst
            final_merged_json= get_json_with_given_knowledge_and_thinkings(flat_answers_lsts,flat_thinkings_lsts,
                                                                           self.meta_codebook,codebook_sub_q)

        else:
            flat_answers_lsts = domain_knowledge_lst[0]
            final_merged_json = get_json_with_given_knowledge(flat_answers_lsts,self.meta_codebook,codebook_sub_q)


        return final_merged_json
    
    
    def collect_results(self, final_merged_json, question):
        llm = self.llm

        new_json_lst = []
        new_result = None

        if self.include_thinkings:
            a_new, t_new = llm.take_questions(final_merged_json, question)
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')
            t_new_json = get_code_book(t_new, type='thinkings')
            new_json_lst.extend([a_new_json, t_new_json])
        else:
            a_new = llm.take_questions(final_merged_json, question)
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')
            new_json_lst.append(a_new_json)
        return new_result,new_json_lst
    
    def update_meta(self,codebook_sub_q,new_json_lst):

        if self.include_thinkings:
            codebook_sub_a,codebook_sub_t = new_json_lst

            self.meta_codebook = merging_codebook(self.meta_codebook,codebook_sub_q,'questions',self.word_emb,True)
            self.meta_codebook = merging_codebook(self.meta_codebook,codebook_sub_a,'answers',self.word_emb,True)
            self.meta_codebook = merging_codebook(self.meta_codebook,codebook_sub_t,'thinkings',self.word_emb,True)

        else:
            codebook_sub_a = new_json_lst[0]
            self.meta_codebook = merging_codebook(self.meta_codebook,codebook_sub_q,'questions',self.word_emb,True)
            self.meta_codebook = merging_codebook(self.meta_codebook,codebook_sub_a,'answers',self.word_emb,True)


    def combine_ents_func(self):

        self.meta_codebook = combine_ents(self.meta_codebook,
                 self.min_exp_num,  
                 self.max_exp_num,  
                 self.include_thinkings) 
        

    def run_work_flow(self,q_prompt,rule = "Answer questions"):
        q_json = self.encode_question(q_prompt,rule)

        # check the meta code book is not empty

        if self.meta_codebook:
            all_answers,all_q_indices = self.retrieve(q_json)
            domain_knowledge_lst= self.find_related_knowledge(all_answers,all_q_indices)
            final_merged_json = self.compact_indicies_for_prompt(q_json,domain_knowledge_lst)

        else:
            final_merged_json = q_json.copy()


        new_result,new_json_lst = self.collect_results(final_merged_json, question=q_prompt)

        self.update_meta(q_json,new_json_lst)

        self.combine_ents_func()

        # return answer

        return new_result
    

### CompressRag RL version

# thinkings extraction choice: keep the overlap(default), not include the thinking, keep the unique thinking
# answers extraction choice: keep the unique (default), not include the answers, keep the overlap
# combine ents choice: not combine, combine per round, combine per 3 round
thinkings_choice = ['overlap','unique','not_include']
answers_choice = ['overlap','unique','not_include']
combine_ents_choice = [0,1,2]

class CompressRag_rl:
    def __init__(
        self,
        ini_meta_codebook = {},
        sentence_emb: Optional[Embeddings] = None,
        word_emb: Optional[Embeddings] = None,
        llm = None,
        combine_ents_rounds = 1, # params to control the combine ents
        thinkings_choice = 'not_include',
        answers_choice = 'overlap'
    ):
        """
        thinkings_choice and answers_choice must be one of 'overlap','unique','not_include'
        combine_ents_rounds must be interger-> how many rounds after combine ents

        
        """

        # meta
        # start with empty codebook
        self.meta_codebook = ini_meta_codebook
        self.llm = llm

        # Embeddings
        self.sentence_emb = sentence_emb 
        self.word_emb = word_emb 

        #coarse filter params
        self.top_k = 10
        self.top_m = 2
        self.question_batch_size = 1
        self.questions_db_batch_size = 1
        self.custom_linearizer = None


        # combine ents
        self.min_exp_num =2
        self.max_exp_num = 10


        # params for dpo
        ### ents param
        self.combine_ents_rounds = combine_ents_rounds
        self.round = 1

        ### thinkings param
        if thinkings_choice == "not_include":
            self.include_thinkings = False
        else:
            self.include_thinkings = True
            if thinkings_choice == "overlap":
                self.thinking_extract_function = find_overlapped_thinkings
            elif thinkings_choice == "unique":
                self.thinking_extract_function = find_unique_thinkings

        self.llm.include_thinkings = self.include_thinkings
        ### answers param
        if answers_choice == "not_include":
            self.include_answers = False
        else:
            self.include_answers = True
            if answers_choice == "overlap":
                self.answers_extract_function = find_overlapped_answers
            elif answers_choice == "unique":
                self.answers_extract_function = find_unique_answers

        ### context fact param
        self.context_json_path = None  
        self._facts_preloaded = False  


    def preload_context_json(self, json_path: str, chunk_chars: int = 800, overlap: int = 120):
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

        combined = None  

        for item in items:
            ctx = (item.get("context") or "").strip()
            if not ctx:
                continue
            for ch in _chunk_text(ctx, chunk_chars=chunk_chars, overlap=overlap):
                fact_cb = get_code_book(ch, type='facts', rule="Store factual statements.")
                if combined is None:
                    combined = {
                        "e": list(fact_cb["e"]),
                        "r": list(fact_cb["r"]),
                        "edge_matrix": list(fact_cb["edges([e,r,e])"]),
                        "facts(edges[i])": [lst for lst in fact_cb["facts(edges[i])"]],
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

        return combined


    def encode_question(self,q_prompt,rule):

        return get_code_book(q_prompt,'questions',rule)

    def _embed_edge_run(self, edge_run, codebook_main):
        decoded = decode_question(edge_run, codebook_main, fmt='embeddings')
        # 利用你已有的 _avg_vec_from_decoded
        dim = len(codebook_main["e_embeddings"][0]) if codebook_main.get("e_embeddings") else 64
        return _avg_vec_from_decoded(decoded, dim)

    def _rank_facts_for_query(self, query_edges, facts_runs, codebook_main, top_m=2):
        if not facts_runs:
            return []
        qv = self._embed_edge_run(query_edges, codebook_main).reshape(1, -1)

        F = np.stack([self._embed_edge_run(run, codebook_main) for run in facts_runs], axis=0)

        qn = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
        fn = F  / (np.linalg.norm(F,  axis=1, keepdims=True)  + 1e-12)
        sims = (qn @ fn.T).ravel()   
        k = min(top_m, sims.shape[0])
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]  

    def _flatten_facts(self, meta):
        """
        返回:
        flat_facts: List[List[int]]   # 每个元素是一条 fact 的边索引链
        map_idx:    List[Tuple[int,int]]  # (facts_lst 的组号, 该组内索引)
        """
        flat, map_idx = [], []
        for gi, group in enumerate(meta.get('facts_lst', [])):
            for fj, run in enumerate(group):
                # run 可能还有一层；这里做个稳健拍平：只要是 [int,...] 就当作一条
                if run and isinstance(run, (list, tuple)) and isinstance(run[0], int):
                    flat.append(run)
                    map_idx.append((gi, fj))
                elif isinstance(run, (list, tuple)):
                    for r2 in run:
                        if r2 and isinstance(r2, (list, tuple)) and isinstance(r2[0], int):
                            flat.append(r2)
                            map_idx.append((gi, fj))
        return flat, map_idx

    def retrieve(self,q_json):

        self.meta_codebook = merging_codebook(self.meta_codebook,q_json,'questions',self.word_emb,True)

        # take the last one 

        questions_edges_index = self.meta_codebook['questions_lst'][-1]

        top_m_results = coarse_filter(
                        questions_edges_index,
                        self.meta_codebook,
                        self.sentence_emb,                 # ← move before defaults
                        self.top_k,                             # word-embedding candidates
                        self.question_batch_size,               # query batch size
                        self.questions_db_batch_size,           # DB batch size
                        self.top_m,                             # sentence-embedding rerank
                        self.custom_linearizer)
        
        result = add_answers_to_filtered_lst(top_m_results,self.meta_codebook)

        all_answers,all_q_indices = get_answers_lst_from_results(result)
        

        flat_facts, facts_map = self._flatten_facts(self.meta_codebook)  

        all_facts = []
        all_f_indices = []   

        for q_edges in questions_edges_index:
            ranked = self._rank_facts_for_query(q_edges, flat_facts, self.meta_codebook, top_m=self.top_m)
            for fact_idx, _score in ranked:
                all_facts.append(flat_facts[fact_idx])
                all_f_indices.append(facts_map[fact_idx])

        return all_answers, all_q_indices, all_f_indices
    
    def _gather_facts_by_indices(self, all_f_indices, codebook_main):
        facts_lsts = []
        facts_store = codebook_main.get('facts_lst', [])
        for fi, fj in all_f_indices or []:
            try:
                facts_lsts.append(facts_store[int(fi)][int(fj)])
            except Exception:
                # 越界或结构不符时跳过
                pass
        return facts_lsts

    def find_related_knowledge(self,all_answers,all_q_indices, all_f_indices):

        domain_knowledge_lst = []

        if self.include_answers:
            final_flat_answers_lsts = self.answers_extract_function(all_answers)
            domain_knowledge_lst.append(final_flat_answers_lsts)


        if self.include_thinkings:
            final_flat_thinkings_lsts = self.thinking_extract_function(all_q_indices,self.meta_codebook)
            domain_knowledge_lst.append(final_flat_thinkings_lsts)
        
        if all_f_indices:
            facts_lsts = self._gather_facts_by_indices(all_f_indices, self.meta_codebook)

            final_flat_facts_lsts = self.answers_extract_function(facts_lsts) \
                                    if self.include_answers else get_flat_answers_lsts(facts_lsts)
            domain_knowledge_lst.append(final_flat_facts_lsts)   

        return domain_knowledge_lst
    

    def compact_indicies_for_prompt(self, codebook_sub_q, domain_knowledge_lst):
        # unpack
        flat_answers_lsts = None
        flat_thinkings_lsts = None
        flat_facts_lsts = None

        ptr = 0
        if self.include_answers:
            flat_answers_lsts = domain_knowledge_lst[ptr]
            ptr += 1
        if self.include_thinkings:
            flat_thinkings_lsts = domain_knowledge_lst[ptr]
            ptr += 1
        if getattr(self, "include_facts", False):  
            flat_facts_lsts = domain_knowledge_lst[ptr]
            ptr += 1

        # cases
        if self.include_answers and self.include_thinkings:
            final_merged_json = get_json_with_given_knowledge_and_thinkings(
                flat_answers_lsts, flat_thinkings_lsts,
                self.meta_codebook, codebook_sub_q
            )

        elif self.include_answers:
            final_merged_json = get_json_with_given_knowledge(
                flat_answers_lsts, self.meta_codebook, codebook_sub_q
            )

        elif self.include_thinkings:
            final_merged_json = get_json_with_given_thinkings(
                flat_thinkings_lsts, self.meta_codebook, codebook_sub_q
            )

        else:
            final_merged_json = codebook_sub_q.copy()

        if flat_facts_lsts is not None:
            final_merged_json["given facts(edges[i])"] = flat_facts_lsts
            final_merged_json["given facts([[e,r,e], ...])"] = decode_questions(
                flat_facts_lsts, final_merged_json, 'edges'
            )

        return final_merged_json

    # might also change these functions,now keep always merge with answers json, and only merge with thinking json if use thinkings
    
    def collect_results(self, final_merged_json, questions):
        llm = self.llm

        new_json_lst = []
        new_result = None

        if self.include_thinkings:
            a_new, t_new = llm.take_questions(final_merged_json, questions)
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')
            t_new_json = get_code_book(t_new, type='thinkings')
            new_json_lst.extend([a_new_json, t_new_json])
        else:
            a_new = llm.take_questions(final_merged_json, questions)
            new_result = a_new
            a_new_json = get_code_book(a_new, type='answers')
            new_json_lst.append(a_new_json)
        return new_result,new_json_lst
    
    def update_meta(self, codebook_sub_q, new_json_lst, facts_cb=None):
        if self.include_thinkings:
            codebook_sub_a, codebook_sub_t = new_json_lst
            self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_q, 'questions', self.word_emb, True)
            self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_a, 'answers',   self.word_emb, True)
            self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_t, 'thinkings', self.word_emb, True)
        else:
            codebook_sub_a = new_json_lst[0]
            self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_q, 'questions', self.word_emb, True)
            self.meta_codebook = merging_codebook(self.meta_codebook, codebook_sub_a, 'answers',   self.word_emb, True)

        if facts_cb:
            print("----------fact is loaded------")
            self.meta_codebook = merging_codebook(self.meta_codebook, facts_cb, 'facts', self.word_emb, False)


    def combine_ents_func(self):

        self.meta_codebook = combine_ents_auto(self.meta_codebook,
                 self.min_exp_num,  
                 self.max_exp_num,  
                 self.include_thinkings) 
        

    def run_work_flow(self, q_prompt, rule="Answer questions", facts_json_path: str = None, chunk_chars: int = 800, overlap: int = 120):
        q_json = self.encode_question(q_prompt, rule)
  
        temp_facts_cb = None
        if not self.meta_codebook and facts_json_path:
            temp_facts_cb = self.preload_context_json(facts_json_path, chunk_chars, overlap)

        if self.meta_codebook:
            all_answers, all_q_indices, all_f_indices = self.retrieve(q_json)
            print("all_f_indices", all_f_indices)
            domain_knowledge_lst = self.find_related_knowledge(all_answers, all_q_indices, all_f_indices)
            print("domain_knowledge_lst", domain_knowledge_lst)
            final_merged_json = self.compact_indicies_for_prompt(q_json, domain_knowledge_lst)
            print("final_merged_json", final_merged_json)
        else:
            if temp_facts_cb:
                final_merged_json = temp_facts_cb
            else:
                final_merged_json = q_json.copy()

        new_result, new_json_lst = self.collect_results(final_merged_json, questions=q_prompt)
        print("new_result", new_result)
        print("new_json_lst", new_json_lst)

        self.update_meta(q_json, new_json_lst, facts_cb=temp_facts_cb)

        if self.round % self.combine_ents_rounds == 0:
            self.combine_ents_func()
        self.round += 1

        return new_result


    
#### see usage on test_for_compressrag.py


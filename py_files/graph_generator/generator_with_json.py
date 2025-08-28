## Version 2 - Fixed for Passive Voice
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re
import json, hashlib
from typing import List, Tuple, Dict, Optional,Iterable
import itertools
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJ_DEPS  = {"dobj", "obj", "attr", "oprd", "dative"}
NEG_DEPS  = {"neg"}

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

# ---------- Deltas for new entities/relations (no aliasing) ----------
def compute_deltas_for_new_triples(
    new_triples: List[Tuple[str, str, str]],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
):
    add_e: List[str] = []
    add_r: List[str] = []
    for h, r, t in new_triples:
        if h not in ent2id:
            ent2id[h] = len(ent2id); add_e.append(h)
        if r not in rel2id:
            rel2id[r] = len(rel2id); add_r.append(r)
        if t not in ent2id:
            ent2id[t] = len(ent2id); add_e.append(t)
    delta = {}
    if add_e: delta["add_e"] = add_e
    if add_r: delta["add_r"] = add_r
    return delta

def make_delta_message(sid: str, delta: dict) -> Optional[str]:
    if not delta: return None
    out = {"sid": sid}
    out.update(delta)
    return json_dump_str(out)


def get_js_msgs_use_triples(question_prompt):
    triples = sentence_relations(question_prompt, include_det=False)
    codebook, ent2id, rel2id = build_codebook_from_triples(triples)
    msg1 = make_codebook_message(codebook)  # send once

    edges = edges_from_triples(triples, ent2id, rel2id)
    msg2 = make_edges_message(codebook["sid"], edges)  # send many times or once

    return msg1,msg2


# python py_files/graph_generator/generator_with_json.py
# check usage in generator_with_rules_v3.py
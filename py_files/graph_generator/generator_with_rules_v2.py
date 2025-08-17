## Version 2 - Fixed for Passive Voice
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re

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
    triples = []

    # Track processed auxiliaries to avoid double-processing
    processed_aux = set()

    for i, tok in enumerate(doc):
        print(tok, " - ", tok.pos_, " - ", tok.dep_)
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
                    triples.append((subj, "subj", v))

                # Handle prepositional phrases attached to main verb
                for prep in (c for c in main_verb.children if c.dep_ == "prep"):
                    for p in (c for c in prep.children if c.dep_ == "pobj"):
                        tail = noun_phrase_label(p, include_det) if p.pos_ in {"NOUN","PROPN"} else p.text
                        triples.append((v, f"prep_{prep.text.lower()}", tail))

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
                triples.append((subj, "subj", v))

            # objects
            for o in (c for c in tok.children if c.dep_ in OBJ_DEPS):
                tail = noun_phrase_label(o, include_det) if o.pos_ in {"NOUN", "PROPN"} else o.text
                triples.append((v, "obj", tail))

            # preps
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                for p in (c for c in prep.children if c.dep_ == "pobj"):
                    tail = noun_phrase_label(p, include_det) if p.pos_ in {"NOUN", "PROPN"} else p.text
                    triples.append((v, f"prep_{prep.text.lower()}", tail))

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
                        triples.append((subj, "isa", pred))

                # Case 2: aux + verb mis-tagged as NOUN ("Does ... stretch")
                else:
                    v = tok.text.lower()
                    if collect_neg(tok):
                        v = f"not {v}"
                    subs = subjects_for(tok)
                    for s in subs:
                        triples.append((noun_phrase_label(s, include_det), "subj", v))
                    for prep in (c for c in tok.children if c.dep_ == "prep"):
                        for p in (c for c in prep.children if c.dep_ == "pobj"):
                            tail = noun_phrase_label(p, include_det)
                            triples.append((v, f"prep_{prep.text.lower()}", tail))

        # Case D: Copular nominal predicates: "X is a Y" → (X, isa, Y)
        elif tok.pos_ == "NOUN" and has_copula(tok):
            subs = subjects_for(tok)
            for s in subs:
                subj = noun_phrase_label(s if s.pos_ in {"NOUN","PROPN"} else s.head, include_det)
                pred = noun_phrase_label(tok, include_det)
                triples.append((subj, "isa", pred))

        # Case E: Copular adjectival predicates: "X is located …"
        elif tok.pos_ == "ADJ" and has_copula(tok):
            v = tok.lemma_
            subs = subjects_for(tok)
            for s in subs:
                subj = noun_phrase_label(s if s.pos_ in {"NOUN","PROPN"} else s.head, include_det)
                triples.append((subj, "subj", v))
            # keep prepositional modifiers anchored to the adjective
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                for p in (c for c in prep.children if c.dep_ == "pobj"):
                    tail = noun_phrase_label(p, include_det) if p.pos_ in {"NOUN","PROPN"} else p.text
                    triples.append((v, f"prep_{prep.text.lower()}", tail))

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

# ---------------- tests ----------------
if __name__ == "__main__":
    questions = [
        "Is the Great Wall of China located in China?",
        "Does the Great Wall span over 13000 miles?",
        "Was the Great Wall built during the Ming Dynasty?",
        "Can the Great Wall be seen from space?",
        "Is the Great Wall made of stone and brick?",
        "Does the Great Wall have watchtowers?",
        "Was the Great Wall constructed over 2000 years?",
        "Is the Great Wall an UNESCO World Heritage Site?",
        "Does the Great Wall stretch across the northern China?",
        "Are millions of tourists visiting the Great Wall annually?"
    ]

    for s in questions:
        triples = sentence_relations(s, include_det=False)
        print(s, "->")
        for t in triples:
            print("   ", t)
        G = build_graph(triples)
        plot_graph(G, s)

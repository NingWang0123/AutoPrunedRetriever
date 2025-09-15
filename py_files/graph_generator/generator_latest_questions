## Version 2 - Fixed for Passive Voice
import string

import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re
import json, hashlib
from typing import List, Tuple, Dict, Optional, Iterable
import itertools
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJ_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}
NEG_DEPS = {"neg"}


# -------- node labels --------
def noun_phrase_label(head, include_det=False, use_ents=True):
    # 1) prefer named entities (incl. FAC)
    if use_ents:
        for ent in head.doc.ents:
            if ent.start <= head.i < ent.end and ent.label_ in {
                "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC"
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
    prt = [c.text for c in tok.children if c.dep_ == "prt"]
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
    """
    Enhanced robust subject finder that handles statements, questions, and edge cases.

    Args:
        pred: The predicate token (verb, adjective, or noun) to find subjects for

    Returns:
        List of tokens that are subjects of the predicate
    """

    # 1) Direct dependency - most common case
    subs = [c for c in pred.children if c.dep_ in SUBJ_DEPS]
    if subs:
        return subs

    # 2) Borrow from coordinated predicate (handles "X runs and jumps")
    if pred.dep_ == "conj" and pred.head.pos_ in {"VERB", "ADJ", "NOUN"}:
        sh = [c for c in pred.head.children if c.dep_ in SUBJ_DEPS]
        if sh:
            return sh

    # 3) For passive constructions, check if there's an auxiliary with the subject
    if pred.pos_ == "VERB":
        for tok in pred.doc:
            if (tok.pos_ == "AUX" and tok.lemma_ == "be" and
                    any(c.dep_ in SUBJ_DEPS for c in tok.children)):
                return [c for c in tok.children if c.dep_ in SUBJ_DEPS]

    # 4) Question-specific: aux-fronted questions like "Does X run?"
    # Look for noun_chunks between last AUX and predicate
    aux_before = [t for t in pred.doc if t.i < pred.i and t.pos_ == "AUX"]
    if aux_before:
        left_idx = max(a.i for a in aux_before)
        chunks = [nc for nc in pred.doc.noun_chunks if left_idx < nc.end <= pred.i]
        if chunks:
            # Get the rightmost chunk (closest to predicate)
            rightmost_chunk = sorted(chunks, key=lambda nc: nc.end)[-1]
            return [rightmost_chunk.root]

    # 5) WH-question handling: "What does X do?" - X is between WH and verb
    wh_words = {"what", "who", "where", "when", "why", "how", "which", "whose"}
    wh_tokens = [t for t in pred.doc if t.text.lower() in wh_words]

    if wh_tokens and pred.pos_ == "VERB":
        wh_token = wh_tokens[0]  # Take first WH-word

        # Look for subjects between WH-word and predicate
        between_chunks = [nc for nc in pred.doc.noun_chunks
                          if wh_token.i < nc.start and nc.end <= pred.i]
        if between_chunks:
            return [between_chunks[-1].root]  # Rightmost chunk

        # Look for individual nouns/pronouns between WH and predicate
        between_nouns = [t for t in pred.doc
                         if (wh_token.i < t.i < pred.i and
                             t.pos_ in {"NOUN", "PROPN", "PRON"} and
                             t.text.lower() not in wh_words)]
        if between_nouns:
            return [between_nouns[-1]]  # Rightmost noun

    # 6) Copular constructions: "X is Y" where Y is the predicate
    if pred.pos_ in {"NOUN", "ADJ"} and any(c.dep_ == "cop" for c in pred.children):
        # Look for subject of the copula
        cop_token = next(c for c in pred.children if c.dep_ == "cop")
        if cop_token:
            cop_subjects = [c for c in cop_token.children if c.dep_ in SUBJ_DEPS]
            if cop_subjects:
                return cop_subjects

            # Sometimes subject attaches to the predicate noun/adj instead
            pred_subjects = [c for c in pred.children if c.dep_ in SUBJ_DEPS]
            if pred_subjects:
                return pred_subjects

    # 7) Imperative detection: if no subject found and verb is at start
    if (pred.pos_ == "VERB" and pred.i <= 2 and  # At beginning of sentence
            pred.tag_ in {"VB", "VBP"} and  # Base form or present tense
            not any(t.pos_ == "AUX" for t in pred.doc[:pred.i])):  # No auxiliary before
        # This might be an imperative - implied "you" subject
        # But only return this if we haven't found anything else
        pass  # We'll fall through to other heuristics first

    # 8) Relative clause handling: "The man who runs"
    if pred.dep_ == "relcl":  # Relative clause
        # Subject is typically the head of the noun phrase this modifies
        if pred.head.pos_ in {"NOUN", "PROPN"}:
            return [pred.head]

    # 9) General fallback: rightmost noun_chunk before predicate
    chunks = [nc for nc in pred.doc.noun_chunks if nc.end <= pred.i]
    if chunks:
        rightmost_chunk = sorted(chunks, key=lambda nc: nc.end)[-1]
        return [rightmost_chunk.root]

    # 10) Token-level fallback: rightmost noun/pronoun before predicate
    cands = [t for t in pred.doc if (t.i < pred.i and
                                     t.pos_ in {"NOUN", "PROPN", "PRON"} and
                                     t.text.lower() not in wh_words)]
    if cands:
        return [cands[-1]]

    # 11) For questions starting with auxiliary, look after the auxiliary
    if pred.pos_ == "VERB" and pred.doc[0].pos_ == "AUX":
        aux_token = pred.doc[0]
        # Find noun chunks after the auxiliary but before the predicate
        after_aux_chunks = [nc for nc in pred.doc.noun_chunks
                            if aux_token.i < nc.start and nc.end <= pred.i]
        if after_aux_chunks:
            return [after_aux_chunks[0].root]  # First chunk after auxiliary

    # 12) Existential constructions: "There is X" - X is the logical subject
    if any(t.text.lower() == "there" and t.dep_ == "expl" for t in pred.doc):
        # Look for the logical subject (usually has "nsubj" or "attr" dependency)
        logical_subjects = [c for c in pred.children if c.dep_ in {"nsubj", "attr"}
                            and c.text.lower() != "there"]
        if logical_subjects:
            return logical_subjects

    # 13) Last resort: for imperatives, return implied "you"
    if (pred.pos_ == "VERB" and pred.i <= 2 and
            pred.tag_ in {"VB", "VBP"} and
            not any(c.dep_ in SUBJ_DEPS for c in pred.children)):
        # Create a dummy token for "you" - this is a bit hacky but works
        # In practice, you might want to return None and handle this in the caller
        return []  # Return empty list - caller should handle imperative case

    return []


# ==================== Helper Functions ====================
# These helpers are included to make the script self-contained.

def is_passive_auxiliary(tok):
    """Checks if a token is a passive auxiliary verb."""
    return tok.pos_ == "AUX" and tok.lemma_ == "be"


def find_main_verb_in_passive(aux_tok):
    """Finds the main verb governed by a passive auxiliary."""
    for child in aux_tok.children:
        # A passive verb will have the 'ROOT' or 'advcl' dependency
        if child.dep_ in ("ROOT", "advcl") and child.pos_ == "VERB":
            return child
    return None


def collect_neg(tok):
    """Checks if a token has a negation child (e.g., 'not')."""
    return any(c.dep_ in NEG_DEPS for c in tok.children)


def subjects_for(tok):
    """
    Finds subjects for a given verb or noun.
    Handles subjects that are children of the verb or its auxiliary.
    """
    if tok.head.dep_ in SUBJ_DEPS:
        return [tok.head]

    subjects = [c for c in tok.children if c.dep_ in SUBJ_DEPS]
    # In passive constructions, the subject might be attached to the auxiliary
    for aux in tok.ancestors:
        subjects.extend([c for c in aux.children if c.dep_ in SUBJ_DEPS])

    # Handle copular nominal predicates with subject attached to root
    if not subjects and tok.dep_ == "ROOT" and tok.pos_ in {"NOUN", "PROPN"}:
        subjects.extend([c for c in tok.children if c.dep_ in SUBJ_DEPS])

    return subjects


def noun_phrase_label(tok, include_det=False):
    """Constructs a noun phrase from a token, including its modifiers."""
    # Using `.subtree` is a robust way to get the entire phrase
    phrase_tokens = [
        t for t in tok.subtree if include_det or t.dep_ != "det"
    ]
    return " ".join(t.text for t in phrase_tokens)


def verb_label(tok):
    """Constructs a verb phrase, including any auxiliaries."""
    aux_parts = [c.text for c in tok.lefts if c.pos_ == "AUX"]
    verb_parts = aux_parts + [tok.text]
    return " ".join(verb_parts)


def has_copula(tok):
    """Checks if a token has a copular verb (`be`) as its head."""
    return tok.head.lemma_ == "be"

import re

def _clean_phrase(s: str) -> str:
    if not s:
        return ""
    # strip leading/trailing whitespace + punctuation
    s = s.strip().strip(string.punctuation)
    # also remove double spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def find_best_np_in_subtree(node) -> str:
    """
    Given a token (e.g. a prep token), return the best NP string found
    inside node.subtree (preferring named entities, then pobj/dobj, then first NOUN/PROPN).
    Always returns string (possibly empty).
    """
    doc = node.doc

    # 1) named entity fully inside this subtree
    for ent in doc.ents:
        if ent.start >= node.left_edge.i and ent.end - 1 <= node.right_edge.i:
            return ent.text

    # 2) find explicit pobj/dobj/attr/nsubj inside subtree
    for t in node.subtree:
        if t.dep_ in {"pobj", "dobj", "attr", "nsubj", "nsubjpass"} and t.pos_ in {"NOUN", "PROPN", "PRON"}:
            return " ".join([w.text for w in t.subtree])

    # 3) handle verb complement -> find its object inside subtree (e.g., "developing BCC")
    # look for a verb child inside subtree which itself has a dobj
    for t in node.subtree:
        if t.pos_ == "VERB":
            for c in t.children:
                if c.dep_ in {"dobj", "obj"} and c.pos_ in {"NOUN", "PROPN"}:
                    return " ".join([w.text for w in c.subtree])
            # also accept xcomp/pcomp objects
            for c in t.children:
                if c.dep_ in {"xcomp","pcomp"}:
                    # try to find dobj under that child
                    for d in c.subtree:
                        if d.dep_ in {"dobj","pobj"} and d.pos_ in {"NOUN","PROPN"}:
                            return " ".join([w.text for w in d.subtree])

    # 4) fallback: first NOUN/PROPN descendant (return its subtree)
    for t in node.subtree:
        if t.pos_ in {"NOUN", "PROPN"}:
            return " ".join([w.text for w in t.subtree])

    return ""

def expand_np_with_modifiers(token, include_det: bool = False) -> str:
    """
    Robust NP expansion:
      - prefer named entity spans,
      - include left-side compounds/amod/nummod/poss (and det if include_det=True),
      - attach "prep X" where X is the best NP found in the prep subtree (handles 'of developing BCC'),
      - include simple acl/relative clause modifiers.
    Always returns string (never None). Caller should skip empty strings.
    """
    if token is None:
        return ""

    doc = token.doc

    # 1) if token is inside a named entity, return that span
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return _clean_phrase(ent.text)

    # 2) collect left modifiers (amod/compound/nummod/poss; optionally det)
    left_parts = []
    for child in sorted(token.lefts, key=lambda t: t.i):
        if child.dep_ in {"amod", "compound", "nummod", "poss"} or (include_det and child.dep_ == "det"):
            left_parts.append(expand_np_with_modifiers(child, include_det))

    # include token text itself
    parts = [p for p in left_parts if p] + [token.text]

    phrase = " ".join(parts)

    # 3) attach prepositional phrases & relative clauses found on the right side
    right_addons = []
    for child in sorted(token.rights, key=lambda t: t.i):
        # common preposition child
        if child.dep_ == "prep":
            pobj_text = find_best_np_in_subtree(child)
            if pobj_text:
                right_addons.append(f"{child.text} {pobj_text}")
            else:
                # sometimes there is no pobj but the prep subtree contains useful nouns
                fallback = find_best_np_in_subtree(child)
                if fallback:
                    right_addons.append(f"{child.text} {fallback}")

        # relative clause / clause modifiers: include text of the entire subtree
        elif child.dep_ in {"acl", "relcl", "advcl"}:
            sub_phrase = " ".join([w.text for w in child.subtree]).strip()
            if sub_phrase:
                right_addons.append(sub_phrase)

    if right_addons:
        phrase = phrase + " " + " ".join(right_addons)

    phrase = _clean_phrase(phrase)
    return phrase



# ==================== Extraction Function (Refined for Questions) ====================
def sentence_relations(sentence, include_det=False):
    """
    Extract triples from QUESTIONS (ending with '?').

    Handles:
      CASE 1: "What is X of Y?" or "What is the primary risk factor for Z?"
      CASE 2: Passive voice questions ("Which locations are affected by X?",
                                       "Was the Wall constructed ...?")
      CASE 3: Active voice ("How does X VERB Y?")
      CASE 4: Source questions ("From which cell type does X arise?")
      CASE 5: Yes/No copular ("Is X Y?" / "Was X Y?")
      CASE 6: Modal + action ("Can X VERB Y?")
    """
    doc = nlp(sentence.strip())
    triples = set()

    # Debug
    for tok in doc:
        print(tok, "-", tok.pos_, "-", tok.dep_)

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if not root:
        return triples

    # -------------------
    # CASE 1: "What is X of Y?" / "What is the risk factor for Z?"
    if root.lemma_ == "be" and any(c.text.lower() == "what" for c in doc):
        subj = next((c for c in root.children if c.dep_ in {"attr", "nsubj"}), None)
        if subj.text.lower() == "what":
            # Try to find a better noun among siblings
            candidates = [c for c in root.children if c.dep_ in {"attr", "nsubj"} and c != subj]
            if candidates:
                subj = candidates[0]  # Prefer the noun like "role", "frequency", "symptom"
        if subj:
            subj_text = expand_np_with_modifiers(subj, include_det)

            # Look for attached prepositional modifiers
            preps = [c for c in subj.children if c.dep_ == "prep"]
            if preps:
                for prep in preps:
                    pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                    if pobj:
                        pobj_text = expand_np_with_modifiers(pobj, include_det)
                        triples.add((subj_text, prep.text, pobj_text))
            else:
                # If no prep: make the subject the answer slot instead of "What"
                triples.add((subj_text, "isa", "?"))

        return triples

    # -------------------
    # CASE 2: Passive voice questions
    # "Which locations are affected by X?" / "Was the Wall constructed ...?"
    # -------------------
    if root.pos_ == "VERB" and any(c.dep_ == "auxpass" for c in root.children):
        subs = [c for c in root.children if c.dep_ == "nsubjpass"]
        for s in subs:
            subj_text = expand_np_with_modifiers(s, include_det)

            # agent phrase "by ..."
            for prep in (c for c in root.children if c.dep_ == "agent"):
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj:
                    agent_text = expand_np_with_modifiers(pobj, include_det)
                    triples.add((subj_text, f"{root.lemma_}_by", agent_text))

            # prepositional modifiers
            for prep in (c for c in root.children if c.dep_ == "prep"):
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj:
                    pobj_text = expand_np_with_modifiers(pobj, include_det)
                    triples.add((subj_text, f"{root.lemma_}_{prep.text}", pobj_text))

            triples.add((subj_text, "subj", root.lemma_))
        return triples

    # -------------------
    # CASE 3: Active voice "How does X VERB Y?"
    # -------------------
    if root.pos_ == "VERB":
        subs = [c for c in root.children if c.dep_ == "nsubj"]
        objs = [c for c in root.children if c.dep_ in {"dobj", "obj"}]

        for s in subs:
            subj_text = expand_np_with_modifiers(s, include_det)
            triples.add((subj_text, "subj", root.lemma_))

        for o in objs:
            obj_text = expand_np_with_modifiers(o, include_det)
            triples.add((root.lemma_, "obj", obj_text))

        for prep in (c for c in root.children if c.dep_ == "prep"):
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj:
                pobj_text = expand_np_with_modifiers(pobj, include_det)
                triples.add((root.lemma_, f"prep_{prep.text}", pobj_text))
        return triples

    # -------------------
    # CASE 4: Source questions "From which cell type does X arise?"
    # -------------------
    if root.pos_ == "VERB" and any(c.text.lower() in {"from", "of"} for c in root.children if c.dep_ == "prep"):
        subs = [c for c in root.children if c.dep_ == "nsubj"]
        for s in subs:
            subj_text = expand_np_with_modifiers(s, include_det)
            for prep in (c for c in root.children if c.dep_ == "prep"):
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj:
                    pobj_text = expand_np_with_modifiers(pobj, include_det)
                    triples.add((subj_text, f"{root.lemma_}_by", pobj_text))
        return triples

    # -------------------
    # CASE 5: Yes/No copular "Is X Y?" / "Was X Y?"
    # -------------------
    if root.lemma_ == "be":
        subs = [c for c in root.children if c.dep_ == "nsubj"]
        attrs = [c for c in root.children if c.dep_ in {"attr", "acomp"}]
        for s in subs:
            subj_text = expand_np_with_modifiers(s, include_det)
            for a in attrs:
                attr_text = expand_np_with_modifiers(a, include_det)
                triples.add((subj_text, "isa", attr_text))
        return triples

    # -------------------
    # CASE 6: Modal + action "Can X VERB Y?"
    # -------------------
    if root.pos_ == "VERB" and any(c.pos_ == "AUX" and c.tag_ == "MD" for c in root.lefts):
        subs = [c for c in root.children if c.dep_ == "nsubj"]
        objs = [c for c in root.children if c.dep_ in {"dobj", "obj"}]

        for s in subs:
            subj_text = expand_np_with_modifiers(s, include_det)
            triples.add((subj_text, "subj", root.lemma_))

        for o in objs:
            obj_text = expand_np_with_modifiers(o, include_det)
            triples.add((root.lemma_, "obj", obj_text))

        for prep in (c for c in root.children if c.dep_ == "prep"):
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj:
                pobj_text = expand_np_with_modifiers(pobj, include_det)
                triples.add((root.lemma_, f"prep_{prep.text}", pobj_text))
        return triples

    return triples



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
        G.add_node(h);
        G.add_node(t)
        G.add_edge(h, t, rel=r)
    return G


def plot_graph(G, title=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=2400, font_size=10, font_weight="bold", arrows=True, arrowsize=18)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'rel'), font_size=9)
    if title: plt.title(title)
    plt.tight_layout();
    plt.show()


# ---------- ) JSON with ID + Dictionary ----------
def triples_to_id_dictionary(triples, tasks='answer the questions'):
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

    return {"entity_dict": entity_dict, "relation_dict": relation_dict, "edges": edges, "tasks": tasks}


# ---------- Utility ----------
def json_dump_str(obj, indent=0):
    """Return compact JSON string by default; pretty-print if indent>0."""
    if indent:
        return json.dumps(obj, ensure_ascii=False, indent=indent)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------- ) Codebook ----------

def all_chains_no_subchains(edges, use_full_edges=True):
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
        for i in range(len(b) - la + 1):
            if b[i:i + la] == a:
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
        _eid(h);
        _rid(r);
        _eid(t)

    # Stable short id for this codebook
    sid_src = json_dump_str({"e": entities, "r": relations})
    sid = hashlib.sha1(sid_src.encode("utf-8")).hexdigest()[:10]

    codebook = {
        "sid": sid,
        "e": entities,  # entity dictionary
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


def make_edges_message(sid: str, edges: List[List[int]], use_full_edges: bool = True) -> str:
    # Send repeatedly; tiny payload (ids only)
    if use_full_edges:
        json_msg = json_dump_str({"sid": sid, 'questions([e,r,e])': all_chains_no_subchains(edges, use_full_edges)})
    else:
        json_msg = json_dump_str({"sid": sid, "edges([e,r,e])": edges,
                                  'questions(edges[i])': all_chains_no_subchains(edges, use_full_edges)})

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
            ent2id[h] = len(ent2id);
            add_e.append(h)
        if r not in rel2id:
            rel2id[r] = len(rel2id);
            add_r.append(r)
        if t not in ent2id:
            ent2id[t] = len(ent2id);
            add_e.append(t)
    delta = {}
    if add_e: delta["add_e"] = add_e
    if add_r: delta["add_r"] = add_r
    return delta


def make_delta_message(sid: str, delta: dict) -> Optional[str]:
    if not delta: return None
    out = {"sid": sid}
    out.update(delta)
    return json_dump_str(out)


# ---------------- tests ----------------
if __name__ == "__main__":
    questions = [
        # "Is the Great Wall of China located in China?",
        # "Does the Great Wall span over 13000 miles?",
        # "Was the Great Wall built during the Ming Dynasty?",
        # # "Can the Great Wall be seen from space?",
        # "Is the Great Wall made of stone and brick?",
        # "Does the Great Wall have watchtowers?",
        # "Was the Great Wall constructed over 2000 years?",
        # "Is the Great Wall an UNESCO World Heritage Site?",
        # "Does the Great Wall stretch across the northern China?",
        # "Are millions of tourists visiting the Great Wall annually?",
        # "The Great Wall is visible from low Earth orbit.",
        # "Is the Great Wall of China located in China? Does the Great Wall span over 13000 miles? Was the Great Wall built during the Ming Dynasty? Can the Great Wall be seen from space? Is the Great Wall made of stone and brick?"
        # "Gold is heavier than silver.",
        # "Basal cell skin cancer, also known as basal cell carcinoma (BCC), is the most common type of skin cancer.",
        # "About 3 million cases of basal cell skin cancer are diagnosed every year in the United States.",
        # "The good news is it can be cured in most cases.",
        # "Treatment usually involves surgery to remove the cancer.",
        # "If caught early, is it easily treatable and curable?",
        # "What is the most common type of skin cancer?",
        "From which cell type does basal cell carcinoma arise?",
        "Which anatomical locations are most commonly affected by basal cell carcinoma?",
        "What is the primary risk factor for basal cell carcinoma?",
        "How does fair skin affect the risk of developing BCC?",
        "Can a history of radiation therapy increase BCC risk?",
        "What is a common presentation of BCC on the skin?",
        "What is the role of biopsy in BCC diagnosis?",
        "Which systemic therapy may be considered for BCC?",
        "What is included in the management of BCC recurrence?",
        "What is the recommended frequency for full skin exams in BCC follow-up?",
        "Can BCC spread to lymph nodes?",
        "What is a brown or glossy black bump with rolled border a symptom of?"
    ]

    for s in questions:
        triples = sentence_relations(s, include_det=False)
        # id_json = triples_to_id_dictionary(triples)
        # print(json_dump_str(id_json, indent=2))
        print('code book method')
        codebook, ent2id, rel2id = build_codebook_from_triples(triples)
        msg1 = make_codebook_message(codebook)  # send once

        edges = edges_from_triples(triples, ent2id, rel2id)
        msg2 = make_edges_message(codebook["sid"], edges)  # send many times or once

        print(msg1)
        print(msg2)
        print(s, "->")
        for t in triples:
            print("   ", t)
        G = build_graph(triples)

        print(triples)
        plot_graph(G, s)

# python py_files/graph_generator/generator_with_rules_v3.py

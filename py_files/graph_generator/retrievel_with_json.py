## Version 2 - Fixed for Passive Voice
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

word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")


def get_word_embeddings(list_of_text,word_emb):
    """
    list_of_text: ['str1 str2 ...',]
    word_emb: embedding model

    list_of_text_embeddings:  [embedding_vals,...]
    """

    list_of_text_embeddings = [word_emb._embed_text(text) for text in list_of_text]


    return list_of_text_embeddings

def get_code_book(question_prompt):
    triples = sentence_relations(question_prompt, include_det=False)

    codebook, ent2id, rel2id = build_codebook_from_triples(triples)

    edges = edges_from_triples(triples, ent2id, rel2id)

    dict_2 = {"edges([e,r,e])": edges,'questions(edges[i])':all_chains_no_subchains(edges,False)}

    codebook.update(dict_2)

    codebook.pop('sid') 

    return codebook


def update_the_index(codebook_main,codebook_sub,select_feature):
    items_needs_merged = codebook_sub[select_feature]
    items_main = codebook_main[select_feature]
    index_item_sub = {val: idx for idx, val in enumerate(codebook_sub[select_feature])}
    index_item_main = {val: idx for idx, val in enumerate(codebook_main[select_feature])}
    total_item_num = len(items_main)
    new_index_replacement_for_sub = {}
    new_added_items = []

    for item_sub in items_needs_merged:
        if item_sub in items_main:
            new_index_replacement_for_sub[index_item_sub[item_sub]] = index_item_main[item_sub]
        else:
            # update the index_item_main by adding at total_item_num+1 space
            total_item_num+=1
            new_index_replacement_for_sub[index_item_sub[item_sub]] = total_item_num
            index_item_main[item_sub] = total_item_num
            new_added_items.append(item_sub)

    return new_index_replacement_for_sub,index_item_main,new_added_items



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


def combine_updated_edges(edges_main,edges_sub):
    index_item_sub = {val: idx for idx, val in enumerate(edges_sub)}
    index_item_main = {val: idx for idx, val in enumerate(edges_main)}

    total_item_num = len(edges_main)
    new_index_replacement_for_sub = {}

    for item_sub in index_item_sub:
        if item_sub in total_item_num:
            new_index_replacement_for_sub[index_item_sub[item_sub]] = index_item_main[item_sub]
        else:
            # update the index_item_main by adding at total_item_num+1 space
            total_item_num+=1
            new_index_replacement_for_sub[index_item_sub[item_sub]] = total_item_num
            index_item_main[item_sub] = total_item_num

    return new_index_replacement_for_sub,index_item_main


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


def merging_codebook(codebook_main,codebook_sub,word_emb=word_emb):

    if codebook_main:
        # get the ents and rs needs to be merged in to codebook_main
        rule = codebook_main['rule']

        questions_needs_merged = codebook_sub['questions(edges[i])']
        lst_questions_main = codebook_main['questions_lst']

        edge_mat_needs_merged = codebook_sub['edges([e,r,e])']
        edge_mat_main = codebook_main['edge_matrix']

        # get new index and updated index for main codebook and sub codebook

        new_index_replacement_for_ent_sub,index_ent_main,new_added_ents = update_the_index(codebook_main,codebook_sub,'e')

        new_index_replacement_for_r_sub,index_r_main,new_added_rs = update_the_index(codebook_main,codebook_sub,'r')

        # get newly added entities word embeddings
        new_ent_embeds = get_word_embeddings(new_added_ents,word_emb)

        # get newly added relations word embeddings
        new_r_embeds = get_word_embeddings(new_added_rs,word_emb)

        # update the edges matrix needs to be merged
        edge_mat_needs_merged_remapped = remap_edges_matrix(edge_mat_needs_merged, new_index_replacement_for_ent_sub, new_index_replacement_for_r_sub)

        # combine the edges matrix and new index for edges_sub
        new_index_replacement_for_edges_sub,index_edges_main = combine_updated_edges(edge_mat_main,edge_mat_needs_merged_remapped)

        # update the questions index 
        updated_questions_sub = remap_question_indices(questions_needs_merged, new_index_replacement_for_edges_sub)

        # combine the questions
        lst_questions_main.append(updated_questions_sub)


        final_codebook = {
            "e": index_ent_main.keys(),   # entity dictionary 
            "r": index_r_main.keys(),  # relation dictionary 
            'edge_matrix':index_edges_main.keys(),
            "questions_lst":lst_questions_main,
            "rule": rule,
            "e_embeddings": codebook_main['e_embeddings']+new_ent_embeds,   # add the new ent embeds
            "r_embeddings": codebook_main['r_embeddings']+new_r_embeds,  # add the new r embeds
        }
    else:
        # main codebook is empty
        final_codebook = {
            "e": codebook_sub['e'],  
            "r": codebook_sub['r'],  
            'edge_matrix':codebook_sub['edges([e,r,e])'],
            "questions_lst":[codebook_sub['questions(edges[i])']],
            "rule": codebook_sub['rule'],
            "e_embeddings": get_word_embeddings(codebook_sub['e'],word_emb),  
            "r_embeddings": get_word_embeddings(codebook_sub['r'],word_emb),  
        }

    return final_codebook


def decode_question(question, codebook_main, fmt='words'):
    """
    question: list[int] of edge indices
    codebook_main:
        {
            "e": [str, ...],
            "r": [str, ...],
            "edge_matrix": [[e_idx, r_idx, e_idx], ...],  # list or np.ndarray
            "questions_lst": [[[edges index,...],...],...]
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

def get_topk_word_embedding_batched(
    questions: List[List[int]],
    codebook_main: Dict[str, Any],
    top_k: int = 3,
    question_batch_size: int = 1,         # number of query questions processed per time
    questions_db_batch_size: int = 1,     # number of db questions processed per time
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Top-k similarity with **decode_question(..., fmt='embeddings')** in two-way batches.

    Returns:
      {query_idx: [{"score": float, "questions_index": int, "question_index": int}, ...], ...}
    """
    # infer embedding dim from e_embeddings (fallback to r if needed)
    if "e_embeddings" in codebook_main and len(codebook_main["e_embeddings"]) > 0:
        dim = len(codebook_main["e_embeddings"][0])
    elif "r_embeddings" in codebook_main and len(codebook_main["r_embeddings"]) > 0:
        dim = len(codebook_main["r_embeddings"][0])
    else:
        raise ValueError("Cannot infer embedding dimension from codebook_main.")

    # Flatten DB questions and keep a map to (questions_index, question_index)
    questions_lst = codebook_main["questions_lst"]
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

    # Process queries in batches
    for q_start in range(0, N_total, question_batch_size):
        q_end = min(q_start + question_batch_size, N_total)
        q_batch_idx = list(range(q_start, q_end))
        q_batch_lists = [questions[i] for i in q_batch_idx]

        # --- embed current query batch via decode_question ---
        q_mat = _embed_questions_with_decode(q_batch_lists, codebook_main, dim)  # (Qb, d)
        Qb = q_mat.shape[0]

        # running top-k for this query batch
        best_scores = [np.array([], dtype=np.float32) for _ in range(Qb)]
        best_cols   = [np.array([], dtype=np.int32)   for _ in range(Qb)]

        # Stream over DB in batches
        for db_start in range(0, M_total, questions_db_batch_size):
            db_end = min(db_start + questions_db_batch_size, M_total)
            db_batch_lists = db_questions[db_start:db_end]

            # --- embed current DB batch via decode_question ---
            db_mat = _embed_questions_with_decode(db_batch_lists, codebook_main, dim)  # (Db, d)
            Db = db_mat.shape[0]
            if Db == 0:
                continue

            # similarities (Qb x Db)
            sims = _cosine_sim(q_mat, db_mat)
            k_local = min(top_k, Db)

            # Merge batch top-k per query
            for i in range(Qb):
                row = sims[i]
                # choose top-k indices within this batch
                cand_idx = np.argpartition(-row, k_local - 1)[:k_local]
                cand_idx = cand_idx[np.argsort(-row[cand_idx])]  # sort by score desc
                batch_scores = row[cand_idx]
                batch_cols   = cand_idx + db_start  # map to global DB index
                merged_scores, merged_cols = _topk_merge(
                    best_scores[i], best_cols[i], batch_scores, batch_cols, top_k
                )
                best_scores[i] = merged_scores
                best_cols[i]   = merged_cols

        # Commit this query batch to final results
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


def coarse_filter():


    return 0


# python py_files/graph_generator/retrievel_with_json.py
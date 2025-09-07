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
def get_code_book(prompt,type = 'questions'):
    """
    prompt:str
    type: one of 'questions' and 'answers'
    """
    triples = sentence_relations(prompt, include_det=False)

    codebook, ent2id, rel2id = build_codebook_from_triples(triples)

    edges = edges_from_triples(triples, ent2id, rel2id)

    feat_name = type+'(edges[i])'

    dict_2 = {"edges([e,r,e])": edges,feat_name:all_chains_no_subchains(edges,False)}

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
def merging_codebook(codebook_main,codebook_sub,type='questions',word_emb=word_emb,use_thinkings = False):

    feat_name = type+'(edges[i])'

    if type=='questions':
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



    if codebook_main:
        # get the ents and rs needs to be merged in to codebook_main
        questions_needs_merged = codebook_sub[feat_name]
        lst_questions_main = codebook_main[main_feat_name]

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


        ### add the knowledge graph and it's related index


        codebook_main["e"].extend(new_added_ents)
        codebook_main["r"].extend(new_added_rs)
        codebook_main["edge_matrix"] = index_edges_main
        codebook_main[main_feat_name] = lst_questions_main
        codebook_main["e_embeddings"] = codebook_main['e_embeddings'] + new_ent_embeds
        codebook_main["r_embeddings"] = codebook_main['r_embeddings'] + new_r_embeds



        if type == 'thinkings':
            codebook_main['questions_to_thinkings'][len(codebook_main['questions_lst'])-1] = len(codebook_main[main_feat_name])-1


    else:
        # main codebook is empty
        codebook_main = {
            "e": codebook_sub['e'],  
            "r": codebook_sub['r'],  
            'edge_matrix':codebook_sub['edges([e,r,e])'],
            main_feat_name:[codebook_sub[feat_name]],
            unupdated_feat_name1:[],
            "rule": codebook_sub['rule'],
            "e_embeddings": get_word_embeddings(codebook_sub['e'],word_emb),  
            "r_embeddings": get_word_embeddings(codebook_sub['r'],word_emb),  
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
    final_codebook = merging_codebook(codebook_with_qa,codebook_sub_t,type='answers')

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

def decode_questions(questions, codebook_main, fmt='words'):
    """
    Decode a list of questions using decode_question.
    
    questions: list of list[int]
        Each inner list is a sequence of edge indices.
    """
    return [decode_question(q, codebook_main, fmt=fmt) for q in questions]


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
    emb: HuggingFaceEmbeddings,                 # ← move before defaults
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
    emb,
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
            entitie_set_len+=1
            new_ent_pos = entitie_set_len
            entitie_set.append(ent)

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
    edge_mat_for_q_sub = remap_edges(codebook_sub_q['edge_matrix'], entitie_index_dict_q, r_index_dict_q)

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
    for ent in codebook_sub_q['e']:
        # check the ent in entities_lst or not
        if ent in entitie_set:
            new_ent_pos = entitie_set.index(ent)
        else:
            new_ent_pos = entitie_set_len
            entitie_set.append(ent)
            edge_matrix_sub_len+=1

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
    edge_mat_for_q_sub = remap_edges(codebook_sub_q['edge_matrix'], entitie_index_dict_q, r_index_dict_q)

    # update the edges
    edge_matrix_sub_len = len(edge_matrix_sub)
    edge_pos = 0
    edge_mat_for_q_sub_dict = {}

    for edge in edge_mat_for_q_sub:
        if edge in edge_matrix_sub:
            new_edge_pos = edge_matrix_sub.index(edge)
        else:
            edge_matrix_sub_len+=1
            new_edge_pos = edge_matrix_sub_len
            edge_matrix_sub.append(edge)

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


def combine_ents(codebook_main: Dict[str, Any],
                 min_exp_num: int = 2, # expected min numbers of candidates in each cluster
                 max_exp_num: int = 20, # expected max numbers of candidates in each cluster
                 use_thinking = True,
                 random_state: int = 0):

    E = codebook_main['e']
    X = np.asarray(codebook_main['e_embeddings'])
    assert X.ndim == 2 and len(E) == X.shape[0], "Mismatch between 'e' and 'e_embeddings' sizes."

    n = X.shape[0]
    # Nothing to merge if 0/1/2 entities
    if n <= 2:
        return codebook_main, {i: i for i in range(n)}, {i: [i] for i in range(n)}

    # L2 normalize to make KMeans distance closer to cosine behavior
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # Choose k via silhouette (primary) + elbow (secondary)
    num_ents = len(E)
    k_min = num_ents/max_exp_num
    k_max = num_ents/min_exp_num

    k_low = max(k_min, 2)
    k_high = max(min(k_max, n - 1), k_low)
    cand_ks = list(range(k_low, k_high + 1))

    best_k = None
    best_sil = -1.0
    inertia_by_k = {}
    sil_by_k = {}

    for k in cand_ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_norm)
        # Silhouette requires at least 2 clusters and less than n samples per cluster is fine
        sil = silhouette_score(X_norm, labels, metric='euclidean')
        inertia_by_k[k] = km.inertia_
        sil_by_k[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k
        elif np.isclose(sil, best_sil, rtol=0, atol=1e-6):
            # tie-breaker: lower inertia (better elbow)
            if inertia_by_k[k] < inertia_by_k.get(best_k, np.inf):
                best_k = k

    # Final fit with best_k
    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_norm)
    centroids = km.cluster_centers_  # in normalized space

    # find in each cluster which candidate is nearest to the centroids
    map_dict = {}
    for cluster_id in range(best_k):
        # indices of points in this cluster
        cluster_points_idx = np.where(labels == cluster_id)[0]
        cluster_points = X[cluster_points_idx]

        # compute distances to centroid
        distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)

        # nearest point
        nearest_idx = cluster_points_idx[np.argmin(distances)]

        # update map_dict
        for idx in cluster_points_idx:
          if idx != nearest_idx:
            map_dict[idx] = nearest_idx

    # update the edges matrix index
    mapped_edges = []
    for e1, r, e2 in codebook_main['edge_matrix']:
        new_e1 = map_dict.get(e1, e1)  
        new_e2 = map_dict.get(e2, e2)
        mapped_edges.append([new_e1, r, new_e2])

    codebook_main['edge_matrix'] = mapped_edges

    # remove the merged index
    merged_indexes = map_dict.values()

    # update the entities index and entities embeddings 
    new_ent_pos = 0
    ent_pos_dict = {}

    for ent_pos in range(num_ents):

      if ent_pos not in merged_indexes:
        ent_pos_dict[ent_pos] = new_ent_pos
        new_ent_pos += 1
        
        
    kept_indexes = list(ent_pos_dict.keys())
    new_ent = np.array(codebook_main['e'])[kept_indexes]
    new_ent_embeddings = np.array(codebook_main['e_embeddings'])[kept_indexes]


    # update the edges matrix indexes again
    new_mapped_edges = []
    for e1, r, e2 in codebook_main['edge_matrix']:
        new_e1 = ent_pos_dict.get(e1, e1)  
        new_e2 = ent_pos_dict.get(e2, e2)
        new_mapped_edges.append([new_e1, r, new_e2]) 


    # update the edges matrix indices for questions, answers

    # remove the reptitive edges
    edges_index_dict = {}
    final_mapped_edges = []
    edges_index = 0
    new_edges_index = 0
    for edge in final_mapped_edges:
        if edge not in final_mapped_edges:
            final_mapped_edges.append(edge)
            edges_index_dict[edges_index] = new_edges_index
            new_edges_index+=1
        edges_index+=1

    # update for questions_lst
    def update_indexing_qat(struct, mapping):
        if isinstance(struct, list):
            return [update_indexing_qat(x, mapping) for x in struct]
        return mapping.get(struct, struct)
    
    if codebook_main['questions_lst']:
        codebook_main['questions_lst'] = update_indexing_qat(codebook_main['questions_lst'],edges_index_dict)

    if codebook_main['answers_lst']:
        codebook_main['answers_lst'] = update_indexing_qat(codebook_main['answers_lst'],edges_index_dict)

    if use_thinking & codebook_main['thinkings_lst']:
        codebook_main['thinkings_lst'] = update_indexing_qat(codebook_main['thinkings_lst'],edges_index_dict)


    # update the final edges matrix, entities and entities embeddings for code bookmain
    codebook_main['e'] = new_ent
    codebook_main['e_embeddings'] = new_ent_embeddings
    codebook_main['edge_matrix'] = final_mapped_edges


    return codebook_main

# =========================
# END-TO-END TEST HARNESS
# =========================
if __name__ == "__main__":
    import random
    from typing import Sequence

    print("\n========== Graph/Codebook E2E Test ==========")

    # ---------- 0) Embedding fallbacks ----------
    class TinyRandomEmbeddings(Embeddings):
        """
        A stable random embedder (deterministic): hash(text) -> seed -> vector
        Used as a fallback when external models are unavailable.
        """
        def __init__(self, dim: int = 64):
            self.dim = dim

        def _vec(self, text: str) -> list[float]:
            rnd = random.Random(hash(text) & 0xffffffff)
            return [rnd.uniform(-1, 1) for _ in range(self.dim)]

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [self._vec(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._vec(text)

    # Ensure word_emb is usable; otherwise replace with TinyRandomEmbeddings
    try:
        _ = word_emb.dim
    except Exception:
        print("[WARN] GloVe KeyedVectors not ready. Using TinyRandomEmbeddings as word embedding fallback.")
        word_emb = TinyRandomEmbeddings(dim=64)

    # Prepare sentence embedder for rerank; fallback if model can't be loaded
    try:
        sent_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _ = sent_emb.embed_query("hello")
        print("[OK] Using HuggingFaceEmbeddings: sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[WARN] Cannot load HuggingFaceEmbeddings ({e}). Using TinyRandomEmbeddings fallback.")
        sent_emb = TinyRandomEmbeddings(dim=64)

    # ---------- 1) Build question & answer codebooks from raw text ----------
    # Multi-sentence to trigger diverse extraction paths (passive / copular ADJ / VERB)
    questions_text = (
        "Is the Great Wall of China visible from space? "
        "Frankenstein was written by Mary Shelley in 1818. "
        "The novel inspired many films."
    )
    answers_text_1 = (
        "The Great Wall is not visible from space with the naked eye. "
        "Mary Shelley wrote Frankenstein in 1818."
    )
    answers_text_2 = (
        "From low Earth orbit, the Great Wall cannot be seen unaided. "
        "Frankenstein is a novel authored by Mary Shelley."
    )
    answers_text_3 = (
    "Frankenstein was published in 1818 by Mary Shelley. "
    "The Great Wall cannot be seen with unaided eyes from orbit."
)

    # Build separate codebooks (questions / answers)
    codebook_q = get_code_book(questions_text, type='questions')   
    codebook_a1 = get_code_book(answers_text_1, type='answers')
    codebook_a2 = get_code_book(answers_text_2, type='answers')
    codebook_a3 = get_code_book(answers_text_3, type='answers')

    print("\n== Sample codebook (questions) ==")
    print(json_dump_str({
        "e": codebook_q["e"][:8],
        "r": codebook_q["r"][:8],
        "edges_sample": codebook_q["edges([e,r,e])"][:6],
        "#edges": len(codebook_q["edges([e,r,e])"]),
        "#q_chains": len(codebook_q["questions(edges[i])"]),
    }, indent=2))

    print("\n== Sample codebook (answers 1) ==")
    print(json_dump_str({
        "e": codebook_a1["e"][:8],
        "r": codebook_a1["r"][:8],
        "edges_sample": codebook_a1["edges([e,r,e])"][:6],
        "#edges": len(codebook_a1["edges([e,r,e])"]),
        "#a_chains": len(codebook_a1["answers(edges[i])"]),
    }, indent=2))

    # ---------- 2) Initialize main codebook & merge Q/A ----------
    # Start with empty -> merge questions -> merge answers_1 -> merge answers_2
    codebook_main = merging_codebook(None, codebook_q, type='questions', word_emb=word_emb)
    codebook_main = merging_codebook(codebook_main, codebook_a1, type='answers',   word_emb=word_emb)
    codebook_main = merging_codebook(codebook_main, codebook_a2, type='answers',   word_emb=word_emb)
    codebook_main = merging_codebook(codebook_main, codebook_a3, type='answers',   word_emb=word_emb)
    print("---------", codebook_main)

    print("\n== Codebook (main) stats after merges ==")
    print(json_dump_str({
        "#entities": len(codebook_main["e"]),
        "#relations": len(codebook_main["r"]),
        "#edges": len(codebook_main["edge_matrix"]),
        "#questions_buckets": len(codebook_main["questions_lst"]),
        "#answers_buckets": len(codebook_main["answers_lst"]),
        "rule": codebook_main["rule"],
    }, indent=2))

    # ---------- 3) Pull some question chains & test decode ----------
    if not codebook_main["questions_lst"] or not codebook_main["questions_lst"][0]:
        raise RuntimeError("No question chains produced; try a different input sentence.")

    all_q_chains = codebook_main["questions_lst"][0]
    query_chains = all_q_chains[: min(3, len(all_q_chains))]  # take a few

    print("\n== Query chains (edge indices) ==")
    for i, ch in enumerate(query_chains):
        print(f"Q{i}: {ch}")

    # Test decode_question on: single int, single chain, batch chains (by yourself)
    # Single edge index (if available)
    if query_chains and query_chains[0]:
        edge_idx = query_chains[0][0]
        print("\n-- decode_question(single int, words) --")
        print(decode_question([edge_idx], codebook_main, fmt='words'))  # wrap into list to keep API consistent

    # Single chain
    print("\n-- decode_question(single chain, edges/words/embeddings) --")
    one_chain = query_chains[0]
    print("edges:",       decode_question(one_chain, codebook_main, fmt='edges'))
    print("words:",       decode_question(one_chain, codebook_main, fmt='words'))
    print("embeddings: [#triples, dims?] ->", len(decode_question(one_chain, codebook_main, fmt='embeddings')), "triples")

    # ---------- 4) Word-embedding coarse top-k over whole DB ----------
    coarse_topk = get_topk_word_embedding_batched(
        questions=query_chains,
        codebook_main=codebook_main,
        top_k=3,
        question_batch_size=2,
        questions_db_batch_size=8,
    )
    print("\n== Coarse Top-K (word embeddings) ==")
    for qi, lst in coarse_topk.items():
        print(f"Q{qi} → {lst}")

    # ---------- 5) Sentence-embedding rerank ----------
    reranked = rerank_with_sentence_embeddings(
        questions=query_chains,
        codebook_main=codebook_main,
        coarse_topk=coarse_topk,
        emb=sent_emb,
        top_m=3,  # keep top-2 to show list
    )
    print("\n== Reranked (sentence embeddings) ==")
    for qi, lst in reranked.items():
        print(f"Q{qi} → {lst}")

    # ---------- 6) One-call coarse+fine wrapper ----------
    print("\n== Coarse+Fine (wrapper: coarse_filter) ==")
    wrapper_res = coarse_filter(
        questions=query_chains,
        codebook_main=codebook_main,
        emb=sent_emb,
        top_k=3,
        question_batch_size=2,
        questions_db_batch_size=8,
        top_m=2,
    )
    for qi, lst in wrapper_res.items():
        print(f"Q{qi} → {lst}")

    # ---------- 7) Attach answers to top-m results ----------
    topm_with_answers = add_answers_to_filtered_lst(wrapper_res, codebook_main)
    print("\n== Top-m with attached answers (edge-index chains) ==")
    for qi, lst in topm_with_answers.items():
        print(f"\nQ{qi}:")
        for item in lst:
            print(json_dump_str(item, indent=2))

    # ---------- 8) Decode & show final top-1 per query ----------
    print("\n== Final decode (top-1 per query) ==")
    for qi, lst in wrapper_res.items():
        if not lst:
            print(f"Q{qi}: <no candidate>")
            continue
        best = lst[0]
        qi_db = best["questions_index"]
        qj_db = best["question_index"]
        best_edges_idx_chain = codebook_main["questions_lst"][qi_db][qj_db]

        decoded_edges = decode_question(best_edges_idx_chain, codebook_main, fmt='edges')
        decoded_words = decode_question(best_edges_idx_chain, codebook_main, fmt='words')
        decoded_vecs  = decode_question(best_edges_idx_chain, codebook_main, fmt='embeddings')

        print(f"\nQ{qi} top-1:")
        print("- edges indices:", decoded_edges)
        print("- words triples:", decoded_words)
        print("- embeddings triples: count =", len(decoded_vecs))

    # ---------- 9) Find overlapped answer snippets (contiguous runs) + BEFORE/AFTER texts ----------
    print("\n== Overlapped contiguous answer runs among selected candidates (with BEFORE/AFTER text) ==")

    for qi, lst in topm_with_answers.items():
        if not lst:
            print(f"\nQ{qi}: <no candidate with answers>")
            continue

        print(f"\nQ{qi} — BEFORE (raw candidate answer texts):")
        per_candidate_answers_texts = []  
        for rank, item in enumerate(lst):
            answers_bucket = item['answers(edges[i])']  
            texts = decode_answers_bucket_to_texts(answers_bucket, codebook_main)
            per_candidate_answers_texts.append(texts)
    
            joined = " | ".join(texts) if texts else "<empty>"
            print(f"  - cand#{rank}: {joined}")

    
        answers_buckets_for_overlap = []
        for item in lst:
            answers_buckets_for_overlap.append(item['answers(edges[i])'])

        overlaps = find_overlapped_answers(answers_buckets_for_overlap)

        print("----------",overlaps)
        if overlaps:
            print(f"Q{qi} — AFTER (merged/common segments):")
            for seg_id, edge_run in enumerate(overlaps):
    
                text_after = decode_chain_to_text(edge_run, codebook_main)
                print(f"  * segment#{seg_id}: {text_after}   [edges: {edge_run}]")
        else:
            print(f"Q{qi} — AFTER: <no common contiguous runs found>")

    print("\n========== Test completed. ==========\n")

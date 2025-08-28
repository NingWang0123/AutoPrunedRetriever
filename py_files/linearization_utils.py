from typing import List, Dict, Optional, Iterable, Tuple
import networkx as nx
import re
import json, hashlib

def _sanitize(text: str) -> str:
    """Clean up newlines/multiple spaces to ensure stable vectorization."""
    s = re.sub(r'\s+', ' ', str(text)).strip()
    return s

def _pick_relation(data: Dict) -> str:
    """Pick the relation name from common keys, fallback by priority."""
    for key in ("causal_type", "relation", "label", "type", "rel"):
        if key in data and data[key]:
            return _sanitize(data[key])
    return "related_to"

def linearize_graph(
    G: nx.Graph,
    *,
    undirected_mode: str = "single",  # ["single", "both"]
    default_rel: str = "related_to"
) -> str:
    """
    Serialize any graph (directed/undirected) into multi-line triples, one edge per line:
    Format: HEAD:<u>  REL:<rel>  TAIL:<v>
    - Directed graph: output once along edge direction
    - Undirected graph: single=output once (u,v), both=output in two directions
    """
    lines: List[str] = []

    is_directed = G.is_directed()

    iterator = G.edges(data=True) if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) \
        else G.edges(keys=True, data=True)

    for e in iterator:
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            u, v, k, data = e
        else:
            u, v, data = e

        u_str = _sanitize(u)
        v_str = _sanitize(v)
        rel = _pick_relation(data) if data else default_rel

        if is_directed:
            lines.append(f"HEAD:{u_str}  REL:{rel}  TAIL:{v_str}")
        else:
            if undirected_mode == "both":
                lines.append(f"HEAD:{u_str}  REL:{rel}  TAIL:{v_str}")
                lines.append(f"HEAD:{v_str}  REL:{rel}  TAIL:{u_str}")
            else:
                h, t = (u_str, v_str) if u_str <= v_str else (v_str, u_str)
                lines.append(f"HEAD:{h}  REL:{rel}  TAIL:{t}")

    lines = sorted(set(lines))
    return "\n".join(lines)

def _needs_article(s: str) -> bool:
    w = s.strip()
    return not w[:1].isupper() and not w.lower().startswith(("the ", "a ", "an "))

def _with_article(s: str) -> str:
    s = s.strip()
    return f"the {s}" if _needs_article(s) else s

def _edge_to_nl(h: str, rel: str, t: str) -> str:
    r = (rel or "").lower()
    h2, t2 = h.strip(), t.strip()

    if r == "isa":
        return f"{h2} is a {t2}"
    if r == "property":
        return f"{h2} is {t2}"
    if r.startswith("prep_"):
        p = r.replace("prep_", "")
        if p == "in":      return f"{h2} is located in {_with_article(t2)}"
        if p == "of":      return f"{h2} is about {_with_article(t2)}"
        if p == "during":  return f"{h2} is during {_with_article(t2)}"
        if p == "on":      return f"{h2} is on {_with_article(t2)}"
        if p == "at":      return f"{h2} is at {_with_article(t2)}"
        return f"{h2} {p} {_with_article(t2)}"

    if r in ("classify", "classified", "classify_as", "classified_as"):
        return f"{h2} is classified as {t2}"
    if r in ("consider", "considered"):
        return f"{h2} is considered {t2}"
    if r in ("locate", "located"):
        return f"{h2} is located in {_with_article(t2)}"

    if r in ("subj", "obj"):
        return f"{h2} relates to {t2}"

    return f"{h2} {r.replace('_',' ')} {t2}"

def linearize_graph_nl(G: nx.Graph) -> str:
    lines = []
    iterator = G.edges(data=True) if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) \
        else G.edges(keys=True, data=True)
    for e in iterator:
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            u, v, k, d = e
        else:
            u, v, d = e
        rel = (d or {}).get("rel") or (d or {}).get("label") or (d or {}).get("causal_type")
        lines.append(_edge_to_nl(str(u), str(rel or "related_to"), str(v)))
    lines = sorted(set(s.strip() for s in lines if s and s.strip()))
    return ",".join(lines) if lines else "__EMPTY_NL__"

# ---------------- v3 helpers (unified JSON) ----------------
def _json_dump(obj, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _triples_from_inputs(g: nx.Graph, rels: Optional[List[Dict]]) -> List[Tuple[str, str, str]]:
    """Prefer triples from 'relations' if they have tail; otherwise derive from graph edges."""
    triples: List[Tuple[str, str, str]] = []
    if rels:
        for r in rels:
            head = r.get('cause') or r.get('head') or r.get('source')
            rel  = r.get('causal_type') or r.get('rel')  or r.get('relation') or "related_to"
            tail = r.get('effect') or r.get('tail') or r.get('target')
            if head and tail:
                triples.append((str(head), str(rel), str(tail)))
    if not triples and g is not None:
        for u, v, d in g.edges(data=True):
            rel = (d or {}).get("rel") or (d or {}).get("label") or (d or {}).get("causal_type") or "related_to"
            triples.append((str(u), str(rel), str(v)))
    return triples

def _triples_to_id_dictionary_v1(tris: List[Tuple[str, str, str]]) -> Dict:
    """v1: {entity_dict, relation_dict, edges}"""
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    entity_dict: List[str] = []
    relation_dict: List[str] = []
    edges: List[List[int]] = []
    def _eid(x: str) -> int:
        if x not in ent2id:
            ent2id[x] = len(entity_dict); entity_dict.append(x)
        return ent2id[x]
    def _rid(x: str) -> int:
        if x not in rel2id:
            rel2id[x] = len(relation_dict); relation_dict.append(x)
        return rel2id[x]
    for h, r, t in tris:
        edges.append([_eid(h), _rid(r), _eid(t)])
    return {"entity_dict": entity_dict, "relation_dict": relation_dict, "edges": edges}

def _build_codebook_v2(tris: List[Tuple[str, str, str]], rule: str = "Reply with a Y/N/? string in order only; no explanations."):
    """v2: (codebook, edges) where codebook has {sid, e, r, rule} and edges is [[h,r,t], ...]."""
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    e: List[str] = []
    r: List[str] = []
    def _eid(x: str) -> int:
        if x not in ent2id:
            ent2id[x] = len(e); e.append(x)
        return ent2id[x]
    def _rid(x: str) -> int:
        if x not in rel2id:
            rel2id[x] = len(r); r.append(x)
        return rel2id[x]
    for h, rel, t in tris:
        _eid(h); _rid(rel); _eid(t)
    sid_src = _json_dump({"e": e, "r": r}, pretty=False)
    sid = hashlib.sha1(sid_src.encode("utf-8")).hexdigest()[:10]
    codebook = {"sid": sid, "e": e, "r": r, "rule": rule}
    edges = [[ent2id[h], rel2id[rel], ent2id[t]] for (h, rel, t) in tris]
    return codebook, edges

def _build_unified_v3(
    tris: List[Tuple[str, str, str]],
    *,
    rule: str = "Reply with a Y/N/? string in order only; no explanations.",
    include_sid: bool = False,  # 你示例里没有 sid，默认关闭；需要可打开
    edge_key: str = "questions([[e,r,e], ...])"  # 按你要求的键名
) -> Dict:
    """
    v3 unified object (your desired shape):
    {
      "e": [...],                # entities
      "r": [...],                # relations
      "rule": "<instruction>",   # rule string
      "questions([[e,r,e], ...])": [[h_id, r_id, t_id], ...],
      # "sid": "<stable-id>"     # optional if include_sid=True
    }
    """
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    e: List[str] = []
    r: List[str] = []
    q_edges: List[List[int]] = []

    def _eid(x: str) -> int:
        if x not in ent2id:
            ent2id[x] = len(e); e.append(x)
        return ent2id[x]

    def _rid(x: str) -> int:
        if x not in rel2id:
            rel2id[x] = len(r); r.append(x)
        return rel2id[x]

    for h, rel, t in tris:
        q_edges.append([_eid(h), _rid(rel), _eid(t)])

    out = {
        "e": e,
        "r": r,
        #"rule": rule,
        edge_key: q_edges,
    }

    if include_sid:
        # 与之前一致的稳定短ID算法（基于 e/r）
        sid_src = _json_dump({"e": e, "r": r}, pretty=False)
        out["sid"] = hashlib.sha1(sid_src.encode("utf-8")).hexdigest()[:10]

    return out



# Graph representation structure of page content in document
def build_relationship_text(
    question: str,
    G: nx.Graph,
    relations: Optional[List[Dict]] = None,
    *,
    include_question: bool = False,
    include_graph_block: bool = False,
    include_relations_block: bool = False,
    include_nl_relation: bool = False,   # natural-language linearization
    include_json_block: bool = True,     # append JSON representation
    json_style: str = "v3",              # NEW DEFAULT: "v3" | "id_dict" | "codebook_edges"
    json_pretty: bool = False,           # pretty-print JSON if True
) -> str:
    """
    Unified construction of text for embedding:
    - [QUESTION] Original question (optional)
    - [GRAPH]    Linearized triples (symbolic)
    - [TRIPLES]  Serialized triples (symbolic, controlled by include_relations_block)
    - [NL]       Natural-language relation sentences (controlled by include_nl_relation)
    - [JSON]     Graph-as-JSON ('v3' unified object, or 'id_dict', or 'codebook_edges')
    """
    parts: List[str] = []

    if include_question and question:
        parts.append(f"[QUESTION] {question}")

    def _linearize_from_graph_edges(g: nx.Graph) -> str:
        lines = []
        iterator = g.edges(data=True) if not isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)) \
            else g.edges(keys=True, data=True)
        for e in iterator:
            if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
                u, v, k, data = e
            else:
                u, v, data = e
            rel = (data or {}).get("rel") or (data or {}).get("label") or (data or {}).get("causal_type") or "related_to"
            lines.append(f"HEAD:{_sanitize(u)}  REL:{_sanitize(rel)}  TAIL:{_sanitize(v)}")
        return ",".join(sorted(set(lines))) if lines else "__EMPTY_GRAPH__"

    # [GRAPH]
    if include_graph_block:
        parts.append(_linearize_from_graph_edges(G))

    # [NL]
    if include_nl_relation:
        nl_text = linearize_graph_nl(G)
        parts.append(nl_text if nl_text.strip() else "__EMPTY_NL__")

    # [TRIPLES]
    if include_relations_block:
        triples_text = ""
        if relations:
            def _has_tail(r: Dict) -> bool:
                return bool(r.get('effect') or r.get('tail') or r.get('target'))
            if all(_has_tail(r) for r in relations):
                def _one(r: Dict) -> str:
                    head = r.get('cause') or r.get('head') or r.get('source') or '?'
                    rel  = r.get('causal_type') or r.get('rel') or r.get('relation') or 'related_to'
                    tail = r.get('effect') or r.get('tail') or r.get('target') or '?'
                    return f"{_sanitize(head)} -> {_sanitize(rel)} -> {_sanitize(tail)}"
                triples_text = ",".join(_one(r) for r in relations)
            else:
                triples_text = ",".join(
                    f"{_sanitize(u)} -> {_sanitize((d or {}).get('rel') or (d or {}).get('label') or (d or {}).get('causal_type') or 'related_to')} -> {_sanitize(v)}"
                    for u, v, d in G.edges(data=True)
                )
        else:
            triples_text = ",".join(
                f"{_sanitize(u)} -> {_sanitize((d or {}).get('rel') or (d or {}).get('label') or (d or {}).get('causal_type') or 'related_to')} -> {_sanitize(v)}"
                for u, v, d in G.edges(data=True)
            )
        parts.append(triples_text if triples_text.strip() else "__EMPTY_TRIPLES__")

    # [JSON] v3 / v1 / v2
    if include_json_block:
        triples = _triples_from_inputs(G, relations)
        if not triples:
            parts.append("__EMPTY_JSON__")
        else:
            if json_style == "v3":
                v3_obj = _build_unified_v3(
                    triples,
                    rule="",
                    include_sid=False,                         
                    edge_key="questions([[e,r,e], ...])",    
                )
                json_piece = _json_dump(v3_obj, json_pretty)
            elif json_style == "codebook_edges":
                codebook, edges = _build_codebook_v2(triples)
                json_piece = _json_dump(codebook, json_pretty) + "\n" + _json_dump({"sid": codebook["sid"], "g": edges}, json_pretty)
            else:  # "id_dict"
                id_dict = _triples_to_id_dictionary_v1(triples)
                json_piece = _json_dump(id_dict, json_pretty)
            parts.append(json_piece)

    return ",".join(parts)
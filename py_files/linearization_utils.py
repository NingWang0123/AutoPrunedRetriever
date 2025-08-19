from typing import List, Dict, Optional, Iterable
import networkx as nx
import re

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

    # Choose edge iterator (works for MultiGraph too; with keys=True to preserve multiedges,
    # but here only one relation name is output)
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
            # Undirected graph
            if undirected_mode == "both":
                lines.append(f"HEAD:{u_str}  REL:{rel}  TAIL:{v_str}")
                lines.append(f"HEAD:{v_str}  REL:{rel}  TAIL:{u_str}")
            else:  # "single"
                # Use lexicographic order to avoid duplicates
                h, t = (u_str, v_str) if u_str <= v_str else (v_str, u_str)
                lines.append(f"HEAD:{h}  REL:{rel}  TAIL:{t}")

    # Sort to ensure determinism
    lines = sorted(set(lines))
    return "\n".join(lines)

# Graph representation structure of page content in document
def build_relationship_text(
    question: str,
    G: nx.Graph,
    relations: Optional[List[Dict]] = None,
    *,
    include_question: bool = False,
    include_graph_block: bool = False,
    include_relations_block: bool = True
) -> str:
    """
    Unified construction of text for embedding:
    - [QUESTION] Original question (optional)
    - [GRAPH]    Linearized triples (general, supports any graph)
    - [TRIPLES]  (optional) If relations are provided (e.g., causal extraction result),
                 serialize them as well. If relations miss tails, fall back to graph edges.
    """
    parts: List[str] = []

    # [QUESTION]
    if include_question and question:
        parts.append(f"[QUESTION] {question}")

    # Helper: linearize from graph edges (fallback & for [GRAPH])
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
        return "\n".join(sorted(set(lines))) if lines else "__EMPTY_GRAPH__"

    # [GRAPH]
    if include_graph_block:
        parts.append("[GRAPH]")
        parts.append(_linearize_from_graph_edges(G))

    # [TRIPLES]
    if include_relations_block:
        triples_text = ""
        if relations:
            # Check if every relation has a tail-like field
            def _has_tail(r: Dict) -> bool:
                return bool(r.get("effect") or r.get("tail") or r.get("target"))

            if all(_has_tail(r) for r in relations):
                def _one(r: Dict) -> str:
                    head = r.get('cause') or r.get('head') or r.get('source') or '?'
                    rel  = r.get('causal_type') or r.get('rel') or r.get('relation') or 'related_to'
                    tail = r.get('effect') or r.get('tail') or r.get('target') or '?'
                    return f"{_sanitize(head)} -> {_sanitize(rel)} -> {_sanitize(tail)}"
                triples_text = "\n".join(_one(r) for r in relations)
            else:
                # Fallback: build triples from graph edges to avoid '?' tails
                triples_text = "\n".join(
                    f"{_sanitize(u)} -> {_sanitize((d or {}).get('rel') or (d or {}).get('label') or (d or {}).get('causal_type') or 'related_to')} -> {_sanitize(v)}"
                    for u, v, d in G.edges(data=True)
                )
        else:
            # No relations provided → fall back to graph edges
            triples_text = "\n".join(
                f"{_sanitize(u)} -> {_sanitize((d or {}).get('rel') or (d or {}).get('label') or (d or {}).get('causal_type') or 'related_to')} -> {_sanitize(v)}"
                for u, v, d in G.edges(data=True)
            )

        #parts.append("[TRIPLES]")
        parts.append(triples_text if triples_text.strip() else "__EMPTY_TRIPLES__")

    return "\n".join(parts)


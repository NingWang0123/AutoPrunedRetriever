from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Tuple, Callable, Optional, Sequence, Iterable, Union
import numpy as np

Triple = Tuple[str, str, str]  # (head, relation, tail)

# ---------- similarity helpers ----------
def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + eps))

def _norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v);  return v / (n + eps)

def _norm_rows(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (norms + eps)

# ---------- LangChain HF embeddings wrapper ----------
def _embed_one(text: str, sent_emb) -> np.ndarray:
    if hasattr(sent_emb, "_embed_text"):
        vec = sent_emb._embed_text(text)
    else:
        vec = sent_emb.embed_query(text)
    return np.asarray(vec, dtype=np.float32)

def _embed_many_texts(texts: Sequence[str], sent_emb) -> np.ndarray:
    if hasattr(sent_emb, "embed_documents"):
        vecs = sent_emb.embed_documents(list(texts))
        V = np.asarray(vecs, dtype=np.float32)
    else:
        V = np.stack([_embed_one(t, sent_emb) for t in texts], axis=0).astype(np.float32)
    return V

# ---------- formatting ----------
def triple_to_sentence(t: Triple) -> str:
    h, r, u = t
    return f"{h} {r} {u}"

# ---------- embedding entry points ----------
def embed_entities(entities: Sequence[str], sent_emb) -> np.ndarray:
    return _embed_many_texts(entities, sent_emb)

def embed_relations(relations: Sequence[str], sent_emb) -> np.ndarray:
    return _embed_many_texts(relations, sent_emb)

def embed_triples_as_sentences(triples: List[Triple], sent_emb) -> np.ndarray:
    texts = [triple_to_sentence(t) for t in triples]
    return _embed_many_texts(texts, sent_emb)

def ensure_list_of_triples(triples):
    if isinstance(triples, set):
        triples = list(triples)
    elif not isinstance(triples, list):
        raise TypeError(f"Expected list or set, got {type(triples)}")
    triples = [list(t) if not isinstance(t, list) else t for t in triples]
    return triples

# ---------- medoid helpers ----------
def _choose_medoid_exact(cur_vecs: List[np.ndarray]) -> int:
    if len(cur_vecs) == 1:
        return 0
    M = np.stack(cur_vecs, axis=0)               # (k, d)
    G = M @ M.T                                   # (k, k) cosine Gram
    avg = G.mean(axis=1)
    return int(np.argmax(avg))

def _choose_medoid_approx(cur_vecs: List[np.ndarray]) -> int:
    if len(cur_vecs) == 1:
        return 0
    M = np.stack(cur_vecs, axis=0)
    c = M.mean(axis=0)
    c /= (np.linalg.norm(c) + 1e-12)
    sims = M @ c
    return int(np.argmax(sims))

# ---------- segmenter with prototype + EMA + hysteresis ----------
tau_default = 0.58  # slightly lower for long-form; use hysteresis to stabilize

def segment_by_prototype_sim(
    triples: List[Triple],
    triple_vecs: np.ndarray,
    *,
    # similarity / thresholds
    tau_leave: float = tau_default,          # cut if sim < tau_leave (after patience)
    tau_enter: Optional[float] = None,       # optional: require sim >= tau_enter to "solidify" a chunk start
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: Optional[float] = 0.0,
    # structural bonus
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
    # prototype behavior
    prototype: str = "medoid",               # "centroid" | "medoid" | "medoid_approx" | "ema" | "hybrid"
    ema_beta: float = 0.85,                  # used if prototype == "ema" or "hybrid"
    # entity overlap hook: returns value in [0,1] for (cur_triples, next_triple)
    entity_overlap: Optional[Callable[[List[Triple], Triple], float]] = None,
    entity_bonus_lambda: float = 0.08,       # weight for entity overlap bonus
) -> List[List[Triple]]:
    """
    Segment an ordered triple list using cosine similarity against a chunk prototype.
    Supports medoid/centroid/EMA and hysteresis thresholds.
    """
    if not triples:
        return []
    if len(triples) != triple_vecs.shape[0]:
        raise ValueError("vecs and triples length mismatch")

    triples = ensure_list_of_triples(triples)

    # Normalize all embeddings once
    V = triple_vecs.astype(np.float32, copy=False)
    V = _norm_rows(V)

    chunks: List[List[Triple]] = []
    cur_triples: List[Triple] = [triples[0]]
    cur_vecs: List[np.ndarray] = [V[0].copy()]
    sum_vec = V[0].copy()                  # for centroid & approx
    ema_vec = V[0].copy()                  # for EMA/hybrid
    bad_streak = 0
    started = True                         # hysteresis: we already started first chunk

    def _proto_vec() -> np.ndarray:
        nonlocal sum_vec, cur_vecs, prototype, ema_vec
        if prototype == "centroid":
            c = sum_vec / (np.linalg.norm(sum_vec) + 1e-12)
            return c
        elif prototype == "medoid":
            mi = _choose_medoid_exact(cur_vecs)
            return cur_vecs[mi]
        elif prototype == "medoid_approx":
            mi = _choose_medoid_approx(cur_vecs)
            return cur_vecs[mi]
        elif prototype == "ema":
            v = ema_vec / (np.linalg.norm(ema_vec) + 1e-12)
            return v
        elif prototype == "hybrid":
            # Blend EMA with centroid for stability + coverage
            c = sum_vec / (np.linalg.norm(sum_vec) + 1e-12)
            e = ema_vec / (np.linalg.norm(ema_vec) + 1e-12)
            h = 0.7 * e + 0.3 * c
            return h / (np.linalg.norm(h) + 1e-12)
        else:
            raise ValueError(f"Unknown prototype: {prototype}")

    for i in range(1, len(triples)):
        # update EMA with *previous* item to avoid peeking at V[i]
        ema_vec = ema_beta * ema_vec + (1.0 - ema_beta) * V[i-1]

        p = _proto_vec()
        sim = float(p @ V[i])

        if relu_floor is not None:
            sim = max(relu_floor, sim)

        if bonus_tail_head:
            prev_h, prev_r, prev_t = cur_triples[-1]
            h, r, u = triples[i]
            if isinstance(prev_t, str) and isinstance(h, str) and prev_t.casefold() == h.casefold():
                sim = min(1.0, sim + tail_head_bonus)

        if entity_overlap is not None:
            # user callback should return [0,1]; we add a small bonus
            try:
                ov = float(entity_overlap(cur_triples, triples[i]))
                ov = max(0.0, min(1.0, ov))
                sim = min(1.0, sim + entity_bonus_lambda * ov)
            except Exception:
                pass

        # hysteresis enter guard (optional)
        if tau_enter is not None and not started:
            if sim >= tau_enter:
                started = True
            # Even if not yet started, we still accumulate into the first chunk
            # to let prototype stabilize before allowing cuts.

        # patience logic with tau_leave
        if sim < tau_leave:
            bad_streak += 1
        else:
            bad_streak = 0

        if started and bad_streak > patience and len(cur_triples) >= min_chunk_len:
            # cut
            chunks.append(cur_triples)
            cur_triples = [triples[i]]
            cur_vecs = [V[i].copy()]
            sum_vec = V[i].copy()
            ema_vec = V[i].copy()
            bad_streak = 0
            started = tau_enter is None  # if we require enter, next chunk needs to "enter" again
        else:
            # accumulate
            cur_triples.append(triples[i])
            cur_vecs.append(V[i].copy())
            sum_vec += V[i]

    if cur_triples:
        chunks.append(cur_triples)
    return chunks

# ---- decoding helpers (your original) ----
def decode_subchunk(question, codebook_main, fmt='words'):
    edges = codebook_main["edge_matrix"]
    idxs = list(question)

    def get_edge(i):
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
        decoded = [[h, r, t] for (h, r, t) in (get_edge(i) for i in idxs)]
    else:
        raise ValueError("fmt must be 'words', 'embeddings' or 'edges'.")
    return decoded

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

# ---- boundary test using the new segmenter ----
def should_merge_boundary(
    last_encoded: List[int],
    next_encoded: List[int],
    segmenter: Callable[..., List[List[Triple]]] = segment_by_prototype_sim,
    codebook_main = None,
    *,
    # pass-through knobs:
    tau_leave: float = tau_default,
    tau_enter: Optional[float] = 0.68,
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: Optional[float] = 0.0,
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
    prototype: str = "hybrid",
    ema_beta: float = 0.85,
    entity_overlap: Optional[Callable[[List[Triple], Triple], float]] = None,
    entity_bonus_lambda: float = 0.08,
    sent_emb: HuggingFaceEmbeddings = None
) -> bool:
    """
    Returns True iff the segmenter would NOT place a cut between the two subchunks.
    """
    last_triples = decode_subchunk(last_encoded, codebook_main, 'words')
    next_triples = decode_subchunk(next_encoded, codebook_main, 'words')
    if not last_triples or not next_triples:
        return False

    all_triples = last_triples + next_triples
    vecs = embed_triples_as_sentences(all_triples, sent_emb)

    chunks = segmenter(
        all_triples,
        vecs,
        tau_leave=tau_leave,
        tau_enter=tau_enter,
        min_chunk_len=min_chunk_len,
        patience=patience,
        relu_floor=relu_floor,
        bonus_tail_head=bonus_tail_head,
        tail_head_bonus=tail_head_bonus,
        prototype=prototype,
        ema_beta=ema_beta,
        entity_overlap=entity_overlap,
        entity_bonus_lambda=entity_bonus_lambda,
    )

    if len(chunks) == 1:
        return True

    first_len = _safe_len(chunks[0]) if chunks else 0
    return first_len != len(last_triples)

# ---- main routine: pass over chunk list and merge neighbors when boundary is weak ----
def merge_chunks_by_boundary(
    chunks: List[List[List[int]]],  # [[[int,...], ...], ...]
    segmenter: Callable[..., List[List[Triple]]] = segment_by_prototype_sim,
    codebook_main = None,
    *,
    tau_leave: float = tau_default,
    tau_enter: Optional[float] = 0.68,
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: Optional[float] = 0.0,
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
    prototype: str = "hybrid",
    ema_beta: float = 0.85,
    entity_overlap: Optional[Callable[[List[Triple], Triple], float]] = None,
    entity_bonus_lambda: float = 0.08,
    sent_emb: HuggingFaceEmbeddings = None
) -> List[List[List[int]]]:
    """
    Walks boundaries and merges chunk i with i+1 if the last subchunk of i and first subchunk
    of i+1 should be together under the given segmenter settings.
    """
    if not chunks:
        return []

    merged: List[List[List[int]]] = []
    cur = chunks[0]

    for i in range(len(chunks) - 1):
        left = cur
        right = chunks[i + 1]
        if not left and not right:
            continue
        if not left:
            cur = right
            continue
        if not right:
            continue

        last_left_encoded = left[-1]
        first_right_encoded = right[0]

        merge = should_merge_boundary(
            last_left_encoded,
            first_right_encoded,
            segmenter=segmenter,
            codebook_main=codebook_main,
            tau_leave=tau_leave,
            tau_enter=tau_enter,
            min_chunk_len=min_chunk_len,
            patience=patience,
            relu_floor=relu_floor,
            bonus_tail_head=bonus_tail_head,
            tail_head_bonus=tail_head_bonus,
            prototype=prototype,
            ema_beta=ema_beta,
            entity_overlap=entity_overlap,
            entity_bonus_lambda=entity_bonus_lambda,
            sent_emb=sent_emb
        )

        if merge:
            left[-1] = last_left_encoded + first_right_encoded
            cur = right[1:]
            if not cur:
                cur = left
            else:
                merged.append(left)
            print(f"merging boundary for chunk{i} and chunk{i+1}")
        else:
            merged.append(left)
            cur = right

    merged.append(cur)
    return merged

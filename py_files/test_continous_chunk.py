from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Tuple, Callable, Optional, Sequence
import numpy as np

Triple = Tuple[str, str, str]  # (head, relation, tail)

# ---------- similarity helpers ----------
def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + eps))

def _norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

# ---------- LangChain HF embeddings wrapper ----------
def _embed_one(text: str, sent_emb) -> np.ndarray:
    """
    Wraps your HuggingFaceEmbeddings to return a single np.ndarray vector.
    Prefers the method you showed (._embed_text), falls back to public API.
    """
    if hasattr(sent_emb, "_embed_text"):
        vec = sent_emb._embed_text(text)
    else:
        # some versions expose embed_query for single text
        vec = sent_emb.embed_query(text)
    return np.asarray(vec, dtype=np.float32)

def _embed_many_texts(texts: Sequence[str], sent_emb) -> np.ndarray:
    """
    Efficient batch embedding if available; otherwise loops.
    Returns (N, D) float32 array.
    """
    if hasattr(sent_emb, "embed_documents"):
        vecs = sent_emb.embed_documents(list(texts))
        V = np.asarray(vecs, dtype=np.float32)
    else:
        V = np.stack([_embed_one(t, sent_emb) for t in texts], axis=0).astype(np.float32)
    return V

# ---------- formatting ----------
def triple_to_sentence(t: Triple) -> str:
    h, r, u = t
    # Simple readable format; you can switch to "[h] --r--> [u]" if you prefer.
    return f"{h} {r} {u}"

# ---------- embedding entry points ----------
def embed_entities(entities: Sequence[str], sent_emb) -> np.ndarray:
    """(N, D) embeddings for single entity strings."""
    return _embed_many_texts(entities, sent_emb)

def embed_relations(relations: Sequence[str], sent_emb) -> np.ndarray:
    """(N, D) embeddings for single relation strings."""
    return _embed_many_texts(relations, sent_emb)

def embed_triples_as_sentences(triples: List[Triple], sent_emb) -> np.ndarray:
    """(N, D) embeddings for whole triples as sentences."""
    texts = [triple_to_sentence(t) for t in triples]
    return _embed_many_texts(texts, sent_emb)

def ensure_list_of_triples(triples):
    """
    Ensures the input is a list of triples (list of tuples or lists).
    Converts set → list if necessary.
    """
    if isinstance(triples, set):
        triples = list(triples)
    elif not isinstance(triples, list):
        raise TypeError(f"Expected list or set, got {type(triples)}")

    # Ensure each triple is also a list (not tuple or other type)
    triples = [list(t) if not isinstance(t, list) else t for t in triples]
    return triples

# ---------- segmenter ----------
def segment_by_centroid_sim(
    triples: List[Triple],
    triple_vecs: np.ndarray,
    tau: float = 0.58,            # similarity threshold to stay in chunk
    min_chunk_len: int = 1,
    patience: int = 0,            # consecutive below-threshold sims before cutting
    relu_floor: float = 0.0,      # clamp negative sims up to 0 if desired
    bonus_tail_head: bool = True, # small structural continuity bonus
    tail_head_bonus: float = 0.05,
) -> List[List[Triple]]:
    """
    Segment an ordered triple list into chunks using cosine similarity of
    triple-level sentence embeddings against the *current chunk centroid*.
    """
    if not triples:
        return []
    
    triples = ensure_list_of_triples(triples)
    
    assert len(triples) == triple_vecs.shape[0], "vecs and triples length mismatch"

    # L2-normalize once (BGE expects cosine)
    V = triple_vecs.astype(np.float32)
    V = np.apply_along_axis(_norm, 1, V)

    chunks: List[List[Triple]] = []
    cur: List[Triple] = [triples[0]]
    cur_vecs = [V[0]]
    bad_streak = 0

    for i in range(1, len(triples)):
        centroid = _norm(np.mean(np.stack(cur_vecs, axis=0), axis=0))
        sim = _cos(centroid, V[i])
        if relu_floor is not None:
            sim = max(relu_floor, sim)

        if bonus_tail_head:
            # tiny nudge if path continuity: tail(prev) == head(curr) (casefolded)
            prev_h, prev_r, prev_t = cur[-1]
            h, r, u = triples[i]
            if prev_t.casefold() == h.casefold():
                sim += tail_head_bonus

        if sim < tau:
            bad_streak += 1
        else:
            bad_streak = 0

        if bad_streak > patience and len(cur) >= min_chunk_len:
            chunks.append(cur)
            cur = [triples[i]]
            cur_vecs = [V[i]]
            bad_streak = 0
        else:
            cur.append(triples[i])
            cur_vecs.append(V[i])

    if cur:
        chunks.append(cur)
    return chunks


# ---- boundary test using default segmenter ----
def decode_subchunk(question, codebook_main, fmt='words'):

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


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def should_merge_boundary(
    last_encoded: List[int],
    next_encoded: List[int],
    segment_by_centroid_sim: Callable[..., List[List[Triple]]],
    *,
    tau: float = 0.58,
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: float = 0.0,
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
) -> bool:
    """
    Returns True iff the segmenter would *not* place a cut between the two
    subchunks (i.e., it's safe to merge the adjacent parent chunks).
    """
    # Decode both sides into triples
    last_triples = decode_subchunk(last_encoded,'edges') 
    next_triples = decode_subchunk(next_encoded,'edges')
    if not last_triples or not next_triples:
        # If either side has no triples, be conservative: don't merge.
        return False

    # Concatenate in boundary order
    all_triples = last_triples + next_triples
    vecs = decode_subchunk(all_triples)

    # Run segmenter over the concatenation
    chunks = segment_by_centroid_sim(
        all_triples,
        vecs,
        tau=tau,
        min_chunk_len=min_chunk_len,
        patience=patience,
        relu_floor=relu_floor,
        bonus_tail_head=bonus_tail_head,
        tail_head_bonus=tail_head_bonus,
    )

    # If there's exactly 1 chunk, the boundary isn't justified → merge.
    if len(chunks) == 1:
        return True

    # If there are 2+ chunks, check if the first cut lands exactly at the boundary.
    # Simple heuristic: if the first chunk length equals len(last_triples),
    # the model wants a cut exactly where your boundary is → don't merge.
    first_len = _safe_len(chunks[0]) if chunks else 0
    return first_len != len(last_triples)

# ---- main routine: pass over chunk list and merge neighbors when boundary is weak ----
def merge_chunks_by_boundary(
    chunks: List[List[List[int]]],  # [[[int,...], ...], ...]
    segment_by_centroid_sim: Callable[..., List[List[Triple]]],
    *,
    tau: float = 0.58,
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: float = 0.0,
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
) -> List[List[List[int]]]:
    """
    Walks boundaries between chunks and merges chunk i with i+1 if the
    last subchunk of i and first subchunk of i+1 *should* be together,
    as judged by segment_by_centroid_sim on decoded+embedded triples.
    """
    if not chunks:
        return []

    merged: List[List[List[int]]] = []
    cur = chunks[0]

    for i in range(len(chunks) - 1):
        left = cur
        right = chunks[i + 1]
        if not left or not right:
            # If either side has no subchunks, do not merge
            merged.append(left)
            cur = right
            continue

        last_left_encoded = left[-1]
        first_right_encoded = right[0]

        merge = should_merge_boundary(
            last_left_encoded,
            first_right_encoded,
            segment_by_centroid_sim,
            tau=tau,
            min_chunk_len=min_chunk_len,
            patience=patience,
            relu_floor=relu_floor,
            bonus_tail_head=bonus_tail_head,
            tail_head_bonus=tail_head_bonus,
        )

        if merge:
            # Merge entire chunks: concatenate subchunk lists
            cur = left + right
        else:
            merged.append(left)
            cur = right

    # push the last carried chunk
    merged.append(cur)
    return merged

# ---------- example wiring ----------
if __name__ == "__main__":
    # Your provided embedding object:
    # from langchain_huggingface import HuggingFaceEmbeddings
    sent_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    triples = [
        ("Basal cell carcinoma","subclass of","skin cancer"),
        ("skin cancer","has part","lesion"),
        ("UV radiation","causes","skin cancer"),
        ("Melanocyte","produces","melanin"),
        ("melanin","located in","epidermis"),
        ("tanning beds","emit","UV radiation"),
    ]

    # (1) Embed whole triples as sentences
    T_vecs = embed_triples_as_sentences(triples, sent_emb)

    # (2) Segment
    chunk1 = segment_by_centroid_sim(
        triples, T_vecs,
        tau=0.7,           # tighter -> more chunks; looser -> fewer chunks
        patience=0,
        bonus_tail_head=True,
        tail_head_bonus=0.05
    )

    for i, ch in enumerate(chunk1, 1):
        print(f"Chunk {i}: {ch}")

    triples2 = [
        ("UV radiation","causes","cancer"),
    ]

    T_vecs2 = embed_triples_as_sentences(triples2, sent_emb)

    chunk2 = segment_by_centroid_sim(
        triples2, T_vecs2,
        tau=0.7,           # tighter -> more chunks; looser -> fewer chunks
        patience=0,
        bonus_tail_head=True,
        tail_head_bonus=0.05
    )

    new_chunks = merge_chunks_by_boundary([chunk1,chunk2])

    print(f'new chunk {new_chunks}')

    # Optional: also embed single entities / relations if you want them separately
    # E_vecs = embed_entities(["Basal cell carcinoma", "skin cancer", "melanin"], sent_emb)
    # R_vecs = embed_relations(["subclass of", "has part", "causes"], sent_emb)



# python test_continous_chunk.py
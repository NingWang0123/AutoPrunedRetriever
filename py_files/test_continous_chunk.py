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

# # ---------- example wiring ----------
# if __name__ == "__main__":
#     # Your provided embedding object:
#     # from langchain_huggingface import HuggingFaceEmbeddings
#     sent_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

#     triples = [
#         ("Basal cell carcinoma","subclass of","skin cancer"),
#         ("skin cancer","has part","lesion"),
#         ("UV radiation","causes","skin cancer"),
#         ("Melanocyte","produces","melanin"),
#         ("melanin","located in","epidermis"),
#         ("tanning beds","emit","UV radiation"),
#     ]

#     # (1) Embed whole triples as sentences
#     T_vecs = embed_triples_as_sentences(triples, sent_emb)

#     # (2) Segment
#     chunks = segment_by_centroid_sim(
#         triples, T_vecs,
#         tau=0.7,           # tighter -> more chunks; looser -> fewer chunks
#         patience=0,
#         bonus_tail_head=True,
#         tail_head_bonus=0.05
#     )

#     for i, ch in enumerate(chunks, 1):
#         print(f"Chunk {i}: {ch}")

    # Optional: also embed single entities / relations if you want them separately
    # E_vecs = embed_entities(["Basal cell carcinoma", "skin cancer", "melanin"], sent_emb)
    # R_vecs = embed_relations(["subclass of", "has part", "causes"], sent_emb)



# python test_continous_chunk.py
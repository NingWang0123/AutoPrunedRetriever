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

tau_default = 0.6
# tau_default = 0.7

def segment_by_centroid_sim(
    triples: List[Triple],
    triple_vecs: np.ndarray,
    tau: float = tau_default,            # similarity threshold to stay in chunk
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
    codebook_main = None,
    *,
    tau: float = tau_default,
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: float = 0.0,
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
    sent_emb:HuggingFaceEmbeddings = None
) -> bool:
    """
    Returns True iff the segmenter would *not* place a cut between the two
    subchunks (i.e., it's safe to merge the adjacent parent chunks).
    """
    # Decode both sides into triples
    last_triples = decode_subchunk(last_encoded,codebook_main,'words') 
    next_triples = decode_subchunk(next_encoded,codebook_main,'words')
    if not last_triples or not next_triples:
        # If either side has no triples, be conservative: don't merge.
        return False

    # Concatenate in boundary order
    all_triples = last_triples + next_triples

    # print('all_triples',all_triples)

    vecs = embed_triples_as_sentences(all_triples,sent_emb) 

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

    # print('new chunks is',chunks)

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
    segment_by_centroid_sim: Callable[..., List[List[Triple]]] = segment_by_centroid_sim,
    codebook_main = None,
    *,
    tau: float = tau_default, 
    min_chunk_len: int = 1,
    patience: int = 0,
    relu_floor: float = 0.0,
    bonus_tail_head: bool = True,
    tail_head_bonus: float = 0.05,
    sent_emb:HuggingFaceEmbeddings = None
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

        if not left and not right:
            continue

        if not left:
            cur = right
            continue

        if not right:
            continue


        last_left_encoded = left[-1]
        first_right_encoded = right[0]

        # print('left',last_left_encoded)
        # print('right',first_right_encoded)

        merge = should_merge_boundary(
            last_left_encoded,
            first_right_encoded,
            segment_by_centroid_sim,
            codebook_main = codebook_main,
            tau=tau,
            min_chunk_len=min_chunk_len,
            patience=patience,
            relu_floor=relu_floor,
            bonus_tail_head=bonus_tail_head,
            tail_head_bonus=tail_head_bonus,
            sent_emb=sent_emb
        )

        if merge:
            # print('merging success')
            # # Merge entire chunks: concatenate subchunk lists
            # print('left',last_left_encoded)
            # print('right',first_right_encoded)
            left[-1] = last_left_encoded+first_right_encoded
            cur = right[1:]
            # if merging make the right empty, we still use the cur as the next cur
            if not cur:
                cur = left
            else:
                merged.append(left)

            print(f'merging boundary for chunk{i} and chunk{i+1}')
        else:
            merged.append(left)
            cur = right

    # push the last carried chunk
    merged.append(cur)
    return merged

# ---------- example wiring ----------
# if __name__ == "__main__":
#     # Your provided embedding object:
#     # from langchain_huggingface import HuggingFaceEmbeddings
#     sent_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


#     # -----------------------------------------------------------------------------
#     # 0) Build a minimal codebook_main with entities, relations, and edge_matrix
#     # -----------------------------------------------------------------------------
#     E = [
#         "Basal cell carcinoma",  # 0
#         "skin cancer",           # 1
#         "lesion",                # 2
#         "UV radiation",          # 3
#         "Melanocyte",            # 4
#         "melanin",               # 5
#         "epidermis",             # 6
#         "tanning beds",          # 7
#         "cancer",                # 8
#     ]
#     R = [
#         "subclass of",  # 0
#         "has part",     # 1
#         "causes",       # 2
#         "produces",     # 3
#         "located in",   # 4
#         "emit",         # 5
#     ]
#     EIDX = {s: i for i, s in enumerate(E)}
#     RIDX = {s: i for i, s in enumerate(R)}

#     EDGE_MATRIX = [
#         [EIDX["Basal cell carcinoma"], RIDX["subclass of"], EIDX["skin cancer"]],
#         [EIDX["skin cancer"],          RIDX["has part"],    EIDX["lesion"]],
#         [EIDX["UV radiation"],         RIDX["causes"],      EIDX["skin cancer"]],
#         [EIDX["Melanocyte"],           RIDX["produces"],    EIDX["melanin"]],
#         [EIDX["melanin"],              RIDX["located in"],  EIDX["epidermis"]],
#         [EIDX["tanning beds"],         RIDX["emit"],        EIDX["UV radiation"]],
#         [EIDX["UV radiation"],         RIDX["causes"],      EIDX["cancer"]],
#     ]

#     codebook_main = {
#         "e": E,
#         "r": R,
#         "edge_matrix": EDGE_MATRIX,
#     }

#     # Make codebook globally visible for the adapter below
#     CODEBOOK_MAIN = codebook_main

#     # -----------------------------------------------------------------------------
#     # 1) OPTIONAL: add entity & relation embeddings to support fmt='embeddings'
#     # -----------------------------------------------------------------------------
#     try:
#         sent_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
#         codebook_main["e_embeddings"] = embed_entities(codebook_main["e"], sent_emb)
#         codebook_main["r_embeddings"] = embed_relations(codebook_main["r"], sent_emb)
#     except Exception as e:
#         print(f"[warn] Could not compute e/r embeddings (that's OK unless you use fmt='embeddings'): {e}")

#     # -----------------------------------------------------------------------------
#     # 2) Compatibility adapter (no body edits): make your existing calls work
#     # -----------------------------------------------------------------------------
#     # Your should_merge_boundary currently calls:
#     #   decode_subchunk(last_encoded, 'edges')
#     #   decode_subchunk(next_encoded, 'edges')
#     #   decode_subchunk(all_triples)   # etc.
#     # but the defined signature is decode_subchunk(question, codebook_main, fmt='words').
#     # We add a tiny adapter that preserves your name but routes calls correctly.
#     try:
#         _orig_decode_subchunk = decode_subchunk  # keep original

#         def decode_subchunk(*args, **kwargs):
#             """
#             Compatibility adapter:
#             - If called like decode_subchunk(indices, 'edges'), assume global CODEBOOK_MAIN.
#             - If called like decode_subchunk(indices, CODEBOOK_MAIN, fmt='words'), pass through.
#             - If called with actual triple rows (list[list[h,r,t]]) and no fmt provided,
#             treat them as *indices* only if they are ints; otherwise, if they look like triples,
#             just return them (segmenter expects words/edges already).
#             """
#             # Case A: the "correct" signature (indices, codebook_main, fmt=...)
#             if len(args) >= 2 and isinstance(args[1], dict) and "edge_matrix" in args[1]:
#                 return _orig_decode_subchunk(*args, **kwargs)

#             # Case B: the "in-code" usage you have (indices, 'edges'|'words')
#             if len(args) == 2 and isinstance(args[0], (list, tuple)) and isinstance(args[1], str):
#                 indices, fmt = args
#                 return _orig_decode_subchunk(indices, CODEBOOK_MAIN, fmt=fmt)

#             # Case C: decode_subchunk(all_triples) — if they're already triplets, return them
#             if len(args) == 1 and isinstance(args[0], (list, tuple)) and args[0]:
#                 first = args[0][0]
#                 # If they look like [h, r, t] (either ints or strings), just return as-is
#                 if isinstance(first, (list, tuple)) and len(first) == 3:
#                     return args[0]

#             # Fallback to original
#             return _orig_decode_subchunk(*args, **kwargs)

#     except NameError:
#         # If decode_subchunk wasn't defined yet for some reason, skip
#         pass

#     # -----------------------------------------------------------------------------
#     # 3) Prepare triples and sentence embeddings for the segmenter
#     # -----------------------------------------------------------------------------
#     triples_all = [
#         ("Basal cell carcinoma","subclass of","skin cancer"),
#         ("skin cancer","has part","lesion"),
#         ("UV radiation","causes","skin cancer"),
#         ("Melanocyte","produces","melanin"),
#         ("melanin","located in","epidermis"),
#         ("tanning beds","emit","UV radiation"),
#     ]
#     T_vecs = embed_triples_as_sentences(triples_all, sent_emb)

#     # Show segmentation behavior on a single contiguous list
#     print("\n=== Segment result on the full triple list ===")
#     chunked = segment_by_centroid_sim(
#         triples_all, T_vecs,
#         tau=0.70,           # tighter threshold -> more cuts
#         patience=0,
#         relu_floor=0.0,
#         bonus_tail_head=True,
#         tail_head_bonus=0.05,
#     )
#     for i, ch in enumerate(chunked, 1):
#         print(f"Chunk {i}: {ch}")

#     # -----------------------------------------------------------------------------
#     # 4) Build chunked edges (indices) like your real pipeline, then test merging
#     # -----------------------------------------------------------------------------
#     # We'll create two "chunks", each holding subchunks = lists of edge indices
#     # chunk_left contains edges [0,1] and [2]
#     # chunk_right contains edges [3] and [4,5] (two subchunks)
#     chunk_left  = [[2],[0]]       # two subchunks on the left side
#     chunk_right = [[1]]       # two subchunks on the right side

#     chunks_indices = [chunk_left, chunk_right,[[3]],[[4]]]  # [[[...],[...]], [[...],[...]]]
#     print("\n=== Raw chunks (by edge indices) ===")
#     print(chunks_indices)

#     # Sanity check: decode the last subchunk of left and first subchunk of right (words)
#     print("\nDecoded last-left (words):")
#     print(decode_subchunk(chunk_left[-1], codebook_main, fmt='words'))
#     print("\nDecoded first-right (words):")
#     print(decode_subchunk(chunk_right[0], codebook_main, fmt='words'))

#     # -----------------------------------------------------------------------------
#     # 5) Run the boundary-based merge pass
#     # -----------------------------------------------------------------------------
#     print("\n=== Merging pass over chunk boundaries ===")
#     merged_chunks = merge_chunks_by_boundary(
#         chunks_indices,
#         segment_by_centroid_sim=segment_by_centroid_sim,
#         tau=0.70,
#         min_chunk_len=1,
#         patience=0,
#         relu_floor=0.0,
#         bonus_tail_head=True,
#         tail_head_bonus=0.05,
#         sent_emb = sent_emb
#     )
#     print("Merged result (by edge indices):")
#     print(merged_chunks)


# python test_continous_chunk.py
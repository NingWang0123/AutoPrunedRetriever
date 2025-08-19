from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from graph_generator.graphparsers import RelationshipGraphParser
from linearization_utils import *
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Basic: Top-k similarity search (with scores) ===
def similarity_search_graph_docs(
    user_question: str,
    parser: RelationshipGraphParser,
    vectordb: FAISS,
    k: int = 5,
):
    """
    1) Parse the new question into a graph + linearized text (same distribution as stored docs)
    2) Perform similarity search in the vector database, return [(Document, score), ...]
    """
    # Parse the new question into graph & linearized text
    G_new, rels_new = parser.question_to_graph(user_question)
    query_text = build_relationship_text(user_question, G_new, rels_new)
    if query_text:
        # Search with scores
        results = vectordb.similarity_search_with_score(query_text, k=k)
    else:
        print("Question graph can't be extracted, cannot search")
        results = []
    return query_text, results


# === Optional: MMR search (for more diversity, avoids redundant hits) ===
def mmr_search_graph_docs(
    user_question: str,
    parser: RelationshipGraphParser,
    vectordb: FAISS,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
):
    """
    Perform Max Marginal Relevance (MMR) search to improve diversity
    and avoid retrieving highly redundant documents.
    Note: MMR usually returns only documents (no scores). If scores are needed,
    you can re-score the returned documents separately.
    """
    G_new, rels_new = parser.question_to_graph(user_question)
    query_text = build_relationship_text(user_question, G_new, rels_new)
    if query_text:

        docs = vectordb.max_marginal_relevance_search(
            query=query_text, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
    else:
        print("Question graph can't be extracted, cannot search")
        docs = []
    return query_text, docs


# === Optional: Threshold filtering (keep only hits above similarity threshold) ===
def similarity_search_with_threshold(
    user_question: str,
    parser: RelationshipGraphParser,
    vectordb: FAISS,
    k: int = 10,
    score_threshold: float = None,
):
    """
    If you want to filter out low similarity hits, set score_threshold.
    Note: FAISS score meaning depends on implementation
          (sometimes smaller = more similar (L2 distance), sometimes larger = more similar).
          You should print once and check before deciding the threshold strategy.
    """
    query_text, results = similarity_search_graph_docs(user_question, parser, vectordb, k=k)

    if score_threshold is None:
        return query_text, results

    # Assume smaller score = more similar (common for L2 distance).
    # If in your setup larger = better, then change to >= for filtering.
    filtered = [(d, s) for (d, s) in results if s <= score_threshold]
    return query_text, filtered



# === Helper: Nicely format and print retrieved hits ===
def pretty_print_hits(results: List[Tuple]):
    if not results:
        print("No results.")
        return

    def to_similarity(score: float) -> float:
        # Convert L2 distance into similarity (higher = more similar, range 0~1)
        return 1.0 / (1.0 + score)

    for i, r in enumerate(results, 1):
        if isinstance(r, tuple):  # (Document, score)
            doc, score = r
            sim = to_similarity(score)
            md = doc.metadata
            print(
                f"[{i}] sim={sim:.4f}  (dist={score:.4f})  "
                f"id={md.get('graph_id')}  q={md.get('question')}"
            )
        else:  # MMR returns only Document
            doc = r
            md = doc.metadata
            print(f"[{i}] id={md.get('graph_id')}  q={md.get('question')}")


if __name__ == "__main__":
    parser = RelationshipGraphParser()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_db = FAISS.load_local("graph_rag_faiss_index", emb, allow_dangerous_deserialization=True)

    user_q = "Is the Great Wall visible from low Earth orbit?"

    print("\n--- Similarity Search (Top-3) ---")
    qtext, hits = similarity_search_graph_docs(user_q, parser, faiss_db, k=3)
    pretty_print_hits(hits)

    print("\n--- MMR Search (Top-3, diverse) ---")
    qtext_mmr, mmr_hits = mmr_search_graph_docs(user_q, parser, faiss_db, k=3, fetch_k=20, lambda_mult=0.5)
    pretty_print_hits(mmr_hits)

    # Optional: apply threshold filtering
    # qtext_thr, hits_thr = similarity_search_with_threshold(user_q, parser, faiss_db, k=10, score_threshold=0.6)
    # pretty_print_hits(hits_thr)


from CompressRag_rl_v1 import CompressRag,WordAvgEmbeddings,decode_questions
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def self_decode_question(question, codebook_main, fmt='words'):
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
    e_item = next((s for s in codebook_main.keys() if "edge" in s.lower()), None)
    
    edges = codebook_main[e_item]

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

def self_decode_questions(questions, questions_source_codebook, fmt='words'):

    """
    questions_source_codebook must be the codebook that contain the questions
    Decode a list of questions using decode_question.
    
    questions: list of list[int]
        Each inner list is a sequence of edge indices.
    """
    return [self_decode_question(q, questions_source_codebook, fmt=fmt) for q in questions]

class LLMRepeat:
    def __init__(self):
        self.q = None

    def take_questions(self,q):
        # assumes prompt_json contains 'questions_lst'
        # q = self.q
        # q_item = next((s for s in q.keys() if "question" in s.lower()), None)
        return 'dog sit on mat'

    

questions = ['Does dog sit on mat?','Does cat sit on mat?']
    
word_emb = WordAvgEmbeddings(model_path="/Users/wangning/desktop/gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")

sentence_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

cr = CompressRag(ini_meta_codebook={},sentence_emb = sentence_emb, word_emb= word_emb, include_thinkings=False,llm= LLMRepeat())

i = 0
for q in questions:
    print(f'q {i}')
    print(cr.run_work_flow(q))
    # print(cr.meta_codebook)
    i+=1
    
# python py_files/test_for_compressrag.py
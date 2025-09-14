from CompressRag_rl_v1 import CompressRag_rl,WordAvgEmbeddings,decode_questions, get_context
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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



class Phi4MiniReasoningLLM:
    def __init__(self, include_thinkings: bool = True,
                model_name: str = "microsoft/Phi-4-mini-reasoning",
                max_new_tokens: int = 256,
                temperature: float = 0.3,
                top_p: float = 0.95):
        self.include_thinkings = include_thinkings
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if not hasattr(self.tokenizer, "apply_chat_template"):
            self._use_chat_template = False
        else:
            self._use_chat_template = True


    def _build_prompt(self, final_merged_json, question, decode = False):
        if decode:
            q_txt, gk_txt, st_txt, ft_txt = get_context(final_merged_json)
            user_msg = ""

            system_msg = (
                "You are a precise QA agent that answers by expressing facts as short, "
                "plain English statements. Keep outputs concise and factual."
            )

            ctx_lines = [
                "<<<RETRIEVED_CONTEXT_START>>>",
                "The system searched for a related question in the database. Below are related question's graph triples and its prior answer as reference. You don't have to follow it completely, just use it as a reference.",
                "[RELATED QUESTION'S GRAPH TRIPLES]:",
                q_txt,
                f"[RELATED QUESTION'S ANSWER TRIPLES]: {gk_txt}",
            ]
            
            if st_txt.strip().lower() != "none.":
                ctx_lines.append(f"[RELATED THINKING“S TRIPLES]: {st_txt}")

            if ft_txt.strip().lower() != "none.":
                ctx_lines.append(f"[RELATED FACTS'S TRIPLES]: {ft_txt}")

            ctx_lines.append("<<<RETRIEVED_CONTEXT_END>>>")

            user_msg += "\n".join(ctx_lines) + "\n"

            user_msg += (
                f"[CURRENT QUESTION]: {question} \n"
                "[TASK]: You are a QA assistant for open-ended questions.\n"
                f"- Give a short, direct answer in 2–3 sentences."
                "- Do NOT restrict to yes/no.\n"
                "[FORMAT]: Write complete sentences (not a single word)."
                "Avoid starting with just 'Yes.' or 'No.'; if the question is yes/no-style, state the conclusion AND 1–2 short reasons.\n"
                "[ANSWER]: "
            )
        else:
            user_msg = ""
            SYSTEM_PROMPT = """
            ---Role---
            You are a helpful assistant responding to user queries.

            ---Goal---
            Generate direct and concise answers based strictly on the provided Knowledge Base.
            Respond in plain text without explanations or formatting.
            Maintain conversation continuity and use the same language as the query.
            If the answer is unknown, respond with "I don't know".

            ---Knowledge Base---
            [Graph Format]
            - e: list of entity strings; index i refers to e[i]
            - r: list of relation strings; index j refers to r[j]
            - edge_matrix: list of [head_e_idx, r_idx, tail_e_idx]
            - questions([[e,r,e], ...]): related questions triples in conversation history 
            - given knowledge([[e,r,e], ...]): prior answer triples according to questions
            - start thinking with(edges[i]): related questions' thinking process triples
            - facts(edges[i]):groups of edge indices of facts
            - facts([[e,r,e], ...]): related facts triples

            [Data]
            {context_data}
            """

            system_msg = SYSTEM_PROMPT.format(
            context_data=final_merged_json
        )

            user_msg += (
                f"\n---Current Question---\n{question}"
                "\n---Answer---\n"
            )
        print(system_msg)
        print(user_msg)
        return system_msg, user_msg


    @torch.no_grad()
    def _generate(self, system_msg: str, user_msg: str) -> str:
        if self._use_chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cut = text.rfind(user_msg)
        return text[cut + len(user_msg):].strip() if cut != -1 else text.strip()
    
    def strip_think(self, s: str) -> Tuple[str, List[str]]:
        """Remove <think>...</think> blocks; also handle dangling <think> at EOF."""
        if not s:
            return "", []
        s_lower = s.lower()
        thinks: List[str] = []
        spans: List[Tuple[int, int]] = []

        for m in re.finditer(r"<think>(.*?)</think>", s, flags=re.S | re.I):
            thinks.append(m.group(1).strip())
            spans.append((m.start(), m.end()))

        last_open = s_lower.rfind("<think>")
        if last_open != -1 and s_lower.find("</think>", last_open) == -1:
            content_start = last_open + len("<think>")
            dangling_text = s[content_start:].strip()
            if dangling_text:
                thinks.append(dangling_text)
            spans.append((last_open, len(s)))

        if spans:
            spans.sort()
            merged = []
            cur_s, cur_e = spans[0]
            for st, en in spans[1:]:
                if st <= cur_e:
                    cur_e = max(cur_e, en)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = st, en
            merged.append((cur_s, cur_e))
        else:
            merged = []

        parts = []
        prev = 0
        for st, en in merged:
            if prev < st:
                parts.append(s[prev:st])
            prev = en
        if prev < len(s):
            parts.append(s[prev:])

        clean = "".join(parts)
        clean = re.sub(r"(?:^|\n)\s*(Okay,|Let’s|Let's|Step by step|Thought:).*", "", clean, flags=re.I)
        return clean.strip(), thinks
        
    def take_questions(self, final_merged_json, question, *, max_regen: int = 3):
        def _clean_answer(s: str, limit=4):
            parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', s) if p.strip()]
            if not parts:
                return ""
            return " ".join(parts[:limit])

        last_thinks: List[str] = []
        ans_clean = ""

        for attempt in range(max_regen):
            sys_msg, usr_msg = self._build_prompt(final_merged_json, question, decode= False)
            out = self._generate(sys_msg, usr_msg)
            print(f"-------------RAW[{attempt+1}/{max_regen}]:", out)

            raw = out.strip()
            m = re.search(r"\[answers\](.*?)(?:\[thinkings\]|\Z)", raw, flags=re.S | re.I)
            ans_region = m.group(1).strip() if m else raw

            ans_no_think, thinks = self.strip_think(ans_region)
            last_thinks = thinks  

            ans_clean = _clean_answer(ans_no_think, 4).strip()
            if ans_clean:
                break

        if not ans_clean:
            ans_clean = "No answer."

        if self.include_thinkings:
            thinks_str = "\n\n".join(t.strip() for t in last_thinks if t.strip())
            return ans_clean, thinks_str
        else:
            return ans_clean


word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")
sentence_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

include_thinking = True
phi_llm = Phi4MiniReasoningLLM(
    include_thinkings=include_thinking,                 
    model_name="microsoft/Phi-4-mini-reasoning",
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.9
)

rag = CompressRag_rl(
    sentence_emb=sentence_emb,
    word_emb=word_emb,
    llm=phi_llm,
    combine_ents_rounds=1,        
    thinkings_choice='overlap',  
    answers_choice='unique'       
)

rag.top_k = 5
rag.top_m = 2
rag.question_batch_size = 2
rag.questions_db_batch_size = 16

questions = [
    "From which cell type does basal cell carcinoma arise?",
    "From which cell type does basal cell carcinoma arise?",
]
import json
import numpy as np

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()   # ndarray -> list
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)       # numpy int -> int
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)     # numpy float -> float
    raise TypeError(f"Type {type(obj)} not serializable")

i = 0
for q in questions:
    print(f'q {i}')
    result = rag.run_work_flow(q, facts_json_path=["context/novel copy.json", "context/medical copy.json"], warm_start="coarse")
    #print(result)

    #with open(f"meta_codebook_{i}.json", "w", encoding="utf-8") as f:
    #    json.dump(rag.meta_codebook, f, ensure_ascii=False, indent=2, default=to_serializable)
    i += 1





    

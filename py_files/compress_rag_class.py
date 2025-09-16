# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional, Any, Dict
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from CompressRag_rl_v1 import CompressRag_rl,WordAvgEmbeddings,decode_questions, get_context
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
from typing import List, Tuple


def _coerce_edges_to_text(e_list, r_list, edge_matrix) -> str:
    """
    Convert index-based triples to human-readable text if possible.
    Falls back gracefully when indices or lists are incomplete.
    """
    if not edge_matrix:
        return "None."
    out = []
    for trip in edge_matrix:
        try:
            h, rr, t = trip
            h_txt = e_list[h] if isinstance(h, int) and 0 <= h < len(e_list) else str(h)
            r_txt = r_list[rr] if isinstance(rr, int) and 0 <= rr < len(r_list) else str(rr)
            t_txt = e_list[t] if isinstance(t, int) and 0 <= t < len(e_list) else str(t)
            out.append(f"({h_txt}) -[{r_txt}]-> ({t_txt})")
        except Exception:
            out.append(str(trip))
    return "\n".join(out) if out else "None."


def get_context(final_merged_json: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Try to extract 4 text blocks for the decode=True prompt:
    q_txt: related questions graph triples
    gk_txt: prior answer (given knowledge) triples
    st_txt: related 'thinking' triples
    ft_txt: related facts triples

    Robust to missing/renamed keys — always returns strings.
    """
    e = final_merged_json.get("e", [])
    r = final_merged_json.get("r", [])
    # Common key variants we’ve seen in your notes/comments:
    questions = (final_merged_json.get("questions")
                 or final_merged_json.get("question_triples")
                 or final_merged_json.get("questions_triples"))
    given_knowledge = (final_merged_json.get("given knowledge")
                       or final_merged_json.get("given_knowledge")
                       or final_merged_json.get("answer_triples")
                       or final_merged_json.get("answers_triples"))
    thinking_edges = (final_merged_json.get("start thinking with")
                      or final_merged_json.get("start_thinking_with")
                      or final_merged_json.get("thinking_triples"))
    facts_edges = final_merged_json.get("facts")

    # If these are already triples, use directly; if they are edge indices into edge_matrix, resolve them.
    edge_matrix = final_merged_json.get("edge_matrix", [])
    def _resolve(maybe_edges):
        if maybe_edges is None:
            return "None."
        # If it's a list of int indices into edge_matrix:
        if maybe_edges and all(isinstance(x, int) for x in maybe_edges):
            edges = [edge_matrix[i] for i in maybe_edges if 0 <= i < len(edge_matrix)]
            return _coerce_edges_to_text(e, r, edges)
        # If it's list of triples already:
        if maybe_edges and all(isinstance(x, (list, tuple)) and len(x) == 3 for x in maybe_edges):
            return _coerce_edges_to_text(e, r, maybe_edges)
        # Fallback to string
        return str(maybe_edges)

    q_txt = _resolve(questions)
    gk_txt = _resolve(given_knowledge)
    st_txt = _resolve(thinking_edges)
    ft_txt = _resolve(facts_edges)
    return q_txt, gk_txt, st_txt, ft_txt


class Phi4MiniReasoningLLM:
    def __init__(self,
                 include_thinkings: bool = True,
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
        # Check chat template support
        self._use_chat_template = hasattr(self.tokenizer, "apply_chat_template")

        # Reasonable pad/eos fallback
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, final_merged_json: Dict[str, Any], question: str, decode: bool = False) -> Tuple[str, str]:
        """
        Build a strict, machine-parsable prompt so we can reliably parse the model’s output.
        Two modes:
          - decode=True: includes a structured 'retrieved context' block produced by get_context(...)
          - decode=False: includes the Knowledge Base JSON and enforces a 2-block output format
        """
        if decode:
            q_txt, gk_txt, st_txt, ft_txt = get_context(final_merged_json)
            user_msg = ""

            system_msg = (
                "You are a precise QA agent that answers by expressing facts as short, "
                "plain English statements. Keep outputs concise and factual.\n"
                "Respond in the EXACT two-field format:\n"
                "[answers]: <1–3 short sentences>\n"
                "[thinkings]: <1–3 brief bullets or sentences>\n"
                "Do not output anything else."
            )

            ctx_lines = [
                "<<<RETRIEVED_CONTEXT_START>>>",
                "The system searched for a related question in the database. Below are related question's graph triples and its prior answer as reference. Use it as a reference only.",
                "[RELATED QUESTION'S GRAPH TRIPLES]:",
                str(q_txt),
                f"[RELATED ANSWER TRIPLES]: {gk_txt}",
            ]

            # NOTE: fixed the curly quote here (was THINKING“S)
            if str(st_txt).strip().lower() != "none.":
                ctx_lines.append(f"[RELATED THINKINGS TRIPLES]: {st_txt}")

            if str(ft_txt).strip().lower() != "none.":
                ctx_lines.append(f"[RELATED FACTS TRIPLES]: {ft_txt}")

            ctx_lines.append("<<<RETRIEVED_CONTEXT_END>>>")

            user_msg += "\n".join(ctx_lines) + "\n"
            user_msg += (
                f"[CURRENT QUESTION]: {question}\n"
                "[TASK]: Answer in 1–3 short sentences; if yes/no-like, state conclusion AND 1–2 brief reasons.\n"
                "[FORMAT]:\n[answers]: ...\n[thinkings]: ...\n"
            )
        else:
            # Compact, strict format — this is the recommended default
            SYSTEM_PROMPT = (
                "You are a precise QA assistant. Use only the provided Knowledge Base.\n"
                "Respond in the EXACT format below and nothing else.\n\n"
                "[RESPONSE FORMAT]\n"
                "[answers]: <1–3 short sentences answering the question>\n"
                "[thinkings]: <1–3 brief bullets of reasoning or evidence>\n"
                "Do not add any extra sections or text outside these two bracketed fields."
            )
            system_msg = SYSTEM_PROMPT

            # Keep the KB compact; if it's huge, consider pre-trimming upstream.
            KB = f"{final_merged_json}"
            user_msg = (
                f"---Knowledge Base---\n{KB}\n"
                f"---Current Question---\n{question}\n\n"
                "Now produce exactly:\n"
                "[answers]: ...\n"
                "[thinkings]: ..."
            )

        # Uncomment for debugging:
        # print(system_msg)
        # print(user_msg)
        return system_msg, user_msg

    @torch.no_grad()
    def _generate(self, system_msg: str, user_msg: str) -> str:
        """
        Generate text and **return only newly generated tokens** (no echoed prompts),
        by slicing the output using the input token length.
        """
        if self._use_chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        else:
            prompt_text = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        input_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = outputs[0, input_len:]  # only newly generated tokens
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    def strip_think(self, s: str) -> Tuple[str, List[str]]:
        """
        Remove <think>...</think> blocks and return (clean_text, think_blocks).
        Also handles a dangling <think> without a closing tag at EOF.
        """
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

        merged = []
        if spans:
            spans.sort()
            cur_s, cur_e = spans[0]
            for st, en in spans[1:]:
                if st <= cur_e:
                    cur_e = max(cur_e, en)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = st, en
            merged.append((cur_s, cur_e))

        parts = []
        prev = 0
        for st, en in merged:
            if prev < st:
                parts.append(s[prev:st])
            prev = en
        if prev < len(s):
            parts.append(s[prev:])

        clean = "".join(parts)
        # Trim common "thinking prefaces" that sometimes leak
        clean = re.sub(r"(?:^|\n)\s*(Okay,|Let’s|Let's|Step by step|Thought:).*", "", clean, flags=re.I)
        return clean.strip(), thinks

    def take_questions(self, final_merged_json: Dict[str, Any], question: str, *, max_regen: int = 3):
        """
        Ask the model, enforce the format, and return (answer, thinkings?) depending on include_thinkings.
        - Retries up to max_regen times if we fail to extract a non-empty answer.
        """
        def _clean_answer(s: str, limit=4) -> str:
            parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', s) if p.strip()]
            return " ".join(parts[:limit]) if parts else ""

        last_thinks: List[str] = []
        ans_clean = ""
        print(f'final_merged_json: {final_merged_json}')

        for attempt in range(max_regen):
            sys_msg, usr_msg = self._build_prompt(final_merged_json, question, decode=False)

            raw = self._generate(sys_msg, usr_msg).strip()
            # print(f"-------------RAW[{attempt+1}/{max_regen}]: {raw}")

            # 1) Extract [answers]
            m_ans = re.search(r"\[answers\]\s*:\s*(.*?)(?=\n\s*\[thinkings\]\s*:|\Z)", raw, flags=re.S | re.I)
            ans_text = (m_ans.group(1).strip() if m_ans else raw).strip()

            # 2) Extract [thinkings]
            m_th = re.search(r"\[thinkings\]\s*:\s*(.*)\Z", raw, flags=re.S | re.I)
            think_text = (m_th.group(1).strip() if m_th else "").strip()

            print(f'ans_text: {ans_text}') 
            print(f'think_text: {think_text}') 
            # 3) Also capture any <think>...</think> spans anywhere
            _, think_spans = self.strip_think(raw)

            # Merge sources of thinking: prefer explicit block, then <think> fallback
            merged_thinks = []
            if think_text:
                merged_thinks.append(think_text)
            merged_thinks.extend(think_spans)

            # Clean the answer and keep the best attempt
            ans_no_think, _ = self.strip_think(ans_text)
            ans_clean = _clean_answer(ans_no_think, 4)
            last_thinks = [t for t in (s.strip() for s in merged_thinks) if t]

            if ans_clean:
                break

        if not ans_clean:
            ans_clean = "I don't know."

        if self.include_thinkings:
            thinks_str = "\n\n".join(last_thinks) if last_thinks else ""
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
    # "From which cell type does basal cell carcinoma arise?",
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

results = []

i = 0
for q in questions:
    print(f'q {i}')
    result = rag.run_work_flow(q, facts_json_path=["/home/ra_daniel/bilby/relational_graph_llm/py_files/medical_sub.json"], warm_start="coarse")
    print(result)
    results.append(result)
    with open(f"meta_codebook_{i}.json", "w", encoding="utf-8") as f:
       json.dump(rag.meta_codebook, f, ensure_ascii=False, indent=2, default=to_serializable)
    i += 1

print(f'results here: {results}')
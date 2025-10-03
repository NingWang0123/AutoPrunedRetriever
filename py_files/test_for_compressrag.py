from AutoPrunedRetriever import  get_context
from WordEmb import WordAvgEmbeddings, Word2VecEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
from typing import List, Tuple
import time

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

            # === metrics containers ===
        self.metrics_runs = []         # append per-call metrics dict
        self.last_metrics = None       # most recent one
        self.model_name = model_name

    # ---------- helpers for metrics ----------
    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _cuda_peak_mib(self) -> float:
        try:
            if torch.cuda.is_available():
                # 清零历史峰值再测当前调用峰值
                torch.cuda.reset_peak_memory_stats()
                return 0.0  # 先返回0，真正的峰值在_generate结束后读取
        except Exception:
            pass
        return 0.0

    def _get_cuda_peak_mib_after(self) -> float:
        try:
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated() / (1024**2))
        except Exception:
            pass
        return 0.0

    def _device_str(self) -> str:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_prompt(self, final_merged_json: str, question: str, decode: bool = False):
        """
        Build chat messages or (fallback) a single text prompt.
        IMPORTANT: We keep final_merged_json unchanged, but we DO NOT show <<<...>>> markers.
        """
        # High-priority rule: don't quote or reproduce context verbatim
        system_rule = (
            "You are a precise QA assistant. Use the contextual notes only to inform your answer. "
            "Do NOT quote or reproduce the notes verbatim. Do NOT output any markup like <<<...>>>. "
            "Do NOT output JSON or code blocks. Answer in plain sentences, 2-3 sentences when possible."
        )

        # We still pass your raw JSON, unchanged, but wrap it in neutral tags.
        # This removes the strong copying prior from <<<RETRIEVED_CONTEXT_*>>>.
        context_block = (
            "Contextual notes (for reference only—do not quote or copy):\n"
            "<context>\n" + final_merged_json + "\n</context>"
        )

        user_block = (
            f"[CURRENT QUESTION]: {question}\n"
            "[TASK]: Provide a short, direct answer. "
            "If the question is yes/no-like, give the conclusion AND 1–2 brief reasons.\n"
            "[ANSWER]:"
        )

        if getattr(self, "_use_chat_template", False):
            # Return structured messages for apply_chat_template
            messages = [
                {"role": "system", "content": system_rule},
                {"role": "system", "content": context_block},
                {"role": "user",   "content": user_block},
            ]
            return messages, None
        else:
            # Fallback: single concatenated text prompt
            prompt = (
                f"[System]\n{system_rule}\n\n"
                f"[Notes]\n{context_block}\n\n"
                f"[User]\n{user_block}\n"
            )
            return None, prompt


    @torch.no_grad()
    def _generate(self, system_msg: str, user_msg: str) -> str:
        self._cuda_peak_mib()
        t0 = time.perf_counter()

        if self._use_chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        t1 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        t2 = time.perf_counter()
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_tokens = int(outputs[0].shape[-1])  # includes input + generated with special tokens (approx)
        output_tokens = max(0, total_tokens - input_tokens)
        cut = text.rfind(user_msg)

        gen_latency_sec = t2 - t1
        latency_sec = t2 - t0
        # prompt chars
        prompt_chars = float(len(prompt))

        gen_info = {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(input_tokens + output_tokens),
            "latency_sec": float(latency_sec),
            "gen_latency_sec": float(gen_latency_sec),
            # retrieval_latency_sec / retrieved_count 由上层调用注入（见 take_questions）
            "retrieval_latency_sec": None,
            "prompt_chars": float(prompt_chars),
            # 衍生指标
            "throughput_tok_per_s": float((output_tokens / gen_latency_sec) if gen_latency_sec > 0 else 0.0),
            "prompt_tok_per_s": float((input_tokens / (latency_sec - gen_latency_sec)) if (latency_sec - gen_latency_sec) > 0 else 0.0),
            "device": self._device_str(),
            "dtype": str(getattr(self.model, "dtype", "unknown")),
            "model_name": self.model_name,
            "timestamp_start": t0,
            "timestamp_end": t2,
        }
        return text[cut + len(user_msg):].strip() if cut != -1 else text.strip(), gen_info
    
    def strip_think(self, s: str) -> Tuple[str, List[str]]:
            """提取思考并返回干净答案；支持 <think>...</think> 与多段 <|assistant|>。"""
            if not s:
                return "", []

            thinks: List[str] = []
            s_lower = s.lower()
            spans = []
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
            no_think_text = "".join(parts)

            blocks = [blk.strip()
                    for blk in re.split(r"(?i)<\|assistant\|>", no_think_text)
                    if blk and blk.strip()]

            if blocks:
                if len(blocks) >= 2:
                    thinks.extend(blocks[:-1])
                clean = blocks[-1].strip()
            else:
                clean = no_think_text.strip()

            clean = re.sub(r"(?:^|\n)\s*(Okay,|Let’s|Let's|Step by step|Thought:).*",
                        "", clean, flags=re.I)
            clean = re.sub(r"(?i)<\|assistant\|>", "", clean).strip()

            return clean, thinks
        
        
    def take_questions(self, final_merged_json, question, *, max_regen: int = 3, retrieval_time):
        def _clean_answer(s: str, limit=4):
            import re
            if not s:
                return ""

            s = s.replace("\ufeff", "").strip()

            s = re.sub(r"(?i)(<\|assistant\|>\s*)+", "", s, flags=re.MULTILINE)
            s = re.sub(r"^\s*assistant\s*[:：-]*\s*", "", s, flags=re.I)
            s = re.sub(r"^(#+\s*response\s*[:：-]*\s*)", "", s, flags=re.I)

            lines = s.splitlines()
            s = " ".join(l.strip() for l in lines).strip()
            s = re.sub(r'\s+', ' ', s)

            parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', s) if p.strip()]
            return " ".join(parts[:limit]) if parts else ""


        last_thinks: List[str] = []
        ans_clean = ""

        for attempt in range(max_regen):
            sys_msg, usr_msg = self._build_prompt(final_merged_json, question, decode= False)
            out, gen_info = self._generate(sys_msg, usr_msg)
            print(f"-------------RAW[{attempt+1}/{max_regen}]:", out)

            gen_info["retrieval_latency_sec"] = float(retrieval_time)
            gen_info["attempt"] = int(attempt + 1)
            gen_info["question_chars"] = float(len(str(question)))
            gen_info["answer_raw_chars"] = float(len(out))
            gen_info["answer_raw_tokens"] = float(self._count_tokens(out))
            gen_info["prompt_to_output_char_ratio"] = float(
                (gen_info["prompt_chars"] / max(1.0, gen_info["answer_raw_chars"]))
            )

            final_metrics = gen_info

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

        metrics_wrapped = {f"{question}":final_metrics or {} }
        self.last_metrics = metrics_wrapped
        self.metrics_runs.append(metrics_wrapped)


        if self.include_thinkings:
            thinks_str = "\n\n".join(t.strip() for t in last_thinks if t.strip())
            print("----------ANS:",ans_clean)
            return ans_clean, thinks_str
        else:
            print("----------ANS:",ans_clean)
            return ans_clean

if __name__ == "__main__":
    #word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")

    word_emb = Word2VecEmbeddings(model_name="word2vec-google-news-300")
    sentence_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")


    include_thinking = True
    phi_llm = Phi4MiniReasoningLLM(
        include_thinkings=include_thinking,                 
        model_name="microsoft/Phi-4-mini-reasoning",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9
    )

    import json
    with open("meta_codebook.json", "r") as f:
        ini = json.load(f)

    rag = CompressRag_rl(
        ini_meta_codebook = ini,
        sentence_emb=sentence_emb,
        word_emb=word_emb,
        llm=phi_llm,    
        thinkings_choice='overlap',  
        answers_choice='unique'       
    )

    rag.top_k = 5
    rag.top_m = 2
    rag.question_batch_size = 2
    rag.questions_db_batch_size = 16


    import json
    import numpy as np
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()   # ndarray -> list
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)       # numpy int -> int
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)     # numpy float -> float
        raise TypeError(f"Type {type(obj)} not serializable")


    def run_eval_case(question, reference_answer, facts_json_path, rag, work_mode="normal", llm_metrics=True, warm_start="coarse", websearch=None):
        if work_mode == "dpo":
            result = rag.run_work_flow_for_dpo(question, facts_json_path=facts_json_path, warm_start=warm_start)
        else:
            result = rag.run_work_flow(question, facts_json_path=facts_json_path, warm_start=warm_start, websearch=websearch)
        if isinstance(result, tuple):
            gen_text = result[0]
        else:
            gen_text = str(result)

        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        smooth = SmoothingFunction().method1
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        bleu = sentence_bleu([reference_answer.split()], gen_text.split(), smoothing_function=smooth)
        scores = scorer.score(reference_answer, gen_text)
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rougeL = scores["rougeL"].fmeasure

        if rag.llm.metrics_runs and isinstance(rag.llm.metrics_runs[-1], dict):
            last_metrics = list(rag.llm.metrics_runs[-1].values())[0]
            last_metrics["BLEU"] = bleu
            last_metrics["ROUGE-1"] = rouge1
            last_metrics["ROUGE-2"] = rouge2
            last_metrics["ROUGE-L"] = rougeL
        if llm_metrics:
            print(rag.llm.metrics_runs)
        print(result)
        print(f'BLEU: {bleu:.4f}, ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}')
        return gen_text, bleu, rouge1, rouge2, rougeL



    # DATA_PATH     = "context/medical_questions.json"
    # DATA_SLICE    = 2

    # with open(DATA_PATH, "r", encoding="utf-8") as f:
    #     raw_data = json.load(f)

    # raw_data = raw_data[:DATA_SLICE]
    # print(f"[INFO] Loaded {len(raw_data)} samples for answering)")
    # questions = [item["question"] for item in raw_data if "question" in item]
    # answers = [item["answer"] if "answer" in item else "" for item in raw_data if "question" in item]
    # i = 0
    # # for q, ref in zip(questions, answers):
    #     run_eval_case(q, ref, ["context/novel copy.json", "context/medical_sub.json"], rag, work_mode="normal", websearch= True)
    #     i += 1
        #with open(f"meta_codebook_{i}.json", "w", encoding="utf-8") as f:
        #    json.dump(rag.meta_codebook, f, ensure_ascii=False, indent=2, default=to_serializable)
        





        

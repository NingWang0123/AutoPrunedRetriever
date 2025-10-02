from  AutoPrunedRetriever import get_context
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,StoppingCriteria, StoppingCriteriaList
import re, os
from typing import List, Tuple
import time
from WordEmb import Word2VecEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# word_emb = Word2VecEmbeddings(model_name="word2vec-google-news-300")

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


class _StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, substrings):
        self.tokenizer = tokenizer
        self.substrings = substrings
        self.cache_len = 0

    def __call__(self, input_ids, scores, **kwargs):
        # CHANGED: decode only the newly generated tail and advance cache_len
        cur_len = int(input_ids.shape[1])
        if self.cache_len == 0:
            self.cache_len = cur_len
            return False
        text = self.tokenizer.decode(input_ids[0][self.cache_len:], skip_special_tokens=True)
        self.cache_len = cur_len
        for s in self.substrings:
            if s in text:
                return True
        return False


class Phi4MiniReasoningLLM:
    def __init__(self, include_thinkings: bool = True,
                model_name: str = "microsoft/Phi-4-mini-reasoning",
                max_new_tokens: int = 256,
                temperature: float = 0.2,
                top_p: float = 0.9):
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
                torch.cuda.reset_peak_memory_stats()
                return 0.0
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

    # CHANGED: hide <<<...>>> markers; keep your JSON EXACTLY as-is, wrapped in neutral <context> tags
    def _build_prompt(self, final_merged_json, question, decode=False):
        # Keep the system prompt bossy and unambiguous
        system_msg = (
            "You are a precise QA agent. "
            "Return ONLY the final answer in as 2-3 short sentences."
            "Do not give yes no, you have to conlude at least in one sentence."
            "No preamble, no labels, no emojis, no follow-up questions."
            "Read the information from the json below"
        )

        # Give the model context but forbid quoting it
        if decode:
            q_txt, gk_txt, st_txt, ft_txt = get_context(final_merged_json)
            ctx_lines = [
                "Context (do NOT quote or copy):",
                "<context>",
                q_txt,
                f"ANSWER_TRIPLES: {gk_txt}",
            ]
            if st_txt.strip().lower() != "none.":
                ctx_lines.append(f"THINK_TRIPLES: {st_txt}")
            if ft_txt.strip().lower() != "none.":
                ctx_lines.append(f"FACT_TRIPLES: {ft_txt}")
            ctx_lines.append("</context>")
            user_msg = "\n".join(ctx_lines) + "\n"
        else:
            user_msg = (
                "Context (do NOT quote or copy):\n<context>\n"
                f"{final_merged_json}\n</context>\n"
            )

        # IMPORTANT: ask for answer only; remove the literal "[ANSWER]:" label
        user_msg += (
            f"Question: {question}\n"
            "Instruction: Reply with ONLY the answer text. No extra words.\n"
        )
        return system_msg, user_msg

    # CHANGED: add sanitizer to strip any leakage if it happens
    def _sanitize_answer(self, text: str) -> str:
        import re
        # Nuke leaked blocks/labels/emoji/debug
        text = re.sub(r"(?is)<context>.*?</context>", "", text)
        text = re.sub(r"<<<RETRIEVED_CONTEXT_START>>>|<<<RETRIEVED_CONTEXT_END>>>", "", text)
        text = re.sub(r"\[ANSWER\]\s*:?|^Answer\s*:?\s*", "", text, flags=re.I)
        text = re.sub(r"^[-–—]+RAW.*?$", "", text, flags=re.I | re.M)
        text = re.sub(r"^[-–—]+ANS.*?$", "", text, flags=re.I | re.M)
        text = re.sub(r"INFO:.*?$", "", text, flags=re.M)
        text = text.replace("Is this correct?", "").replace("🐉", "")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Keep only the first sentence/phrase
        m = re.match(r"(.+?)(?:[.!?](?:\s|$)|$)", text)
        return (m.group(1) if m else text).strip()
    
    @torch.no_grad()
    def _generate(self, system_msg: str, user_msg: str) -> str:
        self._cuda_peak_mib()
        t0 = time.perf_counter()

        if self._use_chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        # Beefed-up stop list to catch your markers & accidental echoes
        stop_strings = [
            "<context>", "</context>",
            "Context (do NOT quote or copy):",
            "Instruction:", "Question:",
            "RAW", "ANS:", "INFO:", "ft_txt",
            "Is this correct"
        ]
        stopping_criteria = StoppingCriteriaList([
            _StopOnSubstrings(self.tokenizer, stop_strings)
        ])

        t1 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.15,   # nudge away from copying context
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        t2 = time.perf_counter()

        text_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_only = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        completion = text_full[len(prompt_only):].strip() if text_full.startswith(prompt_only) else text_full.strip()

        completion = self._sanitize_answer(completion)

        total_tokens = int(outputs[0].shape[-1])
        output_tokens = max(0, total_tokens - input_tokens)

        gen_info = {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(input_tokens + output_tokens),
            "latency_sec": float(t2 - t0),
            "gen_latency_sec": float(t2 - t1),
            "retrieval_latency_sec": None,
            "prompt_chars": float(len(prompt)),
            "throughput_tok_per_s": float((output_tokens / (t2 - t1)) if (t2 - t1) > 0 else 0.0),
            "prompt_tok_per_s": float((input_tokens / max(1e-6, (t2 - t0) - (t2 - t1)))),
            "device": self._device_str(),
            "dtype": str(getattr(self.model, "dtype", "unknown")),
            "model_name": self.model_name,
            "timestamp_start": t0,
            "timestamp_end": t2,
        }
        return completion, gen_info


    # --- REPLACE take_questions (smaller & quieter) ---
    def take_questions(self, final_merged_json, question, *, max_regen: int = 3, retrieval_time):
        last_metrics = None
        ans_clean = ""

        for attempt in range(max_regen):
            sys_msg, usr_msg = self._build_prompt(final_merged_json, question, decode=False)
            out, gen_info = self._generate(sys_msg, usr_msg)
            gen_info["retrieval_latency_sec"] = float(retrieval_time)
            gen_info["attempt"] = int(attempt + 1)
            gen_info["question_chars"] = float(len(str(question)))
            gen_info["answer_raw_chars"] = float(len(out))
            gen_info["answer_raw_tokens"] = float(self._count_tokens(out))
            gen_info["prompt_to_output_char_ratio"] = float(
                (gen_info["prompt_chars"] / max(1.0, gen_info["answer_raw_chars"]))
            )

            candidate = self._sanitize_answer(out)
            if candidate:
                ans_clean = candidate
                last_metrics = gen_info
                break

        if not ans_clean:
            ans_clean = "No answer."
            last_metrics = last_metrics or {}

        metrics_wrapped = {f"{question}": last_metrics}
        self.last_metrics = metrics_wrapped
        self.metrics_runs.append(metrics_wrapped)

        # Always return JUST the clean answer; no thinks/debug
        return ans_clean

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
            sys_msg, usr_msg = self._build_prompt(final_merged_json, question, decode=False)
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

        metrics_wrapped = {f"{question}": final_metrics or {}}
        self.last_metrics = metrics_wrapped
        self.metrics_runs.append(metrics_wrapped)

        if self.include_thinkings:
            thinks_str = "\n\n".join(t.strip() for t in last_thinks if t.strip())
            print("----------ANS:", ans_clean)
            return ans_clean, thinks_str
        else:
            print("----------ANS:", ans_clean)
            return ans_clean

from CompressRag_rl_v1 import CompressRag_rl,decode_questions, get_context
from WordEmb import WordAvgEmbeddings, Word2VecEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
from typing import List, Tuple
import time
import openai
import tiktoken
from pathlib import Path

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



# ============================================
# DeepSeek LLM Class
# ============================================

class DeepSeekLLM:
    def __init__(self, include_thinkings: bool = True,
                model_name: str = "deepseek-chat",
                max_new_tokens: int = 256,
                temperature: float = 0.3,
                top_p: float = 0.95,
                use_cache: bool = True,
                api_key: str = None,
                base_url: str = "https://api.deepseek.com/v1"):
        self.include_thinkings = include_thinkings
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_cache = use_cache
        self.model_name = model_name
        
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key is None:
                try:
                    api_key = Path("deepseek_key.txt").read_text().strip()
                except:
                    raise ValueError("请提供DeepSeek API密钥")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self._use_chat_template = True  

        # === metrics containers ===
        self.metrics_runs = []       
        self.last_metrics = None       #

    # ---------- helpers for metrics ----------
    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _cuda_peak_mib(self) -> float:
        return 0.0  

    def _get_cuda_peak_mib_after(self) -> float:
        return 0.0  

    def _device_str(self) -> str:
        return "deepseek-api"  
    def _build_prompt(self, final_merged_json, question, decode=False):
        # Keep the system prompt bossy and unambiguous
        system_msg = (
            "You are a precise QA agent. "
            "Return ONLY the final answer in as 2-3 short sentences."
            "Do not give yes no or single word, you have to conlude at least in one sentence."
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

    def _generate(self, system_msg: str, user_msg: str) -> str:
        self._cuda_peak_mib()
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt_text = system_msg + user_msg
        input_tokens = self._count_tokens(prompt_text)
        prompt_chars = float(len(prompt_text))

        t1 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            text = response.choices[0].message.content or ""
            output_tokens = response.usage.completion_tokens if response.usage else self._count_tokens(text)
            total_tokens = response.usage.total_tokens if response.usage else (input_tokens + output_tokens)
            
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            text = "API调用失败，请检查网络和API密钥。"
            output_tokens = self._count_tokens(text)
            total_tokens = input_tokens + output_tokens
        
        t2 = time.perf_counter()
        
        gen_latency_sec = t2 - t1
        latency_sec = t2 - t0

        gen_info = {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(total_tokens),
            "latency_sec": float(latency_sec),
            "gen_latency_sec": float(gen_latency_sec),
            "retrieval_latency_sec": None,
            "prompt_chars": float(prompt_chars),
            "throughput_tok_per_s": float((output_tokens / gen_latency_sec) if gen_latency_sec > 0 else 0.0),
            "prompt_tok_per_s": float((input_tokens / (latency_sec - gen_latency_sec)) if (latency_sec - gen_latency_sec) > 0 else 0.0),
            "device": self._device_str(),
            "dtype": "deepseek-api",
            "model_name": self.model_name,
            "timestamp_start": t0,
            "timestamp_end": t2,
        }
        return text.strip(), gen_info
    
    def strip_think(self, s: str) -> Tuple[str, List[str]]:
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

        clean = re.sub(r"(?:^|\n)\s*(Okay,|Let's|Let's|Step by step|Thought:).*",
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

# ============================================
# OpenAI LLM Class
# ============================================

class OpenAILLM:
    def __init__(self, include_thinkings: bool = True,
                model_name: str = "gpt-4o-mini",
                max_new_tokens: int = 256,
                temperature: float = 0.3,
                top_p: float = 0.95,
                use_cache: bool = True,
                api_key: str = None,
                base_url: str = None):
        self.include_thinkings = include_thinkings
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_cache = use_cache
        self.model_name = model_name
        
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                try:
                    api_key = Path("openai_key.txt").read_text().strip()
                except:
                    raise ValueError("Please offer OpenAI API key")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 用于token计算
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self._use_chat_template = True  

        # === metrics containers ===
        self.metrics_runs = []         # append per-call metrics dict
        self.last_metrics = None       # most recent one

    # ---------- helpers for metrics ----------
    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _cuda_peak_mib(self) -> float:
        return 0.0  

    def _get_cuda_peak_mib_after(self) -> float:
        return 0.0  
    
    def _device_str(self) -> str:
        return "api"  

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

    def _generate(self, system_msg: str, user_msg: str) -> str:
        self._cuda_peak_mib()
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        # 计算输入token数
        prompt_text = system_msg + user_msg
        input_tokens = self._count_tokens(prompt_text)
        prompt_chars = float(len(prompt_text))

        t1 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            text = response.choices[0].message.content or ""
            output_tokens = response.usage.completion_tokens if response.usage else self._count_tokens(text)
            total_tokens = response.usage.total_tokens if response.usage else (input_tokens + output_tokens)
            
        except Exception as e:
            print(f"API调用失败: {e}")
            text = "API调用失败，请检查网络和API密钥。"
            output_tokens = self._count_tokens(text)
            total_tokens = input_tokens + output_tokens
        
        t2 = time.perf_counter()
        
        gen_latency_sec = t2 - t1
        latency_sec = t2 - t0

        gen_info = {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(total_tokens),
            "latency_sec": float(latency_sec),
            "gen_latency_sec": float(gen_latency_sec),
            # retrieval_latency_sec / retrieved_count 由上层调用注入（见 take_questions）
            "retrieval_latency_sec": None,
            "prompt_chars": float(prompt_chars),
            # 衍生指标
            "throughput_tok_per_s": float((output_tokens / gen_latency_sec) if gen_latency_sec > 0 else 0.0),
            "prompt_tok_per_s": float((input_tokens / (latency_sec - gen_latency_sec)) if (latency_sec - gen_latency_sec) > 0 else 0.0),
            "device": self._device_str(),
            "dtype": "api",
            "model_name": self.model_name,
            "timestamp_start": t0,
            "timestamp_end": t2,
        }
        return text.strip(), gen_info
    
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

    llm_provider = "deepseek"  

    if llm_provider == "deepseek":
        api_llm = DeepSeekLLM(
            include_thinkings=include_thinking,                 
            model_name="deepseek-chat",  
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
            # api_key="your-deepseek-api-key",  # 可选，不填会从环境变量或deepseek_key.txt读取
            # base_url="https://api.deepseek.com/v1",  # DeepSeek API地址
        )
    else:
        api_llm = OpenAILLM(  # 这个现在是OpenAI兼容的
            include_thinkings=include_thinking,                 
            model_name="gpt-4o-mini",  # 或者 "gpt-3.5-turbo", "gpt-4o" 等
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
            api_key="your-openai-api-key",  # 可选，不填会从环境变量或openai_key.txt读取
            # base_url="https://api.openai.com/v1",  # 可选，使用其他兼容服务
        )

    import json
    with open("meta_codebook.json", "r") as f:
        ini = json.load(f)

    rag = CompressRag_rl(
        ini_meta_codebook = ini,
        sentence_emb=sentence_emb,
        word_emb=word_emb,
        llm=api_llm,    
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



    DATA_PATH     = "context/medical_questions.json"
    DATA_SLICE    = 2

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = raw_data[:DATA_SLICE]
    print(f"[INFO] Loaded {len(raw_data)} samples for answering)")
    questions = [item["question"] for item in raw_data if "question" in item]
    answers = [item["answer"] if "answer" in item else "" for item in raw_data if "question" in item]
    i = 0
    for q, ref in zip(questions, answers):
        run_eval_case(q, ref, ["context/novel copy.json", "context/medical_sub.json"], rag, work_mode="normal", websearch= True)
        i += 1
        #with open(f"meta_codebook_{i}.json", "w", encoding="utf-8") as f:
        #    json.dump(rag.meta_codebook, f, ensure_ascii=False, indent=2, default=to_serializable)


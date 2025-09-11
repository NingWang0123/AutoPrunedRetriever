from CompressRag_rl_v1 import CompressRag,WordAvgEmbeddings,decode_questions
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
        # 一些模型需要 chat 模板；Phi-4-mini-reasoning 支持 messages
        if not hasattr(self.tokenizer, "apply_chat_template"):
            # 兜底：后面直接拼接纯文本
            self._use_chat_template = False
        else:
            self._use_chat_template = True

    def _linearize_triples_block(self, triples):
        # triples: [[h, r, t], ...]  -> "h r t. h r t. ..."
        if not triples:
            return "None."
        return " ".join([f"{h} {r} {t}." for h, r, t in triples])

    def _build_prompt(self, final_merged_json, question):
        """
        把 compact 后的 JSON（含 questions / given knowledge / 可选 thinkings）转换成提示词。
        轻量说明：我们希望模型输出两个段落，便于后续规则抽取。
        """
        user_msg = ""
        print("final_merged_json", final_merged_json)
        qs = final_merged_json.get("questions([[e,r,e], ...])", [])
        gk = final_merged_json.get("given knowledge([[e,r,e], ...])", [])
        st = final_merged_json.get("start thinking with(edges[i])", [])  # 已解码版本可能叫这个或空

        # 只拿第一组作为信号（可以按需扩展为多组）
        q_txt  = self._linearize_triples_block(qs[0] if qs else [])
        gk_txt = self._linearize_triples_block(gk[0] if gk else [])
        st_txt = self._linearize_triples_block(st[0] if st else [])

        system_msg = (
            "You are a precise QA agent that answers by expressing facts as short, "
            "plain English statements. Keep outputs concise and factual."
        )

        user_msg += ("<<<RETRIEVED_CONTEXT_START>>>\n"
                    "The system searched for a related question in the database. Below are related question's graph triples and its prior answer as reference. "
                    "You don't have to follow it completely, just use it as a reference.\n"
                    f"[RELATED QUESTION'S GRAPH TRIPLES]:\n{q_txt}\n"
                    f"[RELATED QUESTION'S ANSWER]: {gk_txt}\n"
                    "<<<RETRIEVED_CONTEXT_END>>>")
        
        user_msg += (
            f"[CURRENT QUESTION]: {question} \n"
            "[TASK]: You are a QA assistant for open-ended questions.\n"
            f"- Give a short, direct answer in 2–3 sentences."
            "- Do NOT restrict to yes/no.\n"
            "[FORMAT]: Write complete sentences (not a single word)."
            "Avoid starting with just 'Yes.' or 'No.'; if the question is yes/no-style, state the conclusion AND 1–2 short reasons.\n"
            "[ANSWER]: "
        )

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
            sys_msg, usr_msg = self._build_prompt(final_merged_json, question)
            out = self._generate(sys_msg, usr_msg)
            print(f"-------------RAW[{attempt+1}/{max_regen}]:", out)

            raw = out.strip()
            m = re.search(r"\[answers\](.*?)(?:\[thinkings\]|\Z)", raw, flags=re.S | re.I)
            ans_region = m.group(1).strip() if m else raw

            # 1) 提取并移除 <think>...</think>，保留原文
            ans_no_think, thinks = self.strip_think(ans_region)
            last_thinks = thinks  # 原文 list[str]

            # 2) 答案清理（不影响 think 原文）
            ans_clean = _clean_answer(ans_no_think, 4).strip()
            if ans_clean:
                break

        if not ans_clean:
            ans_clean = "No answer."

        if self.include_thinkings:
            # 合并成一个字符串，原文不缩略：用空行分隔各个 <think> 块
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
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9
)

rag = CompressRag(
    sentence_emb=sentence_emb,
    word_emb=word_emb,
    include_thinkings=include_thinking,
    llm=phi_llm,
)

rag.top_k = 5
rag.top_m = 2
rag.question_batch_size = 2
rag.questions_db_batch_size = 16

questions = [
    "Is the Great Wall visible from space?",
    "Where is the Great Wall located in China?",
]

i = 0
for q in questions:
    print(f'q {i}')
    print(rag.run_work_flow(q))
    # print(cr.meta_codebook)
    i+=1
    
# python py_files/test_for_compressrag.py
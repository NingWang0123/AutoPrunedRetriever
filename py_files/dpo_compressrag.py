from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from contextlib import contextmanager
import math, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from CompressRag_rl_v1 import CompressRag_rl,WordAvgEmbeddings, get_context,merging_codebook
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
from test_for_compressrag import Phi4MiniReasoningLLM


# ===============================
# 0) GLOBAL CHOICES / MAPPINGS
# ===============================
# THINKINGS_CHOICES = ['overlap','unique','not_include']
# ANSWERS_CHOICES   = ['overlap','unique','not_include']

# THINKINGS_CHOICES = ['overlap','not_include']
THINKINGS_CHOICES = ['not_include']
ANSWERS_CHOICES   = ['overlap','not_include']


TH2I = {v:i for i,v in enumerate(THINKINGS_CHOICES)}
AN2I = {v:i for i,v in enumerate(ANSWERS_CHOICES)}
I2TH = {i:v for v,i in TH2I.items()}
I2AN = {i:v for v,i in AN2I.items()}

# Combine cadence is scheduled by a bandit over these arms.
# Semantics: 0 = "never combine", 1 = every round, 3 = every 3 rounds
# COMBINE_ARMS = [0, 1, 3]
COMBINE_ARMS = [0, 1]

# Large sentinel to emulate "never combine" without modulo-by-zero in user code
_NEVER_COMBINE_SENTINEL = 10**9


# ===============================
# 1) FEATURIZATION
# ===============================
def _hashed_char_ngrams(s: str, dims: int = 384, n: int = 3) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    s = s or ""
    if len(s) >= n:
        for i in range(len(s)-n+1):
            h = hash(s[i:i+n]) % dims
            v[h] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def featurize_query(cr, q: str, dims: int = 384) -> np.ndarray:
    """Use LLM/encoder if present; else fallback to hashed ngrams."""
    enc = getattr(cr, "sentence_emb", None)
    if enc is not None:
        for meth in ("embed_query","encode","embed_documents"):
            if hasattr(enc, meth):
                try:
                    vec = getattr(enc, meth)(q)
                    vec = np.array(vec, dtype=np.float32)
                    return vec[0] if vec.ndim > 1 else vec
                except Exception:
                    pass
    return _hashed_char_ngrams(q, dims=dims, n=3)

def featurize_state(cr) -> np.ndarray:
    """Lightweight CR state features for the scheduler."""
    # meta size
    meta = getattr(cr, "meta_codebook", None)
    if meta is None:
        meta_size = 0
    else:
        # try dict-like cardinality; fallback to len(str(meta))
        try:
            meta_size = len(meta)
        except Exception:
            meta_size = len(str(meta))

    # round counter
    r = getattr(cr, "round", 0)

    # any rolling reward avg tracked on CR? (optional)
    rr = getattr(cr, "_recent_reward_avg", None)
    if rr is None:
        rr_val = 0.0
    else:
        try:
            rr_val = float(rr)
        except Exception:
            rr_val = 0.0

    # simple transforms for scale
    feats = np.array([
        1.0,                                 # bias
        math.log1p(float(meta_size)),        # meta size (log)
        float(r),                            # current round
        rr_val,                              # recent reward avg (0 if not tracked)
    ], dtype=np.float32)
    return feats


# ===============================
# 2) TEMPORARY STRATEGY (ANS/TH)
# ===============================
@contextmanager
def temp_ans_th(cr,
                answers_choice: str,
                thinkings_choice: str,
                isolate_state: bool = True):
    """
    Temporarily set CR's per-question knobs (answers, thinkings).
    State isolation (meta_codebook/round) is optional.
    """
    prev_ans  = getattr(cr, "answers_choice", None)
    prev_th   = getattr(cr, "thinkings_choice", None)
    prev_meta = None
    prev_round = None

    if isolate_state:
        if hasattr(cr, "meta_codebook"):
            import copy
            prev_meta = copy.deepcopy(cr.meta_codebook)
        if hasattr(cr, "round"):
            prev_round = cr.round

    cr.answers_choice   = answers_choice
    cr.thinkings_choice = thinkings_choice

    try:
        yield
    finally:
        if prev_ans is not None:  cr.answers_choice = prev_ans
        if prev_th is not None:   cr.thinkings_choice = prev_th
        if isolate_state:
            if prev_meta is not None: cr.meta_codebook = prev_meta
            if prev_round is not None: cr.round = prev_round


def set_combine_rounds(cr, rounds: int):
    """
    Safely set combine cadence on CR to avoid modulo-by-zero in user code.
    If rounds == 0 => emulate 'never combine' by a huge sentinel.
    """
    if rounds == 0:
        setattr(cr, "combine_ents_rounds", _NEVER_COMBINE_SENTINEL)
        setattr(cr, "_combine_never", True)
    else:
        setattr(cr, "combine_ents_rounds", int(rounds))
        setattr(cr, "_combine_never", False)


# ===============================
# 3) REWARD HELPERS
# ===============================
def _tokset(s: str) -> set:
    return set((s or "").lower().split())

def f1_overlap(pred: str, gold: str) -> float:
    P, G = _tokset(pred), _tokset(gold)
    if not P and not G: return 1.0
    if not P or not G:  return 0.0
    tp = len(P & G); prec = tp/(len(P)+1e-8); rec = tp/(len(G)+1e-8)
    return 0.0 if prec+rec == 0 else 2*prec*rec/(prec+rec)

def default_reward(pred_answer: str, gold_answer: Optional[str]) -> float:
    base = f1_overlap(pred_answer, gold_answer or "")
    toks = len((pred_answer or "").split())
    return base - 0.0005*max(0, toks-256)


# ===============================
# 4) DPO DATA (2-HEAD: ans/th)
# ===============================
@dataclass
class PrefExample2:
    x: np.ndarray                 # question features
    y_pos: Tuple[int,int]         # (ans_idx, th_idx)
    y_neg: Tuple[int,int]

class PrefDataset2(torch.utils.data.Dataset):
    def __init__(self, examples: List[PrefExample2]):
        assert len(examples) > 0, "Empty preference dataset."
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex = self.examples[i]
        return (torch.tensor(ex.x, dtype=torch.float32),
                torch.tensor(ex.y_pos, dtype=torch.long),
                torch.tensor(ex.y_neg, dtype=torch.long))

def make_preference_dataset_2head(
    cr,
    questions: List[str],
    gold_answers: Optional[Dict[str,str]] = None,
    per_q_samples: int = 6,
    feature_dim: int = 384,
    reward_fn: Callable[[str, Optional[str]], float] = None,
    seed: int = 0,
    isolate_state: bool = True,
    combine_rounds_default: int = 1,
    ANSWERS_CHOICES = ANSWERS_CHOICES,
    THINKINGS_CHOICES = THINKINGS_CHOICES,

) -> List[PrefExample2]:
    """
    Build DPO pairs for (answers_choice, thinkings_choice) ONLY.
    Combine cadence is fixed to 'combine_rounds_default' here and will
    be handled by the contextual bandit at runtime.
    """
    if reward_fn is None:
        reward_fn = default_reward

    rng = random.Random(seed)
    all_pairs = [(ai, ti)
                 for ai in range(len(ANSWERS_CHOICES))
                 for ti in range(len(THINKINGS_CHOICES))]

    examples: List[PrefExample2] = []

    # Set combine cadence safely for whole dataset creation
    prev_combine = getattr(cr, "combine_ents_rounds", None)
    set_combine_rounds(cr, combine_rounds_default)

    for q in questions:
        x = featurize_query(cr, q, dims=feature_dim)
        tried = rng.sample(all_pairs, k=min(per_q_samples, len(all_pairs)))

        scored: List[Tuple[Tuple[int,int], float]] = []
        for (ai, ti) in tried:
            ans = ANSWERS_CHOICES[ai]
            th  = THINKINGS_CHOICES[ti]
            # try:
            #     with temp_ans_th(cr, ans, th, isolate_state=isolate_state):
            #         pred = cr.run_work_flow(q)
            #         print(f'pred{pred}')
            # except Exception:
            #     continue
            with temp_ans_th(cr, ans, th, isolate_state=isolate_state):
                pred = cr.run_work_flow(q)
                print(f'pred{pred}')
            score = reward_fn(pred, gold_answers.get(q) if gold_answers else None)
            scored.append(((ai, ti), score))

        if len(scored) < 2:
            continue
        scored.sort(key=lambda z: z[1], reverse=True)
        y_pos, y_neg = scored[0][0], scored[-1][0]
        examples.append(PrefExample2(x=x, y_pos=y_pos, y_neg=y_neg))

    # restore original combine cadence
    if prev_combine is not None:
        set_combine_rounds(cr, prev_combine)

    return examples


# ===============================
# 5) DPO POLICY (2-HEAD)
# ===============================
class StrategyPolicy2Head(nn.Module):
    """MLP with two categorical heads: answers, thinkings."""
    def __init__(self, input_dim: int, hidden: int = 512, drop: float = 0.1,ANSWERS_CHOICES = ANSWERS_CHOICES, THINKINGS_CHOICES = THINKINGS_CHOICES,):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(drop),
        )
        self.ans = nn.Linear(hidden, len(ANSWERS_CHOICES))
        self.th  = nn.Linear(hidden, len(THINKINGS_CHOICES))

    def forward(self, x):  # x: [B,D]
        h = self.ff(x)
        return self.ans(h), self.th(h)


    def log_prob(self, x, y):  # y: [B,2] longs
        la, lt = self.forward(x)
        logpa = F.log_softmax(la, dim=-1).gather(-1, y[:,0:1]).squeeze(-1)
        logpt = F.log_softmax(lt, dim=-1).gather(-1, y[:,1:2]).squeeze(-1)
        return logpa + logpt

    @torch.no_grad()
    def sample(self, x, greedy: bool = True):
        la, lt = self.forward(x)
        if greedy:
            ya, yt = la.argmax(-1), lt.argmax(-1)
        else:
            ya = torch.distributions.Categorical(logits=la).sample()
            yt = torch.distributions.Categorical(logits=lt).sample()
        return torch.stack([ya, yt], dim=-1)

def dpo_loss_2head(policy: StrategyPolicy2Head,
                   ref: StrategyPolicy2Head,
                   x: torch.Tensor,
                   y_pos: torch.Tensor,
                   y_neg: torch.Tensor,
                   beta: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        lp_ref_pos = ref.log_prob(x, y_pos)
        lp_ref_neg = ref.log_prob(x, y_neg)
    lp_pos = policy.log_prob(x, y_pos)
    lp_neg = policy.log_prob(x, y_neg)
    z = beta * (lp_pos - lp_neg) - (lp_ref_pos - lp_ref_neg)
    return F.binary_cross_entropy_with_logits(z, torch.ones_like(z))

@dataclass
class TrainCfg:
    lr: float = 3e-4
    epochs: int = 5
    batch_size: int = 32
    beta: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_dpo_2head(examples: List[PrefExample2], input_dim: int, cfg: TrainCfg = TrainCfg()
                   ) -> Tuple[StrategyPolicy2Head, StrategyPolicy2Head]:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    ds = PrefDataset2(examples)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    policy = StrategyPolicy2Head(input_dim=input_dim).to(cfg.device)
    ref    = StrategyPolicy2Head(input_dim=input_dim).to(cfg.device)
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for e in range(cfg.epochs):
        policy.train()
        tot, nb = 0.0, 0
        for x, y_pos, y_neg in dl:
            x, y_pos, y_neg = x.to(cfg.device), y_pos.to(cfg.device), y_neg.to(cfg.device)
            loss = dpo_loss_2head(policy, ref, x, y_pos, y_neg, beta=cfg.beta)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"[DPO-2H] epoch {e+1}/{cfg.epochs} loss={tot/max(1,nb):.4f}")

    policy.eval()
    return policy, ref


# ===============================
# 6) LINUCB SCHEDULER (COMBINE)
# ===============================
class LinUCBArm:
    def __init__(self, d: int, alpha: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.A = np.eye(d, dtype=np.float32)   # d x d
        self.b = np.zeros((d,1), dtype=np.float32)  # d x 1

    def theta(self) -> np.ndarray:
        A_inv = np.linalg.inv(self.A)
        return (A_inv @ self.b).reshape(-1)  # d,

    def ucb(self, x: np.ndarray) -> float:
        x = x.reshape(-1,1)  # d x 1
        A_inv = np.linalg.inv(self.A)
        mu = float((x.T @ A_inv @ self.b).squeeze())   # mean
        ci = self.alpha * float(np.sqrt((x.T @ A_inv @ x).squeeze()))  # bonus
        return mu + ci

    def update(self, x: np.ndarray, reward: float):
        x = x.reshape(-1,1)
        self.A += (x @ x.T)
        self.b += reward * x

class CombineScheduler:
    """
    LinUCB contextual bandit over arms in COMBINE_ARMS.
    Context = featurize_state(cr).
    """
    def __init__(self, d: int, arms: List[int] = None, alpha: float = 1.0, epsilon: float = 0.05):
        self.arms = arms or COMBINE_ARMS
        self.alpha = alpha
        self.epsilon = epsilon
        self.models = {a: LinUCBArm(d=d, alpha=alpha) for a in self.arms}

    def select(self, x: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(self.arms)
        # pick arm with highest UCB
        scores = [(a, self.models[a].ucb(x)) for a in self.arms]
        scores.sort(key=lambda z: z[1], reverse=True)
        return scores[0][0]

    def update(self, arm: int, x: np.ndarray, reward: float):
        self.models[arm].update(x, reward)


# ===============================
# 7) INFERENCE PIPELINE
# ===============================
@torch.no_grad()
def select_ans_th(policy: StrategyPolicy2Head, cr, q: str, feature_dim: int = 384, greedy: bool = True,  ANSWERS_CHOICES = ANSWERS_CHOICES,THINKINGS_CHOICES = THINKINGS_CHOICES,):
    x = torch.tensor(featurize_query(cr, q, dims=feature_dim),dtype=torch.float32).unsqueeze(0).to(next(policy.parameters()).device)
    y = policy.sample(x, greedy=greedy)[0].cpu().numpy().tolist()
    ai, ti = int(y[0]), int(y[1])
    return (ANSWERS_CHOICES[ai], THINKINGS_CHOICES[ti])

def answer_with_auto_strategy(
    cr: CompressRag_rl,
    policy: StrategyPolicy2Head,
    scheduler: CombineScheduler,
    q: str,
    reward_fn: Callable[[str, Optional[str]], float] = None,
    gold_answer: Optional[str] = None,
    greedy: bool = True,
    ANSWERS_CHOICES = ANSWERS_CHOICES,
    THINKINGS_CHOICES = THINKINGS_CHOICES,
) -> Tuple[str, Dict[str, object]]:
    """
    1) Choose answers/th via DPO policy (question features)
    2) Choose combine_ents_rounds via LinUCB (state features)
    3) Run CR once and return result (+ metadata)
    4) If reward_fn & gold provided, update scheduler online
    """
    # 1) per-question knobs
    ans_choice, th_choice = select_ans_th(policy, cr, q, greedy=greedy,ANSWERS_CHOICES = ANSWERS_CHOICES,THINKINGS_CHOICES = THINKINGS_CHOICES)

    # 2) combine cadence via state features
    state_x = featurize_state(cr)  # d
    arm = scheduler.select(state_x, explore=True)
    # set combine rounds safely
    set_combine_rounds(cr, arm)

    # 3) run
    with temp_ans_th(cr, ans_choice, th_choice, isolate_state=False):
        pred = cr.run_work_flow(q)
        fact_context =  cr.cur_fact_context

    # 4) update bandit online if evaluable
    reward = None
    if reward_fn is not None:
        try:
            reward = float(reward_fn(pred, gold_answer))
        except Exception:
            reward = None
    if reward is not None:
        scheduler.update(arm, state_x, reward)
        # optionally maintain a rolling average on CR
        try:
            rr = getattr(cr, "_recent_reward_avg", 0.0)
            cr._recent_reward_avg = 0.9*float(rr) + 0.1*reward
        except Exception:
            pass

    meta = {
        "answers_choice": ans_choice,
        "thinkings_choice": th_choice,
        "combine_rounds": arm if arm != _NEVER_COMBINE_SENTINEL else 0,
        "reward": reward,
        "fact_context": fact_context,
    }
    return pred, meta


#### class for llm
# class Phi4MiniReasoningLLM:
#     def __init__(self, include_thinkings: bool = True,
#                 model_name: str = "microsoft/Phi-4-mini-reasoning",
#                 max_new_tokens: int = 256,
#                 temperature: float = 0.3,
#                 top_p: float = 0.95):
#         self.include_thinkings = include_thinkings
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         self.top_p = top_p

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             trust_remote_code=True,
#         )
#         if not hasattr(self.tokenizer, "apply_chat_template"):
#             self._use_chat_template = False
#         else:
#             self._use_chat_template = True


#     def _build_prompt(self, final_merged_json, question = 'answer q'):
#         q_txt, gk_txt, st_txt, ft_txt = get_context(final_merged_json)
#         user_msg = ""

#         system_msg = (
#             "You are a precise QA agent that answers by expressing facts as short, "
#             "plain English statements. Keep outputs concise and factual."
#         )

#         ctx_lines = [
#             "<<<RETRIEVED_CONTEXT_START>>>",
#             "The system searched for a related question in the database. Below are related question's graph triples and its prior answer as reference. You don't have to follow it completely, just use it as a reference.",
#             "[RELATED QUESTION'S GRAPH TRIPLES]:",
#             q_txt,
#             f"[RELATED QUESTION'S ANSWER TRIPLES]: {gk_txt}",
#         ]
        
#         if st_txt.strip().lower() != "none.":
#             ctx_lines.append(f"[RELATED THINKING“S TRIPLES]: {st_txt}")

#         if ft_txt.strip().lower() != "none.":
#             ctx_lines.append(f"[RELATED FACTS'S TRIPLES]: {ft_txt}")

#         ctx_lines.append("<<<RETRIEVED_CONTEXT_END>>>")

#         user_msg += "\n".join(ctx_lines) + "\n"

#         user_msg += (
#             f"[CURRENT QUESTION]: {question} \n"
#             "[TASK]: You are a QA assistant for open-ended questions.\n"
#             f"- Give a short, direct answer in 2–3 sentences."
#             "- Do NOT restrict to yes/no.\n"
#             "[FORMAT]: Write complete sentences (not a single word)."
#             "Avoid starting with just 'Yes.' or 'No.'; if the question is yes/no-style, state the conclusion AND 1–2 short reasons.\n"
#             "[ANSWER]: "
#         )

#         print(user_msg)
#         return system_msg, user_msg

#     @torch.no_grad()
#     def _generate(self, system_msg: str, user_msg: str) -> str:
#         if self._use_chat_template:
#             messages = [
#                 {"role": "system", "content": system_msg},
#                 {"role": "user",   "content": user_msg},
#             ]
#             prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         else:
#             prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=(self.temperature > 0),
#             temperature=self.temperature,
#             top_p=self.top_p,
#             pad_token_id=self.tokenizer.eos_token_id,
#         )
#         text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         cut = text.rfind(user_msg)
#         return text[cut + len(user_msg):].strip() if cut != -1 else text.strip()
    
#     def strip_think(self, s: str) -> Tuple[str, List[str]]:
#         """Remove <think>...</think> blocks; also handle dangling <think> at EOF."""
#         if not s:
#             return "", []
#         s_lower = s.lower()
#         thinks: List[str] = []
#         spans: List[Tuple[int, int]] = []

#         for m in re.finditer(r"<think>(.*?)</think>", s, flags=re.S | re.I):
#             thinks.append(m.group(1).strip())
#             spans.append((m.start(), m.end()))

#         last_open = s_lower.rfind("<think>")
#         if last_open != -1 and s_lower.find("</think>", last_open) == -1:
#             content_start = last_open + len("<think>")
#             dangling_text = s[content_start:].strip()
#             if dangling_text:
#                 thinks.append(dangling_text)
#             spans.append((last_open, len(s)))

#         if spans:
#             spans.sort()
#             merged = []
#             cur_s, cur_e = spans[0]
#             for st, en in spans[1:]:
#                 if st <= cur_e:
#                     cur_e = max(cur_e, en)
#                 else:
#                     merged.append((cur_s, cur_e))
#                     cur_s, cur_e = st, en
#             merged.append((cur_s, cur_e))
#         else:
#             merged = []

#         parts = []
#         prev = 0
#         for st, en in merged:
#             if prev < st:
#                 parts.append(s[prev:st])
#             prev = en
#         if prev < len(s):
#             parts.append(s[prev:])

#         clean = "".join(parts)
#         clean = re.sub(r"(?:^|\n)\s*(Okay,|Let’s|Let's|Step by step|Thought:).*", "", clean, flags=re.I)
#         return clean.strip(), thinks
        
#     def take_questions(self, final_merged_json, question, *, max_regen: int = 3):
#         def _clean_answer(s: str, limit=4):
#             parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', s) if p.strip()]
#             if not parts:
#                 return ""
#             return " ".join(parts[:limit])

#         last_thinks: List[str] = []
#         ans_clean = ""

#         for attempt in range(max_regen):
#             sys_msg, usr_msg = self._build_prompt(final_merged_json, question)
#             out = self._generate(sys_msg, usr_msg)
#             print(f"-------------RAW[{attempt+1}/{max_regen}]:", out)

#             raw = out.strip()
#             m = re.search(r"\[answers\](.*?)(?:\[thinkings\]|\Z)", raw, flags=re.S | re.I)
#             ans_region = m.group(1).strip() if m else raw

#             ans_no_think, thinks = self.strip_think(ans_region)
#             last_thinks = thinks  

#             ans_clean = _clean_answer(ans_no_think, 4).strip()
#             if ans_clean:
#                 break

#         if not ans_clean:
#             ans_clean = "No answer."

#         if self.include_thinkings:
#             thinks_str = "\n\n".join(t.strip() for t in last_thinks if t.strip())
#             return ans_clean, thinks_str
#         else:
#             return ans_clean


# ===============================
# 8) SMALL EXAMPLE (USAGE)
# ===============================
# if __name__ == "__main__":

#     # --- 1) create CR 

#     # intialization

#     include_thinking = True
#     word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")
#     sentence_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     phi_llm = Phi4MiniReasoningLLM(
#         include_thinkings=include_thinking,                 
#         model_name="microsoft/Phi-4-mini-reasoning",
#         max_new_tokens=512,
#         temperature=0.2,
#         top_p=0.9
#     )
#     cr = CompressRag_rl(
#         ini_meta_codebook = {},
#         sentence_emb=sentence_emb,
#         word_emb=word_emb,
#         llm=phi_llm,
#         combine_ents_rounds=1,            # default; scheduler will overwrite
#         thinkings_choice='not_include',
#         answers_choice='overlap'
#     )

#     facts_json_paths = 'medical_sub.json'

#     facts_cb = cr.load_and_merge_facts(facts_json_paths, chunk_chars=100, overlap=30)
#     cr._facts_preloaded = True
#     cr.top_m = 2          # sentence-embedding rerank top-m

#     cr.meta_codebook = merging_codebook(
#         cr.meta_codebook, facts_cb,
#         type='facts', word_emb=cr.word_emb, use_thinkings=True
#     )
#     print('ini merging')
#     for k, v in cr.meta_codebook.items():
#         if "fact" in k.lower():
#             print(k, ":", v)
#     # ensure round exists
#     if not hasattr(cr, "round"):
#         cr.round = 0

#     # --- 2) build preference data for DPO (answers/th)
#     train_questions = [
#         "Who discovered penicillin?",
#         # "What is the capital of France?",
#         # "Define mitochondria.",
#         # "When was the UN founded?",
#     ]

#     # for q in train_questions:
#     #     cr.run_work_flow(q)

#     # print('after answer questions merging')

#     # for k, v in cr.meta_codebook.items():
#     #     if "fact" in k.lower():
#     #         print(k, ":", v)

#     gold = {
#         "Who discovered penicillin?": "Alexander Fleming",
#         # "What is the capital of France?": "Paris",
#         # "Define mitochondria.": "Organelle responsible for ATP production",
#         # "When was the UN founded?": "1945",
#     }

#     examples = make_preference_dataset_2head(
#         cr=cr,
#         questions=train_questions,
#         gold_answers=gold,
#         per_q_samples=6,
#         feature_dim=384,
#         reward_fn=default_reward,
#         seed=0,
#         isolate_state=True,
#         combine_rounds_default=1,  # keep combine fixed during DPO data creation
#     )

#     print('after answer questions merging')

#     for k, v in cr.meta_codebook.items():
#         if "fact" in k.lower():
#             print(k, ":", v)

#     print(examples)

#     # --- 3) train DPO policy
#     policy, ref = train_dpo_2head(examples, input_dim=384)

#     # --- 4) init LinUCB scheduler with state feature dim
#     d_state = featurize_state(cr).shape[0]  # typically 4
#     scheduler = CombineScheduler(d=d_state, arms=COMBINE_ARMS, alpha=1.0, epsilon=0.05)

#     # --- 5) inference on new questions (+ optional online bandit updates)
#     test_questions = [
#         "What is the tallest mountain in Africa?",
#         # "Explain CRISPR in one sentence.",
#         # "Who wrote Pride and Prejudice?",
#     ]
#     for q in test_questions:
#         pred, meta = answer_with_auto_strategy(
#             cr=cr,
#             policy=policy,
#             scheduler=scheduler,
#             q=q,
#             reward_fn=default_reward,  # if you have gold; else set to None
#             gold_answer=None,          # supply if you have target
#             greedy=True
#         )
#         print(f"\nQ: {q}\nA: {pred}\nmeta: {meta}")
#         # print(cr.cur_fact_context)



#python py_files/dpo_compressrag.py
#python dpo_compressrag.py

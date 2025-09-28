from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable,Any
from contextlib import contextmanager
import math, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from CompressRag_rl_v2 import CompressRag_rl,WordAvgEmbeddings, get_context,merging_codebook
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
from test_for_compressrag import Phi4MiniReasoningLLM
import json
import asyncio,inspect
from sentence_transformers import SentenceTransformer


# ===============================
# 0) GLOBAL CHOICES / MAPPINGS
# ===============================
# THINKINGS_CHOICES = ['overlap','unique','not_include']
# ANSWERS_CHOICES   = ['overlap','unique','not_include']


THINKINGS_CHOICES = ['not_include']
ANSWERS_CHOICES   = ['unique','not_include']
FACTS_CHOICES = ['unique','include_all'] 


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


_ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# _ENCODER = SentenceTransformer("BAAI/bge-base-en")


def featurize_query(q: str, dims: int = 384) -> np.ndarray:
    """
    Return an embedding for query q.
    Uses a sentence-transformer encoder if available;
    otherwise falls back to hashed n-grams.
    """
    try:
        vec = _ENCODER.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0]   # shape (D,)
    except Exception:
        # fallback to hashed n-grams if encoder fails
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
                facts_choice: str,
                isolate_state: bool = True):
    """
    Temporarily set CR's per-question knobs (answers, thinkings).
    State isolation (meta_codebook/round) is optional.
    """
    prev_ans  = getattr(cr, "answers_choice", None)
    prev_th   = getattr(cr, "thinkings_choice", None)
    prev_facts = getattr(cr, "facts_choice", None)

    prev_meta = None

    if isolate_state:
        if hasattr(cr, "meta_codebook"):
            import copy
            prev_meta = copy.deepcopy(cr.meta_codebook)

    cr.answers_choice   = answers_choice
    cr.thinkings_choice = thinkings_choice
    cr.facts_choice = facts_choice

    try:
        yield
    finally:
        if prev_ans is not None:  cr.answers_choice = prev_ans
        if prev_th is not None:   cr.thinkings_choice = prev_th
        if prev_facts is not None:  cr.facts_choice = prev_facts

        if isolate_state:
            if prev_meta is not None: cr.meta_codebook = prev_meta


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
# 4) DPO DATA (3-HEAD: ans/th/facts)
# ===============================
@dataclass
class PrefExample2:
    x: np.ndarray                 # question features
    y_pos: Tuple[int,int,int]         # (ans_idx, th_idx, facts_idx)
    y_neg: Tuple[int,int,int]

# class PrefDataset2(torch.utils.data.Dataset):
#     def __init__(self, examples: List[PrefExample2]):
#         assert len(examples) > 0, "Empty preference dataset."
#         self.examples = examples
#     def __len__(self): return len(self.examples)
#     def __getitem__(self, i):
#         ex = self.examples[i]
#         return (torch.tensor(ex.x, dtype=torch.float32),
#                 torch.tensor(ex.y_pos, dtype=torch.long),
#                 torch.tensor(ex.y_neg, dtype=torch.long))
    
class PrefDataset2(torch.utils.data.Dataset):
    def __init__(self, examples: List[PrefExample2]):
        assert len(examples) > 0, "Empty preference dataset."
        self.examples = examples

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]

        def _ensure_triple(t):
            t = list(t)
            if len(t) == 2: t.append(0)   # pad facts index
            return tuple(t[:3])

        y_pos = _ensure_triple(ex.y_pos)
        y_neg = _ensure_triple(ex.y_neg)

        return (
            torch.tensor(ex.x, dtype=torch.float32),
            torch.tensor(y_pos, dtype=torch.long),  # [3]
            torch.tensor(y_neg, dtype=torch.long),  # [3]
        )
    
def make_preference_dataset_2head(
    cr,
    questions: List[str],
    gold_answers: Optional[Dict[str,str]] = None,
    per_q_samples: int = 6,
    feature_dim: int = 384,
    reward_fn: Callable[[str, Optional[str]], float] = None,
    seed: int = 0,
    isolate_state: bool = True,
    ANSWERS_CHOICES = ANSWERS_CHOICES,
    THINKINGS_CHOICES = THINKINGS_CHOICES,
    FACTS_CHOICES = FACTS_CHOICES,
) -> List[PrefExample2]:
    """
    Build DPO pairs for (answers_choice, thinkings_choice) ONLY.
    """
    if reward_fn is None:
        reward_fn = default_reward

    rng = random.Random(seed)
    all_pairs = [(ai, ti,fi)
                 for ai in range(len(ANSWERS_CHOICES))
                 for ti in range(len(THINKINGS_CHOICES))
                 for fi in range(len(FACTS_CHOICES))]

    examples: List[PrefExample2] = []

    for q in questions:
        x = featurize_query(q, dims=feature_dim)
        tried = rng.sample(all_pairs, k=min(per_q_samples, len(all_pairs)))

        scored: List[Tuple[Tuple[int,int], float]] = []
        for (ai, ti, fi) in tried:
            ans = ANSWERS_CHOICES[ai]
            th  = THINKINGS_CHOICES[ti]
            facts = FACTS_CHOICES[fi]

            with temp_ans_th(cr, ans, th, facts, isolate_state=isolate_state):
                pred = cr.run_work_flow(q)

            score = reward_fn(pred, gold_answers.get(q) if gold_answers else None)
            scored.append(((ai, ti), score))

        if len(scored) < 2:
            continue
        scored.sort(key=lambda z: z[1], reverse=True)
        y_pos, y_neg = scored[0][0], scored[-1][0]
        examples.append(PrefExample2(x=x, y_pos=y_pos, y_neg=y_neg))


    return examples
    

# {'Who discovered penicillin?': {'input_tokens': 191.0, 'output_tokens': 165.0, 'total_tokens': 356.0, 'latency_sec': 4.346117499982938, 'gen_latency_sec': 4.344727099989541, 'retrieval_latency_sec': 0.003563100006431341, 'peak_vram_MiB': 14869.89306640625, 'prompt_chars': 863.0, 'throughput_tok_per_s': 37.977068801489786, 'prompt_tok_per_s': 137370.54150389606, 'device': 'cuda:0', 'dtype': 'torch.bfloat16', 'model_name': 'microsoft/Phi-4-mini-reasoning', 'temperature': 0.2, 'top_p': 0.9, 'max_new_tokens': 512, 'timestamp_start': 785499.1458307, 'timestamp_end': 785503.4919482, 'attempt': 1, 'question_chars': 26.0, 'answer_raw_chars': 810.0, 'answer_raw_tokens': 164.0, 'prompt_to_output_char_ratio': 1.065432098765432}}

async def _call_reward_fn(
    reward_fn: Callable,
    question: str,
    pred: str,
    gold: Optional[str],
    llm,
    embeddings,
):
    # normalize None -> "" to satisfy embedding/LMM calls
    gold = gold or ""
    if inspect.iscoroutinefunction(reward_fn):
        return await reward_fn(question, pred, gold, llm, embeddings)
    else:
        return reward_fn(question, pred, gold, llm, embeddings)

async def make_preference_dataset_2head_using_llm(
    cr,
    questions: List[str],
    gold_answers: Optional[dict] = None,
    per_q_samples: int = 6,
    feature_dim: int = 384,
    reward_fn: Optional[Callable] = None,  # pass compute_answer_correctness here
    seed: int = 0,
    isolate_state: bool = True,
    ANSWERS_CHOICES = ANSWERS_CHOICES,
    THINKINGS_CHOICES = THINKINGS_CHOICES,
    FACTS_CHOICES = FACTS_CHOICES,
    llm=None,
    embeddings=None,
) -> List["PrefExample2"]:
    """
    Build DPO pairs for (answers_choice, thinkings_choice) ONLY.
    Runs async reward function correctly (awaited).
    """
    if reward_fn is None:
        # fallback to your synchronous default if desired
        reward_fn = default_reward

    gold_answers = gold_answers or {}
    rng = random.Random(seed)
    all_pairs = [(ai, ti,fi)
                 for ai in range(len(ANSWERS_CHOICES))
                 for ti in range(len(THINKINGS_CHOICES))
                 for fi in range(len(FACTS_CHOICES))]

    examples: List["PrefExample2"] = []

    for q in questions:
        x = featurize_query(q, dims=feature_dim)
        tried = rng.sample(all_pairs, k=min(per_q_samples, len(all_pairs)))

        # (optional) parallelize scoring for this question
        tasks = []
        meta = []  # keep (ai, ti) aligned with tasks
        for (ai, ti, fi) in tried:
            ans = ANSWERS_CHOICES[ai]
            th  = THINKINGS_CHOICES[ti]
            facts = FACTS_CHOICES[fi]

            with temp_ans_th(cr, ans, th, facts, isolate_state=isolate_state):
                pred, metrics_from_llm, ft_txt = cr.run_work_flow_for_dpo(q)

            tasks.append(_call_reward_fn(
                reward_fn, q, pred, gold_answers.get(q), llm, embeddings
            ))
            meta.append((ai, ti, fi))

        # await all scores
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        scored: List[Tuple[Tuple[int,int], float]] = []
        for (ai_ti_fi, s) in zip(meta, scores):
            if isinstance(s, Exception) or s is None:
                # skip failures or None
                continue
            try:
                scored.append((ai_ti_fi, float(s)))
            except (TypeError, ValueError):
                # skip non-numeric results
                continue

        if len(scored) < 2:
            continue

        scored.sort(key=lambda z: z[1], reverse=True)
        y_pos, y_neg = scored[0][0], scored[-1][0]
        examples.append(PrefExample2(x=x, y_pos=y_pos, y_neg=y_neg))

    return examples

def _pref2_to_dict(e) -> Dict[str, Any]:
    return {
        "x": np.asarray(e.x, dtype=float).tolist(),   # vector -> list
        "y_pos": list(e.y_pos),                       # (ai, ti) -> [ai, ti]
        "y_neg": list(e.y_neg),
    }

# def _pref2_from_dict(d):
#     # Replace with your real PrefExample2(..) if the signature differs
#     return PrefExample2(
#         x=np.array(d["x"], dtype=np.float32),
#         y_pos=tuple(d["y_pos"]),
#         y_neg=tuple(d["y_neg"]),
#     )

def _pref2_from_dict(d):
    def _ensure_triple(t):
        t = list(t)
        if len(t) == 2: t.append(0)
        return tuple(t[:3])
    return PrefExample2(
        x=np.array(d["x"], dtype=np.float32),
        y_pos=_ensure_triple(d["y_pos"]),
        y_neg=_ensure_triple(d["y_neg"]),
    )

def save_pref_examples(path: str, examples: List["PrefExample2"]) -> None:
    """Save ONLY the examples list to a JSON file."""
    payload = [_pref2_to_dict(e) for e in examples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def load_pref_examples(path: str) -> List["PrefExample2"]:
    """Load ONLY the examples list from a JSON file you saved before."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [_pref2_from_dict(row) for row in payload]


# ===============================
# 5) DPO POLICY 
# ===============================
class StrategyPolicy2Head(nn.Module):
    """MLP with two categorical heads: answers, thinkings."""
    def __init__(self, input_dim: int, hidden: int = 512, drop: float = 0.1,ANSWERS_CHOICES = ANSWERS_CHOICES, THINKINGS_CHOICES = THINKINGS_CHOICES,FACTS_CHOICES= FACTS_CHOICES):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(drop),
        )
        self.ans = nn.Linear(hidden, len(ANSWERS_CHOICES))
        self.th  = nn.Linear(hidden, len(THINKINGS_CHOICES))
        self.facts  = nn.Linear(hidden, len(FACTS_CHOICES))

    def forward(self, x):  # x: [B,D]
        h = self.ff(x)
        return self.ans(h), self.th(h), self.facts(h)


    def log_prob(self, x, y):  # y: [B,3] longs
        la, lt, lf = self.forward(x)
        logpa = F.log_softmax(la, dim=-1).gather(-1, y[:,0:1]).squeeze(-1)
        logpt = F.log_softmax(lt, dim=-1).gather(-1, y[:,1:2]).squeeze(-1)
        logpf = F.log_softmax(lf, dim=-1).gather(-1, y[:,2:3]).squeeze(-1)
        return logpa + logpt + logpf

    @torch.no_grad()
    def sample(self, x, greedy: bool = True):
        la, lt, lf = self.forward(x)
        if greedy:
            ya, yt, yf = la.argmax(-1), lt.argmax(-1), lf.argmax(-1)
        else:
            ya = torch.distributions.Categorical(logits=la).sample()
            yt = torch.distributions.Categorical(logits=lt).sample()
            yf = torch.distributions.Categorical(logits=lf).sample()
        return torch.stack([ya, yt, yf], dim=-1)

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
# 6) INFERENCE PIPELINE
# ===============================
@torch.no_grad()
def select_ans_th(policy: StrategyPolicy2Head, cr, q: str, feature_dim: int = 384, greedy: bool = True,  ANSWERS_CHOICES = ANSWERS_CHOICES,THINKINGS_CHOICES = THINKINGS_CHOICES,FACTS_CHOICES = FACTS_CHOICES):
    x = torch.tensor(featurize_query(q, dims=feature_dim),dtype=torch.float32).unsqueeze(0).to(next(policy.parameters()).device)
    y = policy.sample(x, greedy=greedy)[0].cpu().numpy().tolist()
    ai, ti, fi = int(y[0]), int(y[1]), int(y[2])
    return (ANSWERS_CHOICES[ai], THINKINGS_CHOICES[ti], FACTS_CHOICES[fi])

def answer_with_auto_strategy(
    cr: CompressRag_rl,
    policy: StrategyPolicy2Head,
    q: str,
    reward_fn: Callable[[str, Optional[str]], float] = None,
    gold_answer: Optional[str] = None,
    greedy: bool = True,
    ANSWERS_CHOICES = ANSWERS_CHOICES,
    THINKINGS_CHOICES = THINKINGS_CHOICES,
    FACTS_CHOICES = FACTS_CHOICES
) -> Tuple[str, Dict[str, object]]:
    """
    1) Choose answers/th via DPO policy (question features)
    2) Run CR once and return result (+ metadata)
    3) If reward_fn & gold provided, update scheduler online
    """
    # 1) per-question knobs
    ans_choice, th_choice, facts_choice = select_ans_th(policy, cr, q, greedy=greedy,ANSWERS_CHOICES = ANSWERS_CHOICES,THINKINGS_CHOICES = THINKINGS_CHOICES, FACTS_CHOICES = FACTS_CHOICES)

    # 2) run
    with temp_ans_th(cr, ans_choice, th_choice,facts_choice, isolate_state=False):
        pred = cr.run_work_flow(q)
        fact_context =  cr.cur_fact_context

    # 3) update bandit online if evaluable
    reward = None
    if reward_fn is not None:
        try:
            reward = float(reward_fn(pred, gold_answer))
        except Exception:
            reward = None
    if reward is not None:
        # optionally maintain a rolling average on CR
        try:
            rr = getattr(cr, "_recent_reward_avg", 0.0)
            cr._recent_reward_avg = 0.9*float(rr) + 0.1*reward
        except Exception:
            pass

    meta = {
        "answers_choice": ans_choice,
        "thinkings_choice": th_choice,
        "facts_choice": facts_choice,
        "reward": reward,
        "fact_context": fact_context,
    }
    return pred, meta


# dpo_exactgraphrag.py
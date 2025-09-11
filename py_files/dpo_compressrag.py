from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from contextlib import contextmanager
import random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from CompressRag_rl_v1 import CompressRag_rl


# Choice spaces & mappings
THINKINGS_CHOICES = ['overlap','unique','not_include']
ANSWERS_CHOICES   = ['overlap','unique','not_include']
COMBINE_CHOICES   = [0,1,2]   # your discrete choice space

TH2I = {v:i for i,v in enumerate(THINKINGS_CHOICES)}
AN2I = {v:i for i,v in enumerate(ANSWERS_CHOICES)}
CB2I = {v:i for i,v in enumerate(COMBINE_CHOICES)}
I2TH = {i:v for v,i in TH2I.items()}
I2AN = {i:v for v,i in AN2I.items()}
I2CB = {i:v for v,i in CB2I.items()}

# If CR uses `combine_ents_rounds`, map discrete choices -> cadence in rounds
COMB_DISCRETE_TO_ROUNDS = {0: 0, 1: 1, 2: 3}


# Minimal featurization
def _hashed_char_ngrams(s: str, dims: int = 384, n: int = 3) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    s = s or ""
    for i in range(len(s)-n+1):
        v[hash(s[i:i+n]) % dims] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def featurize_query(cr, q: str, dims: int = 384) -> np.ndarray:
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


# temporary strategy
@contextmanager
def temp_strategy(cr,
                  answers_choice: str,
                  thinkings_choice: str,
                  combine_choice: int,
                  isolate_state: bool = True):
    """
    Temporarily set CR's knobs. If isolate_state=True, snapshot/restore meta to avoid leakage.
    Works whether CR exposes `combine_ents_choice` OR `combine_ents_rounds`.
    """
    # snapshot
    prev_ans  = getattr(cr, "answers_choice", None)
    prev_th   = getattr(cr, "thinkings_choice", None)
    prev_comb_rounds = getattr(cr, "combine_ents_rounds", None)
    prev_comb_choice = getattr(cr, "combine_ents_choice", None)
    prev_meta = None
    prev_round_idx = None

    if isolate_state:
        if hasattr(cr, "meta_codebook"):
            import copy
            prev_meta = copy.deepcopy(cr.meta_codebook)
        if hasattr(cr, "_round_idx"):
            prev_round_idx = cr._round_idx

    # set
    cr.answers_choice   = answers_choice
    cr.thinkings_choice = thinkings_choice
    if hasattr(cr, "combine_ents_choice"):
        cr.combine_ents_choice = combine_choice
    elif hasattr(cr, "combine_ents_rounds"):
        cr.combine_ents_rounds = COMB_DISCRETE_TO_ROUNDS[combine_choice]

    try:
        yield
    finally:
        # restore
        if prev_ans is not None:  cr.answers_choice = prev_ans
        if prev_th is not None:   cr.thinkings_choice = prev_th
        if prev_comb_choice is not None: cr.combine_ents_choice = prev_comb_choice
        if prev_comb_rounds is not None: cr.combine_ents_rounds = prev_comb_rounds
        if isolate_state:
            if prev_meta is not None:     cr.meta_codebook = prev_meta
            if prev_round_idx is not None: cr._round_idx = prev_round_idx



# Preference dataset
@dataclass
class PrefExample:
    x: np.ndarray                 # features for the prompt (shared in the pair)
    y_pos: Tuple[int,int,int]     # (ans_idx, th_idx, comb_idx)
    y_neg: Tuple[int,int,int]

class PrefDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[PrefExample]):
        assert len(examples) > 0, "Empty preference dataset."
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex = self.examples[i]
        return (torch.tensor(ex.x, dtype=torch.float32),
                torch.tensor(ex.y_pos, dtype=torch.long),
                torch.tensor(ex.y_neg, dtype=torch.long))

def make_preference_dataset(
    cr, questions: List[str],
    gold_answers: Optional[Dict[str,str]] = None,
    per_q_samples: int = 6,
    feature_dim: int = 384,
    reward_fn: Callable[[str, Optional[str]], float] = None,
    seed: int = 0,
    isolate_state: bool = True
) -> List[PrefExample]:
    """
    For each q: sample several (ans,think,comb) variants, run CR once per variant,
    score, and keep (best, worst) as DPO pair.
    """
    if reward_fn is None:
        reward_fn = default_reward

    rng = random.Random(seed)
    all_triples = [(ai, ti, ci)
                   for ai in range(len(ANSWERS_CHOICES))
                   for ti in range(len(THINKINGS_CHOICES))
                   for ci in range(len(COMBINE_CHOICES))]

    examples: List[PrefExample] = []
    for q in questions:
        x = featurize_query(cr, q, dims=feature_dim)
        tried = rng.sample(all_triples, k=min(per_q_samples, len(all_triples)))

        scored: List[Tuple[Tuple[int,int,int], float]] = []
        for (ai, ti, ci) in tried:
            ans = ANSWERS_CHOICES[ai]
            th  = THINKINGS_CHOICES[ti]
            cb  = COMBINE_CHOICES[ci]
            try:
                with temp_strategy(cr, ans, th, cb, isolate_state=isolate_state):
                    pred = cr.run_work_flow(q)   # no control tokens needed
            except Exception:
                continue
            score = reward_fn(pred, gold_answers.get(q) if gold_answers else None)
            scored.append(((ai, ti, ci), score))

        if len(scored) < 2:
            continue
        scored.sort(key=lambda z: z[1], reverse=True)
        y_pos, y_neg = scored[0][0], scored[-1][0]
        examples.append(PrefExample(x=x, y_pos=y_pos, y_neg=y_neg))
    return examples

# reward (could be replaced to all evaluation functions)
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
    # light brevity encouragement
    toks = len((pred_answer or "").split())
    return base - 0.0005*max(0, toks-256)


# Strategy policy (3-head)
class StrategyPolicy(nn.Module):
    """Independent heads over answers/thinkings/combine."""
    def __init__(self, input_dim: int, hidden: int = 512, drop: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(drop),
        )
        self.ans = nn.Linear(hidden, len(ANSWERS_CHOICES))
        self.th  = nn.Linear(hidden, len(THINKINGS_CHOICES))
        self.cb  = nn.Linear(hidden, len(COMBINE_CHOICES))

    def forward(self, x):  # x: [B,D]
        h = self.ff(x)
        return self.ans(h), self.th(h), self.cb(h)

    @torch.no_grad()
    def log_prob(self, x, y):  # y: [B,3] longs
        la, lt, lc = self.forward(x)
        logpa = F.log_softmax(la, dim=-1).gather(-1, y[:,0:1]).squeeze(-1)
        logpt = F.log_softmax(lt, dim=-1).gather(-1, y[:,1:2]).squeeze(-1)
        logpc = F.log_softmax(lc, dim=-1).gather(-1, y[:,2:3]).squeeze(-1)
        return logpa + logpt + logpc

    @torch.no_grad()
    def sample(self, x, greedy: bool = True):
        la, lt, lc = self.forward(x)
        if greedy:
            ya, yt, yc = la.argmax(-1), lt.argmax(-1), lc.argmax(-1)
        else:
            ya = torch.distributions.Categorical(logits=la).sample()
            yt = torch.distributions.Categorical(logits=lt).sample()
            yc = torch.distributions.Categorical(logits=lc).sample()
        return torch.stack([ya, yt, yc], dim=-1)


# DPO objective & training
def dpo_loss(policy: StrategyPolicy,
             ref: StrategyPolicy,
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

def train_dpo(examples: List[PrefExample], input_dim: int, cfg: TrainCfg = TrainCfg()
             ) -> Tuple[StrategyPolicy, StrategyPolicy]:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    ds = PrefDataset(examples)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    policy = StrategyPolicy(input_dim=input_dim).to(cfg.device)
    ref    = StrategyPolicy(input_dim=input_dim).to(cfg.device)
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for e in range(cfg.epochs):
        policy.train()
        tot, nb = 0.0, 0
        for x, y_pos, y_neg in dl:
            x, y_pos, y_neg = x.to(cfg.device), y_pos.to(cfg.device), y_neg.to(cfg.device)
            loss = dpo_loss(policy, ref, x, y_pos, y_neg, beta=cfg.beta)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"[DPO] epoch {e+1}/{cfg.epochs} loss={tot/max(1,nb):.4f}")

    policy.eval()
    return policy, ref


# Inference helper
@torch.no_grad()
def select_strategy(policy: StrategyPolicy, cr, q: str, feature_dim: int = 384, greedy: bool = True):
    x = torch.tensor(featurize_query(cr, q, dims=feature_dim), dtype=torch.float32).unsqueeze(0)
    y = policy.sample(x, greedy=greedy)[0].cpu().numpy().tolist()
    ai, ti, ci = int(y[0]), int(y[1]), int(y[2])
    return (ANSWERS_CHOICES[ai], THINKINGS_CHOICES[ti], COMBINE_CHOICES[ci])


# usage workflow

# examples = make_preference_dataset(
#     cr, questions=train_questions, gold_answers=gold_dict,
#     per_q_samples=6, feature_dim=384, seed=0, isolate_state=True
# )

# # Train DPO
# policy, ref = train_dpo(examples, input_dim=384)

# # Deploy
# ans_choice, th_choice, comb_choice = select_strategy(policy, cr, q="New query here")
# with temp_strategy(cr, ans_choice, th_choice, comb_choice, isolate_state=False):
#     final_answer = cr.run_work_flow("New query here")

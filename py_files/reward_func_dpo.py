from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
from sentence_transformers import SentenceTransformer

### all the reward func takes the pred and the gold_answer (ground truth), output reward value

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


def reward_token_f1(pred: str, gold_answer: str) -> float:
    """
    Reward = token-level F1 score between prediction and gold answer
    """
    from collections import Counter
    import re
    def normalize(s): 
        return re.sub(r"\W+", " ", s.lower()).strip()
    p_toks, g_toks = normalize(pred).split(), normalize(gold_answer).split()
    pc, gc = Counter(p_toks), Counter(g_toks)
    overlap = sum((pc & gc).values())
    if overlap == 0: 
        return 0.0
    precision = overlap / max(1, len(p_toks))
    recall = overlap / max(1, len(g_toks))
    return 2 * precision * recall / (precision + recall)


def reward_bleu(pred: str, gold_answer: str) -> float:
    """
    Reward = sentence BLEU score (0–1), n-gram overlap with smoothing
    """
    smooth = SmoothingFunction().method1
    return sentence_bleu([gold_answer.split()], pred.split(), smoothing_function=smooth)


def reward_rouge1(pred: str, gold_answer: str) -> float:
    """
    Reward = ROUGE-1 F1 score (0–1), unigram overlap
    """
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return scorer.score(gold_answer, pred)["rouge1"].fmeasure


def reward_rouge2(pred: str, gold_answer: str) -> float:
    """
    Reward = ROUGE-2 F1 score (0–1), bigram overlap
    """
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    return scorer.score(gold_answer, pred)["rouge2"].fmeasure

def reward_rougeL(pred: str, gold_answer: str) -> float:
    """
    Reward = ROUGE-L F1 score (0–1), LCS-based overlap
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gold_answer, pred)["rougeL"].fmeasure

def reward_bertscore(pred: str, gold_answer: str, model_name: str = "microsoft/deberta-xlarge-mnli") -> float:
    """
    Reward = BERTScore F1 (semantic similarity using embeddings)
    """
    P, R, F1 = score([pred], [gold_answer], model_type=model_name)
    return float(F1.mean())

def reward_sbert(pred: str, gold_answer: str, model=None) -> float:
    """
    Reward = cosine similarity between SBERT embeddings (0–1)
    """
    model = model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_pred, emb_gold = model.encode([pred, gold_answer])
    emb_pred /= (np.linalg.norm(emb_pred) + 1e-9)
    emb_gold /= (np.linalg.norm(emb_gold) + 1e-9)
    return float((emb_pred * emb_gold).sum())
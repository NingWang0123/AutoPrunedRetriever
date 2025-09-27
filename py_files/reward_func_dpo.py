from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import re
from dpo_compressrag_v2 import answer_with_auto_strategy

### all the reward func takes the pred and the gold_answer (ground truth), output reward value

def _tokset(s: str) -> set:
    return set((s or "").lower().split())

def f1_overlap(pred: str, gold: str) -> float:
    P, G = _tokset(pred), _tokset(gold)
    if not P and not G: return 1.0
    if not P or not G:  return 0.0
    tp = len(P & G); prec = tp/(len(P)+1e-8); rec = tp/(len(G)+1e-8)
    return 0.0 if prec+rec == 0 else 2*prec*rec/(prec+rec)

def default_reward(pred_answer: str, gold_answer:str) -> float:
    base = f1_overlap(pred_answer, gold_answer or "")
    toks = len((pred_answer or "").split())
    return base - 0.0005*max(0, toks-256)


def reward_token_f1(pred: str, gold_answer: str) -> float:
    """
    Reward = token-level F1 score between prediction and gold answer
    """
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


def reward_sbert_inclusive(pred: str, gold_answer: str, model=None, threshold: float = 0.85) -> float:
    """
    Reward = 1.0 if the gold_answer is semantically included in pred (similarity >= threshold).
             Otherwise, cosine similarity between SBERT embeddings (0–1).

    pred: system prediction
    gold_answer: reference answer
    model: optional pre-loaded SentenceTransformer
    threshold: similarity cutoff for "full credit"
    """

    model = model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode both separately
    emb_pred, emb_gold = model.encode([pred, gold_answer])
    emb_pred /= (np.linalg.norm(emb_pred) + 1e-9)
    emb_gold /= (np.linalg.norm(emb_gold) + 1e-9)

    cosine_sim = float((emb_pred * emb_gold).sum())

    # If gold is semantically included in pred, give full credit
    if cosine_sim >= threshold:
        return 1.0
    else:
        return cosine_sim



def evaluation_for_correctness(question, gold_answer, rag, policy=None, work_mode="normal", eval_func = reward_sbert):
    if work_mode == "dpo":
        result = answer_with_auto_strategy(cr =rag, 
                                            policy =policy, 
                                            q = question,
                                            reward_fn       = None,
                                            gold_answer     = None,
                                            greedy          = True) # evaluation don't need groundtruth and reward_fn
    else:
        result = rag.run_work_flow(question)
    if isinstance(result, tuple):
        pred = result[0]
    else:
        pred = str(result)

    eval_result = eval_func(pred,gold_answer)

    return eval_result


def evaluation_for_context(question, context_ground_truth, rag, policy=None, eval_func = reward_sbert):
    result,_meta = answer_with_auto_strategy(cr =rag, 
                                        policy =policy, 
                                        q = question,
                                        reward_fn       = None,
                                        gold_answer     = None,
                                        greedy          = True) # evaluation don't need groundtruth and reward_fn


    eval_result = eval_func(_meta['fact_context'],context_ground_truth)

    return eval_result


def evaluation_for_correctness_and_context(question, gold_answer,context_ground_truth, rag, policy=None, eval_func = reward_sbert):
    result,_meta = answer_with_auto_strategy(cr =rag, 
                                        policy =policy, 
                                        q = question,
                                        reward_fn       = None,
                                        gold_answer     = None,
                                        greedy          = True) # evaluation don't need groundtruth and reward_fn
    
    if isinstance(result, tuple):
        pred = result[0]
    else:
        pred = str(result)


    eval_result_context = eval_func(_meta['fact_context'],context_ground_truth)
    eval_result_correctness = eval_func(pred,gold_answer)

    return eval_result_correctness,eval_result_context


def evaluation_for_correctness_and_context_for_giving_results(all_generated_dicts,pred,ground_truth,context,ground_truth_context, eval_func = reward_sbert):

    eval_result_correctness_lst = []
    eval_result_context_lst = []


    def make_sure_str(result):
        if isinstance(result, tuple):
            return result[0]
        else:
            return str(result)

    for generated_dict in all_generated_dicts:

        context_pred_str = make_sure_str(generated_dict[context])
        ground_truth_context_str = make_sure_str(generated_dict[ground_truth_context])
        pred_str = make_sure_str(generated_dict[pred])
        ground_truth_str =make_sure_str(generated_dict[ground_truth])


        eval_result_context = eval_func(context_pred_str,ground_truth_context_str)
        eval_result_correctness = eval_func(pred_str,ground_truth_str)
        eval_result_correctness_lst.append(eval_result_correctness)
        eval_result_context_lst.append(eval_result_context)

    return eval_result_correctness_lst,eval_result_context_lst




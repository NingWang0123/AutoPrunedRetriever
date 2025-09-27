import asyncio
import os, json, re, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pathlib
import numpy   as np
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings

from CompressRag_rl_v3 import (
    CompressRag_rl, merging_codebook
)

from dpo_compressrag_v3 import (    
    make_preference_dataset_2head, train_dpo_2head,make_preference_dataset_2head_using_llm,
    default_reward, answer_with_auto_strategy,save_pref_examples,load_pref_examples,ANSWERS_CHOICES,THINKINGS_CHOICES
)

from test_for_compressrag import Phi4MiniReasoningLLM,Word2VecEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
# from evaluation_func_graphrag import compute_answer_correctness
import reward_func_dpo
from functools import partial

# ---------------------------------------------------------------------
# 0) Paths / constants
# ---------------------------------------------------------------------
REPO_ID      = "GraphRAG-Bench/GraphRAG-Bench"
CORPUS_FILE  = "Datasets/Corpus/medical.json"
QUEST_FILE   = "Datasets/Questions/medical_questions.json"

SEED_N       = 50    # first 30 rows → bootstrap + DPO train
TEST_N       = 950     # next 20 rows  → evaluation
TOPK_CTX     = 5

# ---------------------------------------------------------------------
# 1) Initialise embeddings & LLM
# ---------------------------------------------------------------------

def compress_rag_workflow(REPO_ID,CORPUS_FILE,QUEST_FILE,SEED_N,TEST_N,
                          top_m,top_k,combine_ent_sim,q_combine_sim,aft_combine_sim,  # all the params can be optimized
                          ini_meta_json = Path("meta_codebook.json") ,saved_examples_name = "sbert_pref_examples_medical.json",
                          reward_func = None,reward_func_mode = 'non_llm',final_csv_path = "results/sbert_result"):
    print("» Initialising embeddings & LLM …")
    word_emb = Word2VecEmbeddings(model_name="word2vec-google-news-300")
    sent_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en"
    )
    phi_llm  = Phi4MiniReasoningLLM(
        include_thinkings=False,
        model_name="Qwen/Qwen2.5-3B",
        max_new_tokens=1500,
        temperature=0.2,
        top_p=0.9,
    )


    if ini_meta_json:
        pre_loaded_meta = False

        if ini_meta_json.is_file():
            try:
                with ini_meta_json.open("r", encoding="utf-8") as f:
                    ini = json.load(f)
                    pre_loaded_meta = True
            except json.JSONDecodeError:
                print("[warn] meta_codebook.json is not valid JSON; starting fresh.")
                ini = {}
                pre_loaded_meta = False
        else:
            ini = {}
    else:
        ini = {}


    cr = CompressRag_rl(
        ini_meta_codebook = ini,
        sentence_emb      = sent_emb,
        word_emb          = word_emb,
        llm               = phi_llm,
        thinkings_choice    = 'not_include',
        answers_choice      = 'overlap',
        top_m = top_m,
        top_k = top_k,
        combine_ent_sim = combine_ent_sim,
        q_combine_sim = q_combine_sim,
        aft_combine_sim = aft_combine_sim,
    )

    # ---------------------------------------------------------------------
    # 2) Load benchmark Q-A-E data
    # ---------------------------------------------------------------------
    print("» Loading benchmark questions / answers / evidence …")
    q_fp = hf_hub_download(REPO_ID, QUEST_FILE, repo_type="dataset")
    qrows = json.load(open(q_fp, encoding="utf-8"))

    row_lookup  = {r["question"].strip(): r for r in qrows}
    gold_lookup = {q: r["answer"]        for q, r in row_lookup.items()}

    all_questions  = list(row_lookup.keys())
    seed_questions = all_questions[:SEED_N]
    test_questions = all_questions[SEED_N : SEED_N+TEST_N]

    # ---------------------------------------------------------------------
    # 3) Pre-load corpus as facts into CR
    # ---------------------------------------------------------------------
    # only load if we do not have ini_meta 
    if not pre_loaded_meta:
        # corpus file is the facts
        facts_cb = cr.load_and_merge_facts(CORPUS_FILE, chunk_chars=100, overlap=30)
        cr._facts_preloaded = True
        cr.top_m = 2          # sentence-embedding rerank top-m

        cr.meta_codebook = merging_codebook(
            cr.meta_codebook, facts_cb,
            type='facts', word_emb=cr.word_emb, use_thinkings=True
        )
        print(cr.meta_codebook)

        def make_json_safe(obj):
            """Recursively convert numpy arrays into lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            return obj

        with open("meta_codebook.json", "w") as f:
            json.dump(make_json_safe(cr.meta_codebook), f, indent=2)


        print(f"[DEBUG] after facts-merge: |E|={len(cr.meta_codebook['e'])} "
            f"|R|={len(cr.meta_codebook['r'])} "
            f"|edges|={len(cr.meta_codebook['edge_matrix'])}")
        

    # ---------------------------------------------------------------------
    # 4) Build DPO preference dataset on seed Q-A pairs
    # ---------------------------------------------------------------------
    # print(cr.meta_codebook)
    print("» Building preference pairs for DPO …")


    # # using llm one to replace the old one
    # # using answer_correctness from graph rag benchmark
    if os.path.exists(saved_examples_name):
        pref_ds = load_pref_examples(saved_examples_name)
        print(f"loaded {len(pref_ds)} cached preference examples")

    else:
        if reward_func_mode == 'llm':
            # check if it is saved or not, reuse the trained one
            async def build_or_load_pref_ds() -> list:
                # init only when needed
                embedding_for_reward = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-large-en-v1.5"
                )
                BASE_URL = "https://api.deepseek.com/v1"
                API_KEY = pathlib.Path("deepseek_key.txt").read_text().strip()

                llm = ChatOpenAI(
                    model="deepseek-chat",
                    base_url=BASE_URL,
                    api_key=API_KEY,
                    temperature=0.0,
                    max_retries=3,
                    timeout=30,
                )

                pref_ds = await make_preference_dataset_2head_using_llm(
                    cr=cr,
                    questions=seed_questions,
                    gold_answers=gold_lookup,
                    per_q_samples=6,
                    reward_fn=compute_answer_correctness,   
                    seed=42,
                    llm=llm,
                    embeddings=embedding_for_reward,
                    ANSWERS_CHOICES=ANSWERS_CHOICES,
                    THINKINGS_CHOICES=THINKINGS_CHOICES,
                    isolate_state = True,
                    feature_dim = 384
                )
                print(f"   generated {len(pref_ds)} preference examples")
                save_pref_examples(saved_examples_name, pref_ds)
                return pref_ds
            
            pref_ds = asyncio.run(build_or_load_pref_ds())
            
        else:

            pref_ds = make_preference_dataset_2head(
                    cr,
                    questions= seed_questions,
                    gold_answers=gold_lookup,
                    per_q_samples = 6,
                    feature_dim = 384,
                    reward_fn = reward_func,
                    seed = 0,
                    isolate_state = True,
                    ANSWERS_CHOICES = ANSWERS_CHOICES,
                    THINKINGS_CHOICES = THINKINGS_CHOICES,

                )
            


    policy, _ = train_dpo_2head(pref_ds, input_dim=384)

    def dump_results(
        questions: List[str],
        out_path: str,
        metrics_path: str
    ):
        rows = []
        run_metrics = []

        for q in questions:
            start_idx = len(cr.llm.metrics_runs)
            pred, _meta = answer_with_auto_strategy(
                cr, policy, q,
                reward_fn       = default_reward,
                gold_answer     = gold_lookup[q],
                greedy          = True
            )

            gen_metrics = (cr.llm.metrics_runs[start_idx:] or [{}])[-1]
            run_metrics.append({"question": q, **gen_metrics})    
        
            row = row_lookup[q]
            rows.append({
                "id":               row["id"],
                "question":         q,
                "source":           row["source"],
                "context":          _meta['fact_context'],
                "evidence":         row["evidence"],
                "question_type":    row["question_type"],
                "generated_answer": pred,
                "ground_truth":     row["answer"],
            })

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
            print(f"✓ wrote {len(rows)} rows → {out_path}")

        with open(metrics_path, "w") as f:
            json.dump({"run_meta": run_metrics}, f, indent=2)
        print(f"✓ wrote metrics         → {metrics_path}")

        return rows


    print("» Answering evaluation questions …")
    generated_rows = dump_results(test_questions, out_path= "results/compressrag_medical_data.json", metrics_path ="results/compressrag_medical_metrics.json")

    eval_result_correctness_lst,eval_result_context_lst = reward_func_dpo.evaluation_for_correctness_and_context_for_giving_results(generated_rows,"generated_answer","ground_truth",
                                                                                                                                    "context","evidence",
                                                                                                                                      eval_func = partial(reward_func,model = sent_emb))
    
    df = pd.DataFrame({
    "correctness": eval_result_correctness_lst,
    "context": eval_result_context_lst
    })

    print(df)

    df.to_csv(final_csv_path)

    return df


if __name__ == "__main__":

    aft_combine_sim = 0.9
    top_m = 10

    reward_func = reward_func_dpo.reward_sbert_inclusive

    df = compress_rag_workflow(REPO_ID,CORPUS_FILE,QUEST_FILE,SEED_N,TEST_N, 
                            top_m,top_m*10,aft_combine_sim,aft_combine_sim,aft_combine_sim,
                            Path("meta_codebook.json") ,f"pref_examples_medical_ss_new_embed.json",reward_func,
                            reward_func_mode = 'non_llm',final_csv_path = f"results/{str(reward_func.__name__)}_result_new_embed")

    df.to_csv('results/result_sbertinclusive_new_embed.csv')

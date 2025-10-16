# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.



import os
import sys
import json
import tempfile
import asyncio
import numpy as np
import re
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append("/Users/lancelotchu/Desktop/autoprune")

from pipeline_for_autopruned_openai_new_test_ver_for_v3 import ExactGraphRag_rl, answer_with_auto_strategy, default_reward
from dpo_exactgraphrag import (
    make_preference_dataset_2head, train_dpo_2head, save_pref_examples, load_pref_examples,
    ANSWERS_CHOICES, THINKINGS_CHOICES, FACTS_CHOICES
)
import reward_func_dpo
from langchain_community.embeddings import HuggingFaceEmbeddings
from llm_api import OpenAILLM

class UserModel:
    def __init__(self):
        print("» Initialising CRAG-integrated RAG pipeline with full DPO training …")
        
        self.top_m = 10  
        self.top_k = 100  
        self.aft_combine_sim = 0.9  
        self.semantic_overlap_sim = 0.95  
        
        self.word_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.sent_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.api_llm = OpenAILLM(
            include_thinkings=True,
            model_name="gpt-4o-mini",
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
            api_key="",
        )

        self.meta_codebook_path = Path("/Users/lancelotchu/Desktop/autoprune/meta_codebook_new.json")  
        self.pref_examples_path = "/Users/lancelotchu/Desktop/autoprune/pref_examples_medical_exact_openai_v3.json"  
        
        if self.meta_codebook_path.is_file():
            try:
                with self.meta_codebook_path.open("r", encoding="utf-8") as f:
                    ini = json.load(f)
                    self.pre_loaded_meta = True
                    print("» Loaded existing meta_codebook from disk")
            except json.JSONDecodeError:
                print("[warn] meta_codebook.json is not valid JSON; starting fresh.")
                ini = {}
                self.pre_loaded_meta = False
        else:
            ini = {}
            self.pre_loaded_meta = False

        self.cr = ExactGraphRag_rl(
            ini_meta_codebook=ini,
            sentence_emb=self.sent_emb,
            word_emb=self.word_emb,
            llm=self.api_llm,
            thinkings_choice='not_include',
            answers_choice='unique',
            facts_choice='include_all',
            top_m=self.top_m,
            top_k=self.top_k,
            combine_ent_sim=self.aft_combine_sim,
            q_combine_sim=self.aft_combine_sim,
            aft_combine_sim=self.aft_combine_sim,
            semantic_overlap_sim=self.semantic_overlap_sim,
            use_word=True
        )

        self.policy = self._initialize_policy()
        
        # Initialize sentence transformer for reward calculation (like pipeline)
        self._sbert_model = None
        
        # Results storage (like pipeline)
        self.results_path = "/Users/lancelotchu/Desktop/autoprune/CRAG-main/results/crag_compressrag_results.json"
        self.all_results = []
        
        print("» UserModel initialization complete")

    def _initialize_policy(self):
        

        if os.path.exists(self.pref_examples_path):
            print(f"» Loading cached preference examples from {self.pref_examples_path}")
            pref_ds = load_pref_examples(self.pref_examples_path)
            print(f"   loaded {len(pref_ds)} cached preference examples")
        else:
            print("» No cached preference examples found, will generate during first batch")
    
            return None
            
        print("» Training DPO policy...")
        policy, _ = train_dpo_2head(pref_ds, input_dim=1024)
        print("» DPO policy training complete")
        return policy

    def _generate_preference_dataset(self, facts_for_queries):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump([{"context": fact} for fact in facts_for_queries], tmp_file, indent=2)
            tmp_corpus_path = tmp_file.name

        try:
            temp_cr = ExactGraphRag_rl(
                ini_meta_codebook={},
                sentence_emb=self.sent_emb,
                word_emb=self.word_emb,
                llm=self.api_llm,
                thinkings_choice='not_include',
                answers_choice='unique',
                facts_choice='include_all',
                top_m=self.top_m,
                top_k=self.top_k,
                combine_ent_sim=self.aft_combine_sim,
                q_combine_sim=self.aft_combine_sim,
                aft_combine_sim=self.aft_combine_sim,
                semantic_overlap_sim=self.semantic_overlap_sim,
                use_word=True
            )


            temp_cr.load_and_merge_facts(
                tmp_corpus_path,
                chunk_tokens=1200,
                overlap_tokens=100,
                sub_chunk_chars=300,  
                sub_chunk_overlap=50,
                tokenizer_name="gpt-4o-mini",
                subchunk_batch=1000
            )
            temp_cr._facts_preloaded = True
            temp_cr.top_m = 5  

            seed_questions = [
                "What is the main topic?",
                "What are the key points?",
                "Can you summarize this?",
                "What is important here?",
                "What should I know?"
            ]
            
            train_answers = ["Summary", "Key information", "Important details", "Main points", "Key facts"]
            

            temp_cr.record_labeled_q_and_a(seed_questions, train_answers)

            reward_func = reward_func_dpo.reward_sbert_inclusive
            pref_ds = make_preference_dataset_2head(
                temp_cr,
                questions=seed_questions,
                gold_answers={q: a for q, a in zip(seed_questions, train_answers)},
                per_q_samples=6,
                feature_dim=1024,
                reward_fn=reward_func,
                seed=0,
                isolate_state=True,
                ANSWERS_CHOICES=ANSWERS_CHOICES,
                THINKINGS_CHOICES=THINKINGS_CHOICES,
                FACTS_CHOICES=FACTS_CHOICES
            )

            save_pref_examples(self.pref_examples_path, pref_ds)
            print(f"   generated and saved {len(pref_ds)} preference examples")
            
            return pref_ds

        finally:

            os.unlink(tmp_corpus_path)

    def get_batch_size(self):
        return 8
    
    def _normalize_space(self, s):
        """Helper function like in pipeline"""
        if isinstance(s, list):
            s = " ".join(str(x) for x in s if x is not None)
        return re.sub(r"\s+", " ", (s or "").strip())
    
    def get_sbert_model(self):
        """Initialize SBERT model for reward calculation (like pipeline)"""
        if self._sbert_model is None:
            self._sbert_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
        return self._sbert_model
    
    def reward_sbert_cached(self, pred, gold):
        """Calculate SBERT similarity reward (like pipeline)"""
        model = self.get_sbert_model()
        emb_pred, emb_gold = model.encode([pred, gold])
        emb_pred /= (np.linalg.norm(emb_pred) + 1e-9)
        emb_gold /= (np.linalg.norm(emb_gold) + 1e-9)
        return float((emb_pred * emb_gold).sum())

    def batch_generate_answer(self, batch):
            queries = batch["query"]
            batch_search_results = batch["search_results"]
            results = []

            all_facts_for_batch = []
            for i, search_results in enumerate(batch_search_results):
                facts_for_this_query = []
                for result in search_results:
                    if "page_result" in result:
                        text = result["page_result"]
                        if text and len(text.strip()) > 20:
                            facts_for_this_query.append(text.strip())
                all_facts_for_batch.extend(facts_for_this_query)

            if self.policy is None and all_facts_for_batch:
                print("» First batch - generating preference dataset and training policy...")
                pref_ds = self._generate_preference_dataset(all_facts_for_batch[:50])  
                if pref_ds:
                    print("» Training DPO policy...")
                    self.policy, _ = train_dpo_2head(pref_ds, input_dim=1024)
                    print("» DPO policy training complete for this session")

            for i, q in enumerate(queries):
                search_results = batch_search_results[i]
                facts_for_this_query = []
                for result in search_results:
                    if "page_result" in result:
                        text = result["page_result"]
                        if text and len(text.strip()) > 20:
                            facts_for_this_query.append(text.strip())

                if self.pre_loaded_meta:
                    temp_cr = self.cr
                elif facts_for_this_query:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                        json.dump([{"context": fact} for fact in facts_for_this_query], tmp_file, indent=2)
                        tmp_corpus_path = tmp_file.name
                    try:
                        temp_cr = ExactGraphRag_rl(
                            ini_meta_codebook={},
                            sentence_emb=self.sent_emb,
                            word_emb=self.word_emb,
                            llm=self.api_llm,
                            thinkings_choice='not_include',
                            answers_choice='unique',
                            facts_choice='include_all',
                            top_m=self.top_m,
                            top_k=self.top_k,
                            combine_ent_sim=self.aft_combine_sim,
                            q_combine_sim=self.aft_combine_sim,
                            aft_combine_sim=self.aft_combine_sim,
                            semantic_overlap_sim=self.semantic_overlap_sim,
                            use_word=True
                        )
                        temp_cr.load_and_merge_facts(
                            tmp_corpus_path,
                            chunk_tokens=1200,
                            overlap_tokens=100,
                            sub_chunk_chars=300,  
                            sub_chunk_overlap=50,
                            tokenizer_name="gpt-4o-mini",
                            subchunk_batch=1000
                        )
                        temp_cr._facts_preloaded = True
                        temp_cr.top_m = 5  
                        if not self.pre_loaded_meta:
                            print(f"» Saving meta_codebook with {len(temp_cr.meta_codebook.get('facts_lst', []))} facts")
                            def make_json_safe(obj):
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                if isinstance(obj, dict):
                                    return {k: make_json_safe(v) for k, v in obj.items()}
                                if isinstance(obj, list):
                                    return [make_json_safe(v) for v in obj]
                                return obj
                            with open(self.meta_codebook_path, "w") as f:
                                json.dump(make_json_safe(temp_cr.meta_codebook), f, indent=2)
                            self.pre_loaded_meta = True
                            self.cr = temp_cr
                    finally:
                        os.unlink(tmp_corpus_path)
                else:
                    pred = "I don't know"
                    from utils import trim_predictions_to_max_token_length
                    pred = trim_predictions_to_max_token_length(pred)
                    results.append({
                        "question": q,
                        "generated_answer": pred,
                        "meta": {},
                    })
                    continue

                if self.policy is None:
                    pred = temp_cr.run_work_flow(q)
                    fact_context = getattr(temp_cr, 'cur_fact_context', "")
                    _meta = {
                        'fact_context': fact_context,
                        'answers_choice': 'unique',
                        'thinkings_choice': 'not_include',
                        'facts_choice': 'include_all',
                        'meta_codebook_json_bytes': 0,
                        'meta_codebook_json_MB': 0.0
                    }
                else:
                    pred, _meta = answer_with_auto_strategy(
                        temp_cr, self.policy, q,
                        reward_fn=default_reward,
                        gold_answer=None,
                        greedy=True
                    )
                from utils import trim_predictions_to_max_token_length
                pred = trim_predictions_to_max_token_length(pred)

                start_idx = len(self.api_llm.metrics_runs) if hasattr(self.api_llm, 'metrics_runs') and self.api_llm.metrics_runs else 0
                gen_metrics = {}
                if hasattr(self.api_llm, 'metrics_runs') and self.api_llm.metrics_runs:
                    gen_metrics = (self.api_llm.metrics_runs[start_idx:] or [{}])[-1] or {}
                    if isinstance(gen_metrics, dict) and q in gen_metrics and isinstance(gen_metrics[q], dict):
                        gen_metrics = gen_metrics[q]

                _ALLOWED_KEYS = {
                    "input_tokens", "output_tokens", "total_tokens",
                    "latency_sec", "gen_latency_sec", "retrieval_latency_sec",
                    "prompt_chars", "throughput_tok_per_s", "prompt_tok_per_s",
                    "device", "dtype", "model_name",
                    "timestamp_start", "timestamp_end",
                    "attempt", "question_chars", "answer_raw_chars", "answer_raw_tokens",
                    "prompt_to_output_char_ratio", "retrieved_count",
                    "peak_vram_MiB", "total_latency_sec",
                }
                gen_metrics = {k: v for k, v in gen_metrics.items() if k in _ALLOWED_KEYS}
                
                predicted_answer_norm = self._normalize_space(pred)
                context_ret_norm = self._normalize_space(_meta.get('fact_context', ''))
                
                gold_answer_norm = "Unknown"  
                ground_truth_context = ""  
                
                if "no answer" in predicted_answer_norm.lower():
                    eval_result_correctness = 0.0
                else:
                    eval_result_correctness = 1.0  
                
                if ground_truth_context:
                    eval_result_context = self.reward_sbert_cached(context_ret_norm, ground_truth_context)
                else:
                    eval_result_context = 0.0  
                
                result_row = {
                    "question": q,
                    "generated_answer": pred,
                    "context": _meta.get('fact_context', ""),
                    "answers_choice": _meta.get('answers_choice', ""),
                    "thinkings_choice": _meta.get('thinkings_choice', ""),
                    "facts_choice": _meta.get('facts_choice', ""),
                    "correctness": eval_result_correctness,
                    "context_similarity": eval_result_context,
                    "meta_codebook_json_bytes": _meta.get('meta_codebook_json_bytes', 0),
                    "meta_codebook_json_MB": _meta.get('meta_codebook_json_MB', 0.0),
                    **gen_metrics  
                }
                results.append(result_row)
                self.all_results.append(result_row)

            Path(self.results_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_path, "w") as f:
                json.dump(self.all_results, f, indent=2)
                print(f"✓ wrote {len(self.all_results)} total results → {self.results_path}")
            
            answers = [result["generated_answer"] for result in results]
            return answers

# UserModel = UserModel

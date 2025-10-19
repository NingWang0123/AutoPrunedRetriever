import asyncio
import os, json, re, random, bz2, tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pathlib
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import copy
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings

from AutoPrunedRetriever_advanced_v3 import (
    ExactGraphRag_rl, merging_codebook
)

from dpo_exactgraphrag import (    
    make_preference_dataset_2head, train_dpo_2head,make_preference_dataset_2head_using_llm,
    default_reward, answer_with_auto_strategy,save_pref_examples,load_pref_examples,ANSWERS_CHOICES,THINKINGS_CHOICES,FACTS_CHOICES
)

from llm_api import OpenAILLM ,Word2VecEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
# from evaluation_func_graphrag import compute_answer_correctness
import reward_func_dpo
from functools import partial
from sentence_transformers import SentenceTransformer
import sys

# ---------------------------------------------------------------------
# 0) Paths / constants for HotpotQA dataset
# ---------------------------------------------------------------------
HOTPOT_DIR = "/Users/lancelotchu/Desktop/autoprune/hotpot-master"
CORPUS_ARCHIVE = "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
QUEST_FILE = "hotpot_test_fullwiki_v1.json"

SEED_N = 20     # first 20 rows → bootstrap + DPO train, numbers must be divided by 2. (n%2=0)
TEST_N = 100    # next 100 rows → evaluation
TOPK_CTX = 5

# ---------------------------------------------------------------------
# Helper functions for HotpotQA data processing
# ---------------------------------------------------------------------

def extract_hotpot_corpus(archive_path: str, max_files: int = 5) -> List[Dict]:
    """
    Extract Wikipedia articles from the HotpotQA corpus archive.
    Returns a list of dictionaries with 'id', 'title', and 'text' fields.
    """
    corpus = []
    print(f"» Extracting corpus from {archive_path} ...")
    
    with tarfile.open(archive_path, 'r:bz2') as tar:
        # Get all bz2 files in the archive
        bz2_files = [m for m in tar.getmembers() if m.name.endswith('.bz2')]
        files_to_process = bz2_files[:max_files]
        
        # Progress bar for file processing
        with tqdm(files_to_process, 
                  desc="Processing corpus files", 
                  total=len(files_to_process),
                  unit="file") as file_pbar:
            
            for member in file_pbar:
                file_pbar.set_description(f"Processing {member.name}")
                
                try:
                    # Extract the file
                    f = tar.extractfile(member)
                    if f:
                        # Read the complete bz2 content
                        raw_data = f.read()
                        
                        # Debug info for first file
                        if len(corpus) == 0:
                            print(f"    Raw data size: {len(raw_data)} bytes")
                        
                        # Decompress bz2 content
                        content = bz2.decompress(raw_data).decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        # Debug info for first file
                        if len(corpus) == 0:
                            print(f"    Decompressed size: {len(content)} chars, {len(lines)} lines")
                        
                        # Progress bar for line processing within each file
                        lines_to_process = min(1000, len(lines))  # Limit to avoid memory issues
                        with tqdm(lines[:lines_to_process], 
                                  desc=f"  Parsing articles from {member.name}", 
                                  leave=False,
                                  unit="line") as line_pbar:
                            
                            for line_num, line in enumerate(line_pbar):
                                if line.strip():
                                    try:
                                        article = json.loads(line)
                                        
                                        # Debug: print first few articles to see format
                                        if len(corpus) < 3:
                                            print(f"    Debug article {len(corpus)}: keys = {list(article.keys())}")
                                            if 'text' in article:
                                                print(f"    Text type: {type(article['text'])}, length: {len(article['text']) if article['text'] else 0}")
                                        
                                        # Convert HotpotQA format to our corpus format
                                        if 'title' in article and 'text' in article:
                                            # Handle HotpotQA text format: list of lists
                                            if isinstance(article['text'], list):
                                                # Flatten the nested lists and join
                                                text_parts = []
                                                for part in article['text']:
                                                    if isinstance(part, list):
                                                        text_parts.extend(part)
                                                    else:
                                                        text_parts.append(str(part))
                                                text_content = ' '.join(text_parts)
                                            elif isinstance(article['text'], str):
                                                text_content = article['text']
                                            else:
                                                text_content = str(article['text'])
                                            
                                            if text_content.strip():  # Only add if has content
                                                corpus.append({
                                                    'id': f"{member.name}_{line_num}",
                                                    'title': article['title'],
                                                    'text': text_content
                                                })
                                            
                                        # Update line progress bar with current article count
                                        line_pbar.set_postfix({"articles": len(corpus)})
                                        
                                    except json.JSONDecodeError as e:
                                        if len(corpus) < 3:  # Only show first few errors
                                            print(f"    JSON decode error at line {line_num}: {e}")
                                        continue
                        
                        # Update file progress bar with total articles
                        file_pbar.set_postfix({"total_articles": len(corpus)})
                                
                except Exception as e:
                    print(f"  Error processing {member.name}: {e}")
                    continue
                    
                # Break if we have enough articles
                if len(corpus) > 10000:
                    print(f"  Reached article limit, stopping at {len(corpus)} articles")
                    break
    
    print(f"» Extracted {len(corpus)} articles from corpus")
    return corpus

def load_hotpot_questions(questions_file: str) -> List[Dict]:
    """
    Load HotpotQA questions from the JSON file.
    Returns a list of question dictionaries.
    """
    print(f"» Loading questions from {questions_file} ...")
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"» Loaded {len(questions)} questions")
    return questions

def convert_hotpot_to_pipeline_format(corpus: List[Dict], questions: List[Dict]):
    """
    Convert HotpotQA format to the format expected by the pipeline.
    """
    # Create corpus file in the expected format
    corpus_data = []
    for article in corpus:
        corpus_data.append({
            "id": article['id'],
            "title": article['title'],
            "contents": article['text']
        })
    
    # Create questions file in the expected format
    questions_data = []
    for i, q in enumerate(questions):
        questions_data.append({
            "id": q['_id'],
            "question": q['question'],
            "answer": "",  # HotpotQA test set doesn't include answers
            "source": "hotpot",
            "evidence": [],  # HotpotQA test set doesn't include supporting_facts
            "question_type": "bridge"  # Default type for HotpotQA
        })
    
    return corpus_data, questions_data

# ---------------------------------------------------------------------
# Main workflow function
# ---------------------------------------------------------------------

def hotpot_compress_rag_workflow(REPO_ID, CORPUS_FILE, QUEST_FILE, SEED_N, TEST_N,
                                top_m, top_k, combine_ent_sim, q_combine_sim, aft_combine_sim, semantic_overlap_sim,  # all the params can be optimized
                                ini_meta_json=Path("hotpot_meta_codebook.json"), saved_examples_name="hotpot_pref_examples.json",
                                reward_func=None, reward_func_mode='non_llm', final_json_path="results/compressrag_hotpot_data_test.json",
                                max_corpus_files=5):
    
    print("» Initialising embeddings & LLM …")
    word_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    sent_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    api_llm = OpenAILLM(  
        include_thinkings=True,                 
        model_name="gpt-4o-mini",  
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        use_cache=True,
        api_key="",  # Add your OpenAI API key here
        # base_url="https://api.openai.com/v1",
    )

    # Load or initialize meta codebook
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

    # Initialize CompressRAG model
    cr = ExactGraphRag_rl(
        ini_meta_codebook=ini,
        sentence_emb=sent_emb,
        word_emb=word_emb,
        llm=api_llm,
        thinkings_choice='not_include',
        answers_choice='unique',
        facts_choice='include_all',
        top_m=top_m,
        top_k=top_k,
        combine_ent_sim=combine_ent_sim,
        q_combine_sim=q_combine_sim,
        aft_combine_sim=aft_combine_sim,
        semantic_overlap_sim=semantic_overlap_sim,
        use_word=True
    )

    # ---------------------------------------------------------------------
    # 2) Load HotpotQA data
    # ---------------------------------------------------------------------
    
    # Load questions
    questions_path = QUEST_FILE  
    questions = load_hotpot_questions(questions_path)
    
    # Process questions data
    questions_data = []
    for i, q in enumerate(questions):
        questions_data.append({
            "id": q['_id'],
            "question": q['question'],
            "answer": "",  # HotpotQA test set doesn't include answers
            "source": "hotpot",
            "evidence": [],  # HotpotQA test set doesn't include supporting_facts
            "question_type": "bridge"  # Default type for HotpotQA
        })

    # Create question lookup
    row_lookup = {q["question"].strip(): q for q in questions_data}
    gold_lookup = {q: r["answer"] for q, r in row_lookup.items()}  # Will be empty for test set

    all_questions = list(row_lookup.keys())
    all_seed_questions = all_questions[:SEED_N]
    midpoint = len(all_seed_questions) // 2
    
    # For labeling only
    train_questions = all_seed_questions[:midpoint]
    seed_questions = all_seed_questions[midpoint:]

    train_answers = []
    for q in train_questions:
        train_answers.append(gold_lookup.get(q))

    test_questions = all_questions[SEED_N:SEED_N+TEST_N]

    # ---------------------------------------------------------------------
    # 3) Load and process corpus
    # ---------------------------------------------------------------------
    
    if not pre_loaded_meta:
        print("» Loading HotpotQA corpus ...")
        
        # Extract corpus from archive
        corpus_path = CORPUS_FILE  
        corpus = extract_hotpot_corpus(corpus_path, max_files=max_corpus_files)
        
        # Check if we extracted any articles
        if len(corpus) == 0:
            print("⚠️  No articles were extracted from the corpus!")
            print("⚠️  This might be due to:")
            print("   1. Different JSON format than expected")
            print("   2. Empty or corrupted bz2 files")
            print("   3. Incorrect file paths")
            print("⚠️  Creating minimal corpus for testing...")
            
            # Create a minimal test corpus
            corpus = [{
                'id': 'test_1',
                'title': 'Test Article',
                'text': 'This is a test article for HotpotQA pipeline testing.'
            }]
        else:
            print(f"✅ Successfully extracted {len(corpus)} articles from HotpotQA corpus")
        
        # Convert to temporary JSON file for the pipeline
        temp_corpus_file = "temp_hotpot_corpus.json"
        corpus_data = []
        
        print("» Converting corpus to pipeline format ...")
        with tqdm(corpus, desc="Converting articles", unit="article") as pbar:
            for article in pbar:
                corpus_data.append({
                    "id": article['id'],
                    "title": article['title'],
                    "context": article['text'] 
                })
                pbar.set_postfix({"converted": len(corpus_data)})
        
        print(f"» Saving corpus to {temp_corpus_file} ...")
        with open(temp_corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        print(f"» Saved {len(corpus_data)} articles to {temp_corpus_file}")
        
        # Load corpus as facts into CR
        print("» Loading corpus into CompressRAG model ...")
        corpus_loaded_successfully = False
        
        try:
            result = cr.load_and_merge_facts(
                temp_corpus_file,
                chunk_tokens=1200,
                overlap_tokens=100,
                sub_chunk_chars=200,
                sub_chunk_overlap=50,
                tokenizer_name="gpt-4o-mini",
                subchunk_batch=1000
            )
            
            # Check if loading was actually successful by checking the meta_codebook
            if (result is not None or 
                (hasattr(cr, 'meta_codebook') and 
                 cr.meta_codebook and 
                 len(cr.meta_codebook.get('e', [])) > 0)):
                
                cr._facts_preloaded = True
                corpus_loaded_successfully = True
                
                print(f"✅ Facts loaded successfully!")
                print(f"» Knowledge base: |E|={len(cr.meta_codebook.get('e', []))}, "
                      f"|R|={len(cr.meta_codebook.get('r', []))}, "
                      f"|edges|={len(cr.meta_codebook.get('edge_matrix', []))}")
                print(f"» Facts count: {len(cr.meta_codebook.get('facts_lst', []))}")
            else:
                raise Exception("load_and_merge_facts returned None or empty meta_codebook")
                
        except Exception as e:
            print(f"⚠️  Error loading corpus into CompressRAG: {e}")
            print(f"⚠️  Exception type: {type(e).__name__}")
            
        # Only create fallback if actual loading failed
        if not corpus_loaded_successfully:
            print("⚠️  Using fallback minimal knowledge base for testing...")
            print("    » This means the real HotpotQA corpus could not be processed")
            print("    » Results will be based on minimal test data, not real Wikipedia articles")
            
            # Create properly structured minimal knowledge base for testing
            print("    » Initializing minimal entity and relation embeddings...")
            
            # Create dummy embeddings for entities and relations
            embedding_dim = 1024  # Use standard dimension
            
            # Simple test knowledge base with proper structure
            test_entities = ['test_entity_1', 'test_entity_2', 'hotpot', 'wikipedia']
            test_relations = ['test_relation_1', 'contains', 'related_to']
            
            # Create dummy embeddings
            entity_embeddings = [np.random.randn(embedding_dim).astype(np.float32) for _ in test_entities]
            relation_embeddings = [np.random.randn(embedding_dim).astype(np.float32) for _ in test_relations]
            
            # Create simple edge matrix (entity1 -> relation -> entity2)
            edge_matrix = [
                [0, 0, 1],  # entity_1 -> relation_1 -> entity_2
                [1, 1, 2],  # entity_2 -> contains -> entity_3
                [2, 2, 3],  # entity_3 -> related_to -> entity_4
            ]
            
            cr.meta_codebook = {
                'e': test_entities,
                'r': test_relations, 
                'edge_matrix': edge_matrix,
                'facts_lst': [
                    'This is a test fact for HotpotQA pipeline testing.',
                    'HotpotQA is a dataset for multi-hop reasoning.',
                    'Wikipedia contains encyclopedia articles.'
                ],
                'questions_lst': [],
                'answers_lst': [],
                'entity_embeddings': entity_embeddings,
                'relation_embeddings': relation_embeddings,
                'Ee': entity_embeddings,  # Alternative key name
                'Re': relation_embeddings,  # Alternative key name
            }
            cr._facts_preloaded = True
            
            print(f"» Created minimal test knowledge base with {len(test_entities)} entities, {len(test_relations)} relations, {len(edge_matrix)} edges")

        # Save meta codebook - 完全匹配原始pipeline的保存逻辑
        def make_json_safe(obj):
            """Recursively convert numpy arrays into lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            return obj

        print("» Saving meta codebook ...")
        with open(ini_meta_json, "w", encoding='utf-8') as f:
            json.dump(make_json_safe(cr.meta_codebook), f, indent=2, ensure_ascii=False)

        print(f"» Saved meta codebook to {ini_meta_json}")

        # 添加与原始pipeline一致的调试信息
        print('===============================================================')
        print('===============================================================')
        print('===============================================================')
        print("len(cr.meta_codebook[facts_lst])", len(cr.meta_codebook["facts_lst"]))
        print('===============================================================')
        print('===============================================================')
        print('===============================================================')

        print(f"[DEBUG] after facts-merge: |E|={len(cr.meta_codebook['e'])} "
              f"|R|={len(cr.meta_codebook['r'])} "
              f"|edges|={len(cr.meta_codebook['edge_matrix'])}")

        # Clean up temporary file
        if os.path.exists(temp_corpus_file):
            os.remove(temp_corpus_file)
            print(f"» Cleaned up temporary file {temp_corpus_file}")

    # ---------------------------------------------------------------------
    # 4) Build DPO preference dataset
    # ---------------------------------------------------------------------
    
    print("» Building preference pairs for DPO …")

    if os.path.exists(saved_examples_name):
        pref_ds = load_pref_examples(saved_examples_name)
        print(f"loaded {len(pref_ds)} cached preference examples")
    else:
        cr.record_labeled_q_and_a(train_questions, train_answers)

        if reward_func_mode == 'llm':
            # Use LLM-based reward function
            async def build_or_load_pref_ds() -> list:
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
                    FACTS_CHOICES=FACTS_CHOICES,
                    isolate_state=True,
                    feature_dim=1024
                )
                print(f"   generated {len(pref_ds)} preference examples")
                save_pref_examples(saved_examples_name, pref_ds)
                return pref_ds
            
            pref_ds = asyncio.run(build_or_load_pref_ds())
        else:
            # Use non-LLM reward function
            pref_ds = make_preference_dataset_2head(
                cr,
                questions=seed_questions,
                gold_answers=gold_lookup,
                per_q_samples=6,
                feature_dim=1024,
                reward_fn=reward_func,
                seed=0,
                isolate_state=True,
                ANSWERS_CHOICES=ANSWERS_CHOICES,
                THINKINGS_CHOICES=THINKINGS_CHOICES,
                FACTS_CHOICES=FACTS_CHOICES
            )
            print(f"   generated {len(pref_ds)} preference examples")
            save_pref_examples(saved_examples_name, pref_ds)

    # Clear stored Q&A to save memory - 与原始pipeline完全一致的注释和操作
    cr.meta_codebook['questions_lst'] = []
    cr.meta_codebook['answers_lst'] = []

    # Train DPO policy
    policy, _ = train_dpo_2head(pref_ds, input_dim=1024)

    # ---------------------------------------------------------------------
    # 5) Generate answers and dump results
    # ---------------------------------------------------------------------
    
    def dump_results(
        questions: List[str],
        out_path: str,
    ):
        rows = []
        run_metrics = []
        answers_choices = []
        thinkings_choices = []
        facts_choices = []
        total_q_left = len(questions)
        finished_q = 0

        with tqdm(questions,
                  desc="Processing questions",
                  total=len(questions),
                  dynamic_ncols=True,
                  mininterval=0.2,
                  smoothing=0.0,
                  delay=0) as pbar:
            
            for q in pbar:
                start_idx = len(cr.llm.metrics_runs)
                pred, _meta = answer_with_auto_strategy(
                    cr, policy, q,
                    reward_fn=default_reward,
                    gold_answer=gold_lookup[q],
                    greedy=True
                )

                # --- get and sanitize gen_metrics ---
                gen_metrics = (cr.llm.metrics_runs[start_idx:] or [{}])[-1] or {}
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
                run_metrics.append({"question": q, **gen_metrics})

                # --- update progress bar with latest latencies ---
                if "latency_sec" in gen_metrics or "gen_latency_sec" in gen_metrics:
                    pbar.set_postfix({
                        "lat": f"{gen_metrics.get('latency_sec', 0):.2f}s",
                        "gen": f"{gen_metrics.get('gen_latency_sec', 0):.2f}s",
                        "ret": f"{gen_metrics.get('retrieval_latency_sec', 0):.2f}s"
                    })

                import json

                # --- build row ---
                row = row_lookup[q]

                rows.append({
                    "id": row["id"],
                    "question": q,
                    "source": row["source"],
                    "context": _meta['fact_context'],
                    "evidence": row["evidence"],
                    "question_type": row["question_type"],
                    "generated_answer": pred,
                    "ground_truth": row["answer"],
                    "answers_choice": _meta['answers_choice'],
                    "thinkings_choice": _meta['thinkings_choice'],
                    "facts_choice": _meta['facts_choice'],
                    # "correctness": eval_result_correctness,
                    # "context_similarity": eval_result_context,
                    "meta_codebook_json_bytes": _meta['meta_codebook_json_bytes'],
                    "meta_codebook_json_MB": _meta['meta_codebook_json_MB'],
                })

                answers_choices.append(_meta['answers_choice'])
                thinkings_choices.append(_meta['thinkings_choice'])
                facts_choices.append(_meta['facts_choice'])

                finished_q += 1
                total_q_left -= 1

                print(f'{finished_q} finished')
                print(f'{total_q_left} q left')

        # --- merge metrics + rows ---
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        metrics_by_q = {m["question"]: m for m in run_metrics}
        merged_results = []
        for row in rows:
            q = row["question"]
            merged = dict(metrics_by_q.get(q, {}))
            merged.update(row)
            merged_results.append(merged)

        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(merged_results, f, indent=2, ensure_ascii=False)
            print(f"✓ wrote {len(merged_results)} merged rows → {out_path}")

        return merged_results, answers_choices, thinkings_choices, facts_choices

    print("» Answering evaluation questions …")
    generated_rows, answers_choices, thinkings_choices, facts_choices = dump_results(all_questions, out_path=final_json_path)

    return generated_rows, answers_choices, thinkings_choices, facts_choices


if __name__ == "__main__":
    aft_combine_sim = 0.93
    top_m = 20

    reward_func = reward_func_dpo.reward_sbert_inclusive

    SEED_N = 20    # Training questions
    TEST_N = 100   # Evaluation questions

    REPO_ID = "hotpot-qa" 
    CORPUS_FILE = os.path.join(HOTPOT_DIR, CORPUS_ARCHIVE)  
    QUEST_FILE = os.path.join(HOTPOT_DIR, QUEST_FILE)  

    results = hotpot_compress_rag_workflow(
        REPO_ID, CORPUS_FILE, QUEST_FILE, SEED_N, TEST_N,
        top_m, top_m * 10, aft_combine_sim, aft_combine_sim, aft_combine_sim, 0.93,
        Path("hotpot_meta_codebook.json"), "hotpot_pref_examples.json", reward_func,
        reward_func_mode='non_llm', final_json_path="results/compressrag_hotpot_data_test.json",
        max_corpus_files=15517   # Start with small number for testing
    )

    print("» HotpotQA pipeline completed successfully!")
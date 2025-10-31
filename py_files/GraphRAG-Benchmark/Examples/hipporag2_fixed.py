import os
import asyncio
import argparse
import json
import logging
from typing import Dict, List
from dotenv import load_dotenv
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Import HippoRAG components after setting environment
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.misc_utils import string_to_bool
from hipporag.utils.config_utils import BaseConfig
from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hipporag_processing.log")
    ]
)

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

def split_text(
    text: str, 
    tokenizer: AutoTokenizer, 
    chunk_token_size: int = 1200, 
    chunk_overlap_token_size: int = 100
) -> List[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_path: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: int
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Prepare output directory
    output_dir = f"./results/hipporag2/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize tokenizer for text splitting
    try:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        logging.info(f"✅ Loaded tokenizer: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Split text into chunks
    chunks = split_text(context, tokenizer)
    logging.info(f"✂️ Split corpus into {len(chunks)} chunks")
    
    # Format chunks as documents
    docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare queries and gold answers
    all_queries = [q["question"] for q in corpus_questions]
    gold_answers = [[q['answer']] for q in corpus_questions]
    
    # Configure HippoRAG
    config = BaseConfig(
        save_dir=os.path.join(base_dir, corpus_name),
        llm_base_url=llm_base_url,
        llm_name=model_name,
        embedding_model_name=embed_model_path,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path="hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=5,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online"
    )
    
    # Override LLM configuration for Ollama mode
    if mode == "ollama":
        config.llm_mode = "ollama"
        logging.info(f"✅ Using Ollama mode: {model_name} at {llm_base_url}")
    else:
        config.llm_mode = "openai"
        logging.info(f"✅ Using OpenAI mode: {model_name} at {llm_base_url}")
    
    # Initialize HippoRAG
    hipporag = HippoRAG(global_config=config)
    
    # Index the corpus content
    hipporag.index(docs)
    logging.info(f"✅ Indexed corpus: {corpus_name}")
    
    # Process questions
    results = []

    queries_solutions, _, _, _, _ = hipporag.rag_qa(queries=all_queries, gold_docs=None, gold_answers=gold_answers)
    solutions = [query.to_dict() for query in queries_solutions]
    
    for question in corpus_questions:
        solution = next((sol for sol in solutions if sol['question'] == question['question']), None)
        if solution:
            results.append({
                "id": question["id"],
                "question": question["question"],
                "source": corpus_name,
                "context": solution.get("docs", ""),
                "evidence": question.get("evidence", ""),
                "question_type": question.get("question_type", ""),
                "generated_answer": solution.get("answer", ""),
                "ground_truth": question.get("answer", "")
            })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    # Define subset paths
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.json",
            "questions": "./Datasets/Questions/medical_complex_reasoning.json"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.json",
            "questions": "./Datasets/Questions/novel_complex_reasoning.json"
        },
        "2wikimultihop": {
            "corpus": "./Datasets/Corpus/2wikimultihop.json",
            "questions": "./Datasets/Questions/2wikimultihop_questions.json"
        },
        "hotpotqa": {
            "corpus": "./Datasets/Corpus/hotpotqa.json",
            "questions": "./Datasets/Questions/hotpotqa_questions.json"
        },
        "history": {
            "corpus": "./Datasets/Corpus/history_corpus.json",
            "questions": "./Datasets/Questions/history_QnA.json"
        },
        "stem": {
            "corpus": "./Datasets/Corpus/stem_corpus.json",
            "questions": "./Datasets/Questions/stem_QnA_modified.json"
        },
        "tv": {
            "corpus": "./Datasets/Corpus/tv_corpus.json",
            "questions": "./Datasets/Questions/tv_QnA_modified.json"
        }
    }
    
    parser = argparse.ArgumentParser(description="HippoRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel", "2wikimultihop", "hotpotqa", "history"],
                        help="Subset to process (medical, novel, 2wikimultihop, hotpotqa, or history)")
    
    parser.add_argument("--base_dir", default="./hipporag2_workspace",
                        help="Base working directory for HippoRAG")

    
    # Model configuration
    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="gpt-4o-mini", 
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="BAAI/bge-large-en-v1.5",
                            help="Path to embedding model directory or HF model name (for tokenizer splitting & HippoRAG config)")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use OPENAI_API_KEY environment variable)")

    args = parser.parse_args()
    
    # logging.info(f"🚀 Starting HippoRAG processing for subset: {args.subset}")
    
    # # Validate subset
    # if args.subset not in SUBSET_PATHS:
    #     logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
    #     return
    
    # # Get file paths for this subset
    # corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    # questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # # Handle API key security
    # api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    # if not api_key:
    #     logging.warning("⚠️ No API key provided! Requests may fail.")
    
    # # Create workspace directory
    # os.makedirs(args.base_dir, exist_ok=True)
    
    # # Load corpus data
    # try:
    #     corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
    #     corpus_data = []
    #     for item in corpus_dataset:
    #         corpus_data.append({
    #             "corpus_name": item["corpus_name"],
    #             "context": item["context"]
    #         })
    #     logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    # except Exception as e:
    #     logging.error(f"❌ Failed to load corpus: {e}")
    #     return
    
    # # Sample corpus data if requested
    # if args.sample:
    #     corpus_data = corpus_data[:1]
    
    # # Load question data
    # try:
    #     questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
    #     question_data = []
    #     for item in questions_dataset:
    #         question_data.append({
    #             "id": item["id"],
    #             "source": item["source"],
    #             "question": item["question"],
    #             "answer": item["answer"],
    #             "question_type": item["question_type"],
    #             "evidence": item["evidence"]
    #         })
    #     grouped_questions = group_questions_by_source(question_data)
    #     logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    # except Exception as e:
    #     logging.error(f"❌ Failed to load questions: {e}")
    #     return
    
    # # Process each corpus concurrently using asyncio + threads
    # async def _run_all():
    #     tasks = []
    #     for item in corpus_data:
    #         tasks.append(asyncio.to_thread(
    #             process_corpus,
    #             item["corpus_name"],
    #             item["context"],
    #             args.base_dir,
    #             args.mode,
    #             args.model_name,
    #             args.embed_model_path,
    #             args.llm_base_url,
    #             api_key,
    #             grouped_questions,
    #             args.sample,
    #         ))
    #     results = await asyncio.gather(*tasks, return_exceptions=True)
    #     for r in results:
    #         if isinstance(r, Exception):
    #             logging.exception(f"❌ Task failed: {r}")

    # asyncio.run(_run_all())


    logging.info(f"🚀 Starting HippoRAG processing for subset: {args.subset} (mode={args.mode})")

    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if args.mode == "API" and not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")

    os.makedirs(args.base_dir, exist_ok=True)

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return

    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        print(corpus_name)
        print(context)
        # process_corpus(
        #     corpus_name=corpus_name,
        #     context=context,
        #     base_dir=args.base_dir,
        #     mode=args.mode,
        #     model_name=args.model_name,
        #     embed_model_path=args.embed_model_path,
        #     llm_base_url=args.llm_base_url,
        #     llm_api_key=api_key,
        #     questions=grouped_questions,
        #     sample=args.sample,
        # )
 

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  
    main()
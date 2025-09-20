# lightrag_example.py
import asyncio
import os
import logging
import nest_asyncio
import argparse
import json
from typing import Dict, List

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def group_questions_by_source(question_list):
    grouped_questions = {}

    for question in question_list:
        source = question.get("source")

        if source not in grouped_questions:
            grouped_questions[source] = []

        grouped_questions[source].append(question)

    return grouped_questions


SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""

async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """LLM interface function using OpenAI-compatible API"""
    # Get API configuration from kwargs
    model_name = kwargs.get("model_name", "qwen2.5-14b-instruct")
    base_url = kwargs.get("base_url", "")
    api_key = kwargs.get("api_key", "")
    
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )

async def initialize_rag(
    base_dir: str,
    source: str,
    mode: str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str
) -> LightRAG:
    working_dir = os.path.join(base_dir, source)
    os.makedirs(working_dir, exist_ok=True)

    if mode == "API":
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        embed_model = AutoModel.from_pretrained(embed_model_name)
        embedding_dim = 1024 if "large" in embed_model_name.lower() else 768
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tokenizer, embed_model),
        )
        llm_kwargs = {
            "model_name": model_name,
            "base_url": llm_base_url,
            "api_key": llm_api_key
        }

        llm_model_func_input = llm_model_func

    elif mode == "ollama":
        ollama_host = llm_base_url or "http://localhost:11434"

        if embed_model_name in ("nomic-embed-text",):
            embedding_dim = 768
        else:
            embedding_dim = 1024

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model_name, host=ollama_host
            ),
        )
        llm_kwargs = {
            "host": ollama_host,
            "options": {"num_ctx": 32768},
        }
        llm_model_func_input = ollama_model_complete

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'API' or 'ollama'.")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func_input,  
        llm_model_name=model_name,
        llm_model_max_async=4,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: List[dict],
    sample: int,
    retrieve_topk: int
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Initialize RAG for this corpus
    rag = await initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        mode=mode,
        model_name=model_name,
        embed_model_name=embed_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key
    )
    
    # Index the corpus content
    rag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")
    
    corpus_questions = questions.get(corpus_name, [])
    
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare output path
    output_dir = f"./results/lightrag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Process questions
    results = []
    query_type = 'hybrid'
    
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        # Prepare query parameters
        query_param = QueryParam(
            mode=query_type,
            top_k=retrieve_topk,
        )

        result = rag.query(
            q["question"],
            param=query_param,
            system_prompt=SYSTEM_PROMPT
        )
        if asyncio.iscoroutine(result):
            result = await result

        context = ""
        if isinstance(result, tuple):
            response = result[0]
            if len(result) >= 2:
                context = result[1] if result[1] is not None else ""
        else:
            response = result

        predicted_answer = str(response)


        # Collect results
        results.append({
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": context,
            "evidence": q["evidence"],
            "question_type": q["question_type"],
            "generated_answer": predicted_answer,
            "ground_truth": q.get("answer"),

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
            "questions": "./Datasets/Questions/medical_questions.json"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.json",
            "questions": "./Datasets/Questions/novel_questions.json"
        }
    }
    
    parser = argparse.ArgumentParser(description="LightRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./lightrag_workspace", help="Base working directory")
    
    # Model configuration
    parser.add_argument("--mode", required=True, choices=["API", "ollama"], help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", help="LLM model identifier")
    parser.add_argument("--embed_model", default="bge-base-en", help="Embedding model name")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    
    # Validate subset and mode
    if args.subset not in SUBSET_PATHS:
        logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    if args.mode not in ["API", "ollama"]:
        valid_modes = ["API", "ollama"]
        logging.error(f"Invalid mode: {args.mode}. Valid options: {valid_modes}")
        return

    
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("No API key provided! Requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]

    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
            grouped_questions = group_questions_by_source(question_data)
        logging.info(f"Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return
    
    # Process each corpus in the subset
    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        asyncio.run(
            process_corpus(
                corpus_name=corpus_name,
                context=context,
                base_dir=args.base_dir,
                mode=args.mode,
                model_name=args.model_name,
                embed_model_name=args.embed_model,
                llm_base_url=args.llm_base_url,
                llm_api_key=api_key,
                questions=grouped_questions,
                sample=args.sample,
                retrieve_topk=args.retrieve_topk
            )
        )

if __name__ == "__main__":
    main()


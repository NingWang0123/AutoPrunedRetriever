import asyncio
import argparse
import json
import numpy as np
import os
from typing import Dict, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from Evaluation.metrics import compute_context_relevance, compute_context_recall
from langchain_community.embeddings import OllamaEmbeddings
from Evaluation.llm import OllamaClient,OllamaWrapper

    
async def evaluate_dataset(
    dataset: Dataset,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    max_concurrent: int = 1,
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate context relevance and recall for a dataset"""
    results = {
        "context_relevancy": [],
        "context_recall": []
    }
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    evidences = dataset["evidences"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                contexts=contexts_list[i],
                evidences=evidences[i],
                llm=llm,
                embeddings=embeddings
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "contexts": contexts_list[i],
                    "evidences": evidences[i],
                    "metrics": sample_metrics
                }
            return sample_metrics

    tasks = [evaluate_with_semaphore(i) for i in range(total_samples)]
    sample_results = []
    completed = 0
    
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            if detailed_output:
                detailed_results.append(result)
                # metrics aggregation
                for metric, score in result["metrics"].items():
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        results[metric].append(score)
            else:
                sample_results.append(result)
                for metric, score in result.items():
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        results[metric].append(score)
            completed += 1
            print(f"✅ Completed sample {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")
        except Exception as e:
            print(f"❌ Sample failed: {e}")
            completed += 1
    
    avg_results = {
        "context_relevancy": np.nanmean(results["context_relevancy"]),
        "context_recall": np.nanmean(results["context_recall"])
    }
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results


async def evaluate_sample(
    question: str,
    contexts: List[str],
    evidences: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate retrieval metrics for a single sample"""
    # Evaluate both metrics in parallel
    relevance_task = compute_context_relevance(question, contexts, llm)
    recall_task = compute_context_recall(question, contexts, evidences, llm)
    
    # Wait for both tasks to complete
    relevance_score, recall_score = await asyncio.gather(relevance_task, recall_task)

    print(f"Relevance Score: {relevance_score}, Recall Score: {recall_score}")

    return {
        "context_relevancy": relevance_score,
        "context_recall": recall_score
    }

async def main(args: argparse.Namespace):
    """Main retrieval evaluation function"""
    if args.mode == "API":
        # Check API key
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY environment variable is not set")
        
        # Initialize models
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0.0,
            max_retries=3,
            timeout=30
        )
        
        # Initialize the embedding model
        embedding = HuggingFaceBgeEmbeddings(model_name=args.embedding_model)

    elif args.mode == "ollama":
        ollama_client = OllamaClient(base_url=args.base_url)
        llm = OllamaWrapper(ollama_client, args.model)
        ollama_embeddings = OllamaEmbeddings(
            model=args.bge_model,
            base_url=args.base_url
        )
        embedding = LangchainEmbeddingsWrapper(embeddings=ollama_embeddings)

    # Load evaluation data
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        file_data = json.load(f)  # List of question items
    
    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    
    all_results = {}
    
    # Evaluate each question type
    for question_type in list(grouped_data.keys()):
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]

        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        evidences = [item['evidence'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "contexts": contexts,
            "evidences": evidences
        }
        dataset = Dataset.from_dict(data)
        
        # If sample
        if args.num_samples:
            dataset = dataset.select([i for i in list(range(args.num_samples))])
        
        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm, 
            embeddings=embedding,
            detailed_output=args.detailed_output
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        if args.detailed_output:
            for metric, score in results["average_scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
            
    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print('\nEvaluation complete.')

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add command-line arguments
    parser.add_argument(
        "--mode", 
        required=True,
        choices=["API", "ollama"],
        type=str,
        default="API",
        help="Use API or ollama for LLM"
    )

    parser.add_argument(
        "--model", 
        type=str,
        default="gpt-4-turbo",
        help="OpenAI model to use for evaluation"
    )
    
    parser.add_argument(
        "--base_url", 
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI API"
    )
    
    parser.add_argument(
        "--embedding_model", 
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for embeddings"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="retrieval_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int,
        default=None,
        help="Number of samples per question type to evaluate (optional)"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))
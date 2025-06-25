import asyncio
import argparse
import json
import numpy as np
import os
from typing import Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from .metrics import compute_context_relevance, compute_context_recall

async def evaluate_dataset(
    dataset: Dataset,
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate context relevance and recall for a dataset"""
    results = {
        "context_relevancy": [],
        "context_recall": []
    }
    
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    # Create evaluation tasks
    tasks = []
    for i in range(total_samples):
        tasks.append(
            evaluate_sample(
                question=questions[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                llm=llm,
                embeddings=embeddings
            )
        )
    
    # Collect results with progress
    sample_results = []
    for i, future in enumerate(asyncio.as_completed(tasks)):
        result = await future
        sample_results.append(result)
        print(f"Completed sample {i+1}/{total_samples} - {((i+1)/total_samples)*100:.1f}%")
    
    # Aggregate results
    for sample in sample_results:
        for metric, score in sample.items():
            if not np.isnan(score):
                results[metric].append(score)
    
    return {
        "context_relevancy": np.nanmean(results["context_relevancy"]),
        "context_recall": np.nanmean(results["context_recall"])
    }

async def evaluate_sample(
    question: str,
    contexts: List[str],
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate retrieval metrics for a single sample"""
    # Evaluate both metrics in parallel
    relevance_task = compute_context_relevance(question, contexts, llm)
    recall_task = compute_context_recall(question, contexts, ground_truth, embeddings)
    
    # Wait for both tasks to complete
    relevance_score, recall_score = await asyncio.gather(relevance_task, recall_task)
    
    return {
        "context_relevancy": relevance_score,
        "context_recall": recall_score
    }

async def main(args: argparse.Namespace):
    """Main retrieval evaluation function"""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize models
    llm = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
        max_retries=3,
        timeout=30
    )
    
    # Initialize embeddings
    bge_embeddings = HuggingFaceBgeEmbeddings(model_name=args.bge_model)
    embedding = LangchainEmbeddingsWrapper(embeddings=bge_embeddings)
    
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
        
        # Apply sample limit if specified
        if args.num_samples:
            group_items = group_items[:args.num_samples]
        
        questions = [item['question'] for item in group_items]
        ground_truths = [item['gold_answer'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "question": questions,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm, 
            embeddings=embedding  
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
    
    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\nFinal Evaluation Summary:")
    print("=" * 50)
    for q_type, metrics in all_results.items():
        print(f"\nQuestion Type: {q_type}")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
    
    print('\nEvaluation complete.')

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add command-line arguments
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
        "--bge_model", 
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for BGE embeddings"
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))
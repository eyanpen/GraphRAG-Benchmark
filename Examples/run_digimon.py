import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import argparse
import asyncio
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
from Data.QueryDataset import RAGQueryDataset
from Core.Utils.Evaluation import Evaluator

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

async def initialize_rag(
    config_path: Path,
    source: str,
) -> GraphRAG:
    """Initialize GraphRAG instance for a specific source"""
    logger.info(f"🛠️ Initializing GraphRAG for source: {source}")
    
    # Parse configuration
    opt = Config.parse(config_path, dataset_name=source)
    logger.info(f"Configuration parsed: {opt}")
    
    # Create RAG instance
    rag = GraphRAG(config=opt)
    logger.info(f"✅ GraphRAG initialized for {source}")
    return rag

async def process_corpus(
    rag: GraphRAG,
    corpus_name: str,
    context: str,
    questions: Dict[str, List[dict]],
    sample: int,
    output_dir: str = "./results/GraphRAG"
):
    """Process a single corpus: index it and answer its questions"""
    logger.info(f"📚 Processing corpus: {corpus_name}")
    
    # Index the corpus
    corpus = [{
        "title": corpus_name,
        "content": context,
        "doc_id": 0,
    }]
    
    await rag.insert(corpus)
    logger.info(f"🔍 Indexed corpus: {corpus_name} ({len(context.split())} words)")
    
    corpus_questions = questions.get(corpus_name, [])
    
    if not corpus_questions:
        logger.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
        logger.info(f"🔍 Sampled {sample} questions from {len(corpus_questions)} total")
    
    logger.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{corpus_name}_predictions.json")
    
    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            response, context = await rag.query(q["question"])
            results.append({
                "id": q["id"],
                "source": corpus_name,
                "question": q["question"],
                "context": context,
                "generated_answer": response,
                "ground_truth": q.get("answer"),
                "question_type": q.get("question_type", "unknown")
            })
        except Exception as e:
            logger.error(f"❌ Failed to process question {q['id']}: {e}")
            results.append({
                "id": q["id"],
                "error": str(e)
            })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved {len(results)} predictions to: {output_path}")
    
    return results

def main():
    # Define default paths
    DEFAULT_PATHS = {
        "medical": {
            "corpus_path": "./Corpus/medical",
            "questions_path": "./Questions/medical_questions.json"
        },
        "novel": {
            "corpus_path": "./Corpus/novel",
            "questions_path": "./Questions/novel_questions.json"
        }
    }
    
    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process")
    parser.add_argument("--config", default="./config.yml", 
                        help="Path to configuration YAML file")
    parser.add_argument("--output_dir", default="./results/GraphRAG", 
                        help="Output directory for results")
    
    # Sampling and debugging
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    
    args = parser.parse_args()
    
    # Get paths for this subset
    if args.subset in DEFAULT_PATHS:
        corpus_path = DEFAULT_PATHS[args.subset]["corpus_path"]
        questions_path = DEFAULT_PATHS[args.subset]["questions_path"]
    else:
        corpus_path = f"./Corpus/{args.subset}.json"
        questions_path = f"./Questions/{args.subset}_questions.json"
        logger.warning(f"Using inferred paths for unknown subset: {args.subset}")
    
    # Load corpus data
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data: Dict[str, str] = json.load(f)
        logger.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    if args.sample:
        corpus_data = dict(list(corpus_data.items())[:1])
        logger.info(f"🔍 Sampled 1 corpus from {len(corpus_data)} total")
    
    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logger.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load questions: {e}")
        return
    
    # Initialize RAG
    rag = asyncio.run(
        initialize_rag(
            config_path=Path(args.config),
            source=args.subset
        )
    )
    
    # Process each corpus in the subset
    for corpus_name, context in corpus_data.items():
        asyncio.run(
            process_corpus(
                rag=rag,
                corpus_name=corpus_name,
                context=context,
                questions=grouped_questions,
                sample=args.sample,
                output_dir=args.output_dir
            )
        )

if __name__ == "__main__":
    main()
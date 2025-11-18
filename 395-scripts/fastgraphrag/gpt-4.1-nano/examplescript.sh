import asyncio
import os
import logging
import argparse
import json
from typing import Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


load_dotenv()

DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into 'staves' rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions



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
    logging.info(f"📚 Processing corpus: {corpus_name}")

    output_dir = f"./results/fast-graphrag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    
    try:
        logging.info(f"🔄 Loading embedding model from: {embed_model_path}")
        embedding_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        embedding_model = AutoModel.from_pretrained(embed_model_path)
        logging.info(f"✅ Successfully loaded embedding model: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load embedding model: {e}")
        return

    
    llm_service = OpenAILLMService(
        model=model_name,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    logging.info(f"✅ Using OpenAI-compatible LLM service: {model_name}")

   
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(llm_service=llm_service),
    )

    
    grag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return

    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            response = grag.query(q["question"])
            context_chunks = response.to_dict()['context']['chunks']
            contexts = [item[0]["content"] for item in context_chunks]
            predicted_answer = response.response

            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": [str(c) for c in contexts],
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": str(predicted_answer),  # ✅ 修复 JSON 序列化错误
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            logging.error(f"❌ Error processing question {q.get('id')}: {e}")
            results.append({
                "id": q["id"],
                "error": str(e)
            })

    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")



def main():
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions.parquet"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.parquet",
            "questions": "./Datasets/Questions/novel_questions.parquet"
        }
    }

    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--base_dir", default="./Examples/graphrag_workspace")
    parser.add_argument("--mode", choices=["API"], default="API")
    parser.add_argument("--model_name", default="gpt-4.1-nano")
    parser.add_argument("--embed_model_path", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1")
    parser.add_argument("--llm_api_key", default="", help="API key for LLM service")

    args = parser.parse_args()

    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(f"graphrag_{args.subset}.log")]
    )

    logging.info(f"🚀 Starting GraphRAG processing for subset: {args.subset}")

    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")

    os.makedirs(args.base_dir, exist_ok=True)

    
    try:
        corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = [{"corpus_name": item["corpus_name"], "context": item["context"]} for item in corpus_dataset]
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    
    try:
        questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
        question_data = [{
            "id": item["id"],
            "source": item["source"],
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "evidence": item["evidence"]
        } for item in questions_dataset]
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return

    async def _run_all():
        tasks = [
            asyncio.to_thread(
                process_corpus,
                item["corpus_name"],
                item["context"],
                args.base_dir,
                args.mode,
                args.model_name,
                args.embed_model_path,
                args.llm_base_url,
                api_key,
                grouped_questions,
                args.sample,
            )
            for item in corpus_data
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception(f"❌ Task failed: {r}")

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()

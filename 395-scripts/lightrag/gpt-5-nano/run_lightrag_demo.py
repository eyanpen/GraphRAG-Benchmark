
import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.hf import hf_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
import json
import nest_asyncio
import argparse
from typing import Dict, List
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from openai import AsyncOpenAI

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


## llm func 
async def gpt_complete(
    prompt,
    model_name,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


async def initialize_rag(
        base_dir: str,
        embed_model_name: str,
        source: str,
        model_name: str,
        mode: str,
        llm_base_url: str,
        **kwargs,
):
    WORKING_DIR = os.path.join(base_dir, source)
    if mode == "API":
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        embed_model = AutoModel.from_pretrained(embed_model_name)
        embedding_func = EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: hf_embed(texts, tokenizer, embed_model),
            )
        
        llm_model_func_final = lambda prompt, **kwargs: gpt_complete(
            prompt, model_name=model_name, **kwargs)
        # llm_kwargs = {
        #         "model_name": model_name,
        #         "base_url": llm_base_url,
        #         "api_key": llm_api_key
        #     }
    elif mode == "ollama":
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model_name, host=llm_base_url
            ),
        )
        llm_model_func_final = ollama_model_complete
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'API' or 'ollama'.")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=llm_model_func_final,
        llm_model_name=model_name,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        # rerank_model_func = 
        # llm_model_kwargs=llm_kwargs,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag



async def main():

    if not os.getenv("LLM_API_KEY"):
        print(
            "Error: LLM_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export LLM_API_KEY='your-llm-api-key'")
        return  # Exit the async function

    
    parser = argparse.ArgumentParser(description="LightRAG: Process Corpora and Answer Questions")
    
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                    help="Subset to process (medical or novel)")
    parser.add_argument("--model_name", default="gpt-4o-mini",required=True,
                    help="Name of the llm model to use")
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1",required=True,
                    help="Base URL for the LLM API")
    parser.add_argument("--llm_api_key",
                    help="API key for the LLM")
    parser.add_argument("--mode", default="API", choices=["API", "ollama"], help="Use API or ollama for LLM")
    parser.add_argument("--llm_model_kwargs", default={}, type=json.loads,
                    help="Additional keyword arguments for the LLM model in JSON format")
    parser.add_argument("--base_dir", default="./lightrag_workspace", help="Base working directory")
    parser.add_argument("--embed_model", default="bge-base-en", help="Embedding model name")

    # parser.add_argument("--mode", required=True, choices=["API", "ollama"], help="Use API or ollama for LLM")

    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")
    
    args = parser.parse_args()

    WORKING_DIR = os.path.join(args.base_dir, args.subset)
    os.makedirs(WORKING_DIR, exist_ok=True)
    # Initialize and run the LightRAG demo

    try: #to build graph uncomment the below code
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
            "kv_store_llm_response_cache.json",
        ]

        work_dir = os.path.join(args.base_dir, args.subset)
        for file in files_to_delete:
            file_path = os.path.join(work_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        print("Initializing RAG...")
        # Initialize RAG instance
        rag = await initialize_rag(base_dir=args.base_dir,
                                   embed_model_name=args.embed_model,
                                   source=args.subset,
                                      mode=args.mode,
                                        llm_base_url=args.llm_base_url,
                                   model_name=args.model_name,)
        print("RAG initialized.")

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

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

        if args.subset not in SUBSET_PATHS:
            logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
            return
        if args.mode not in ["API", "ollama"]:
            logging.error(f'Invalid mode: {args.mode}. Valid options: {["API", "ollama"]}')
            return
        cpath = SUBSET_PATHS[args.subset]["corpus"]
        with open(cpath, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        for c in corpus:
            try:
                await rag.ainsert(c.get("context").strip())
            except Exception as e:
                print(f"Error inserting context id {c.get('id')}: {e}")
        
        

        qpath = SUBSET_PATHS[args.subset]["questions"]

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        with open(qpath, "r", encoding="utf-8") as f:
            questions = json.load(f)

        out_dir = os.path.join(args.base_dir, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "medical_results_naive.json")

        naive_results = []
        for q in questions[0]:
            id = q.get("id")
            question = q.get("question")
            question_type = q.get("question_type")
            evidence = q.get("evidence")
            source = q.get("source", "")
            ground_truth = q.get("answer")
            record = {"id": id, 
                      "question": question, 
                      "source": source, 
                      "question_type": question_type, 
                      "evidence": evidence, 
                      "ground_truth": ground_truth,
                      "generated_answer": None, 
                      "data": {}}
            try:
                res = await rag.aquery_llm( #if switch to aquery will get only text answer, no other llm outputs
                        question.strip(), param=QueryParam(mode="naive")) # using naive will not output entity and rel, but hybrid will
                
                if isinstance(res, dict):
                    # extract llm text (various possible keys)
                    llm_resp = res.get("llm_response") or res.get("llm") or {}
                    answer_text = (
                        llm_resp.get("content")
                        or llm_resp.get("text")
                        or res.get("text")
                        or res.get("answer")
                        or str(res)
                    )
                    data = res.get("data") or {}
                else:
                    answer_text = str(res)
                    data = {}
                chunks = data.get("chunks", []) or []
                entities = data.get("entities", []) or []
                relations = data.get("relationships", []) or data.get("relations", []) or []
                record["generated_answer"] = answer_text
                record["data"]["chunks"] = chunks
                record["data"]["entities"] = entities
                record["data"]["relationships"] = relations

            except Exception as e:
                answer_text = f"ERROR: {e}"
                record["generated_answer"] = answer_text
                record["data"] = {}
            naive_results.append(record)

        # save results
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(naive_results, outf, ensure_ascii=False, indent=2)

        print(f"Saved results to {out_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()
    

if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")

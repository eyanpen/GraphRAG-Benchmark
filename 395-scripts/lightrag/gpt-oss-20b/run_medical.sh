#!/bin/bash
set -e

python 395-scripts/lightrag/run_lightrag_ollama.py \
    --subset medical \ 
    --mode ollama \
    --model_name gpt-oss:20b \
    --embed_model bge-m3 \
    --base_dir 395-scripts/lightrag/gpt-oss-20b \
    --llm_base_url http://localhost:11434
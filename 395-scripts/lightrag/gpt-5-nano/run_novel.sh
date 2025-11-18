#!/bin/bash
set -e

python 395-scripts/lightrag/lightrag_test.py \
    --subset novel \
    --model_name gpt-5-nano \
    --mode API \
    --llm_base_url https://api.openai.com/v1 \
    --embed_model BAAI/bge-large-en-v1.5 \
    --base_dir 395-scripts/lightrag/gpt-5-nano

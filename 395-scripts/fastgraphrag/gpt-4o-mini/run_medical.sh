
python 395-scripts/fastgraphrag/testrun_fast-graphrag.py \
  --subset medical \
  --mode API \
  --model_name gpt-4o-mini \
  --embed_model_path BAAI/bge-large-en-v1.5 \
  --llm_base_url https://api.openai.com/v1 \
  --llm_api_key \
  --base_dir 394-scripts/fastgraphrag/gpt-4o-mini


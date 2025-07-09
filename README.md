# GraphRAG-Bench

This repository is for the GraphRAG-Bench project, a comprehensive benchmark for evaluating Graph Retrieval-Augmented Generation models.
![pipeline](./pipeline.jpg)

## 1. Website Overview

### 🎉 News
- **[2025-05-25]** We release [GraphRAG-Bench](https://graphrag-bench.github.io), the benchmark for evaluating GraphRAG models.
- **[2025-05-14]** We release the [GraphRAG-Bench dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).
- **[2025-01-21]** We release the [GraphRAG survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

### 📖 About
- Introduces Graph Retrieval-Augmented Generation (GraphRAG) concept
- Compares traditional RAG vs GraphRAG approach
- Explains research objective: Identify scenarios where GraphRAG outperforms traditional RAG
- Visual comparison diagram of RAG vs GraphRAG

![overview](./RAGvsGraphRAG.jpg)


### 🏆 Leaderboards
Two domain-specific leaderboards with comprehensive metrics:

**1. GraphRAG-Bench (Novel)**
- Evaluates models on literary/fictional content

**2. GraphRAG-Bench (Medical)**
- Evaluates models on medical/healthcare content

**Evaluation Dimensions:**
- Fact Retrieval (Accuracy, ROUGE-L)
- Complex Reasoning (Accuracy, ROUGE-L)
- Contextual Summarization (Accuracy, Coverage)
- Creative Generation (Accuracy, Factual Score, Coverage)

### 🧩 Task Examples
Four difficulty levels with representative examples:

**Level 1: Fact Retrieval**  
*Example: "Which region of France is Mont St. Michel located?"*

**Level 2: Complex Reasoning**  
*Example: "How did Hinze's agreement with Felicia relate to the perception of England's rulers?"*

**Level 3: Contextual Summarization**  
*Example: "What role does John Curgenven play as a Cornish boatman for visitors exploring this region?"*

**Level 4: Creative Generation**  
*Example: "Retell King Arthur's comparison to John Curgenven as a newspaper article."*


## 2. Access the Website
Our benchmark was released:  
[**https://graphrag-bench.github.io**](https://graphrag-bench.github.io)

## 3. Getting Started(GraphRAG-Bench Examples)
The  **'Examples'** directory contains example implementations for running inference on the GraphRAG-Bench dataset using various RAG frameworks. Each framework has a dedicated implementation file that generates prediction outputs compatible with our unified evaluation pipeline.

### 🛠 Installation Guide

**To prevent dependency conflicts, we strongly recommend using separate Conda environments for each framework:**

We use the installation of LightRAG as an example. For other frameworks, please refer to their respective installation instructions.
```bash
# Create and activate environment (example for LightRAG)
conda create -n lightrag python=3.10 -y
conda activate lightrag

# Install LightRAG
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .

```

### 🚀 Running Example
Next, we provide detailed instructions on how to use GraphRAG-Bench to evaluate each framework. Specifically, we introduce how to perform index construction and batch inference for each framework. Note that the evaluation code is standardized across all frameworks to ensure fair comparison.
#### 1. Indexing and inference
##### a. LightRAG
Before running the above script, you need to modify the source code(LightRAG) to enable extraction of the corresponding context used during generation. Please make the following changes:
1. In lightrag/operate.py, update the kg_query method to return the context along with the response:
```python
# Original Code
async def kg_query(...) -> str | AsyncIterator[str]:
  return response

# Modified Code
async def kg_query(...) -> tuple[str, str] | tuple[AsyncIterator[str], str]:
  return response, context
```
2. In lightrag/lightrag.py, update the aquery method to receive and return the context when calling kg_query:
```python
# Modified Code
async def aquery(...):
  ...
  if param.mode in ["local", "global", "hybrid"]:
      response, context = await kg_query(...)
  ...
  return response, context

```
Then you can run the following command to indexing and inference:
```shell
export LLM_API_KEY=your_actual_api_key_here

python run_lightrag.py \
  --subset medical \
  --base_dir ./Examples/lightrag_workspace \
  --model_name bge-large-en-v1.5 \
  --embed_model bge-base-en \
  --retrieve_topk 5 \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1

```
##### b. fast-graphrag
```shell
export LLM_API_KEY=your_actual_api_key_here

python run_fast-graphrag.py \
  --subset medical \
  --base_dir ./Examples/fast-graphrag_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path bge-large-en-v1.5 \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1

```

##### c. hipporag2

```shell
export OPENAI_API_KEY=your_actual_api_key_here

python run_hipporag2.py \
  --subset medical \
  --base_dir ./Examples/hipporag2_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path contriever \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1
```
We will continue updating other GraphRAG frameworks as much as possible. If you wish to integrate a different framework, you can refer to the structure of our result format. As long as your returned output matches the following fields, the evaluation code will run successfully:
```json
{
  "id": q["id"],
  "question": q["question"],
  "source": corpus_name,
  "context": context,
  "evidence": q["evidence"],
  "question_type": q["question_type"],
  "generated_answer": predicted_answer,
  "gold_answer": q["answer"]
}

```
#### 2. Evaluation
##### a. Generation
```shell
cd Evaluation
export OPENAI_API_KEY=your_actual_api_key_here

python -m Evaluation.generation_eval \
  --model gpt-4-turbo \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/evaluation_results.json
```

##### b. Retrieval
```shell
cd Evaluation
export OPENAI_API_KEY=your_actual_api_key_here

python -m Evaluation.retrieval_eval \
  --model gpt-4-turbo \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/evaluation_results.json
```

## 4. Contribution & Contact
Contributions to improve the benchmark website are welcome. Please contact the project team via <a href="mailto:GraphRAG@hotmail.com">GraphRAG@hotmail.com</a>.

## 5. Citation
If you find this benchmark helpful, please cite our paper:
```
@article{xiang2025use,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}
```
## 6.Stars History
![history](./star-history.png)

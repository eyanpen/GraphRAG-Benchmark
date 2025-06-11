# GraphRAG-Bench Website

This repository hosts the official website for the GraphRAG-Bench project, a comprehensive benchmark for evaluating Graph Retrieval-Augmented Generation models.
![pipeline](./pipeline.jpg)

## Website Overview

### 🎉 News
- **[2025-05-25]** We release [GraphRAG-Bench](https://graphrag-bench.github.io), the benchmark for evaluating GraphRAG models.
- **[2025-05-14]** We release the [GraphRAG-Bench dataset](https://www.kaggle.com/datasets/wuchuanjie/graphrag-bench).
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

### 📬 Contact
- Project email: [GraphRAG@hotmail.com](mailto:GraphRAG@hotmail.com)

## Access the Website
Our benchmark was released:  
[**https://graphrag-bench.github.io**](https://graphrag-bench.github.io)

## Getting Started
```bash
git clone https://github.com/GraphRAG-Bench/GraphRAG-Bench.git](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git
cd GraphRAG-Benchmark
# Open index.html in browser
```

## Contribution
Contributions to improve the benchmark website are welcome. Please contact the project team via email.

# 🍀 Citation
If you find this survey helpful, please cite our paper:
```
@misc{xiang2025usegraphsragcomprehensive,
      title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation}, 
      author={Zhishang Xiang and Chuanjie Wu and Qinggang Zhang and Shengyuan Chen and Zijin Hong and Xiao Huang and Jinsong Su},
      year={2025},
      eprint={2506.05690},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.05690}, 
}
```

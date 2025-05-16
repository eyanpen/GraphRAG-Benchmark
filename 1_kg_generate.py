from kg_gen_updated import KGGen
import asyncio
from utils.kg_gen.models import Graph
import os
import json
from pathlib import Path

async def generate_graph(kg, input_file, file_name, cache_dir, chunk_size=None, cluster=True, overwrite=False):
    with open(input_file, 'r') as f:
        large_text = f.read()
    # 如果存在cache则读取
    cache_path = f"{cache_dir}/{file_name}.json"
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path) and not overwrite:
        graph = Graph.load_from_cache(cache_path)
    else:
        if chunk_size is None:
            graph = await kg.generate(
                input_data=large_text,
                cluster=cluster
            )
        else:
            graph = await kg.generate(
                input_data=large_text,
                chunk_size=chunk_size,  # Process text in chunks of 5000 chars
                cluster=cluster
            )
        graph.save_to_cache(cache_path)
    return graph

async def save_graph_info(graph, save_dir):
    subgraphs = graph.extract_largest_subgraphs(top_k=5)
    os.makedirs(save_dir, exist_ok=True)
    save_res = []

    for i, subgraph in enumerate(subgraphs):
        subgraph_info = {
            'entities': list(subgraph.entities),
            'edges': [list(e) for e in subgraph.edges],
            'relations': [list(r) for r in subgraph.relations],
            'triples': [list(t) for t in subgraph.to_triples()]
        }
        save_res.append(subgraph_info)
        subgraph.visualize_interactive(f"{save_dir}/subgraph_{i}.html")

    with open(f"{save_dir}/subgraph_info.json", 'w') as f:
        json.dump(save_res, f, indent=2)


async def main():
    # file_list = ['test_30752_processed','test_26183_processed','test_26239_processed', 'test_29973_processed', 'test_24553_processed', 'test_25646_processed', 'test_28444_processed', 'test_2544_processed']
    file_list = []
    text_type = "pg19/pretend"
    text_dir = Path(text_type)
    base_cache_dir = Path(f"cache/llama-70b/kg_gen/{text_type}")
    save_root_dir = Path(f"cache/llama-70b/kg_gen/{text_type}")

    # 初始化 KG 生成器
    # kg = KGGen(api_key="sk-1238825ff1e4446f87053832bbfc818d", base_url="https://api.deepseek.com", model="deepseek-chat", max_concurrent=20, temperature=1.0)
    kg = KGGen(api_key="0", base_url="http://0.0.0.0:8003/v1", model="moonshot-v1-auto", max_concurrent=50, temperature=1)

    # 遍历目录下所有 txt 文件
    for file_path in text_dir.glob("*.txt"):
        text_name = file_path.stem  # 提取文件名（不含扩展名）
        if "processed" not in text_name:
            continue
        if text_name in file_list:
            continue
        print(f"Processing {text_name}...")
        input_file = str(file_path)
        cache_dir = base_cache_dir / text_name

        # 异步生成图
        graph = await generate_graph(kg, input_file, text_name, cache_dir, cluster=False, chunk_size=5000, overwrite=False)

        # 保存图信息
        save_dir = save_root_dir / text_name
        subgraph_save_dir = save_root_dir / text_name / "subgraph"
        await save_graph_info(graph, subgraph_save_dir)

        graph.visualize_interactive(f"{save_dir}/whole_graph.html")

        print(f"Graph for {text_name} generated and saved to {save_dir}")

if __name__ == "__main__":
    asyncio.run(main())
# # Output: 
# # entities={'Linda', 'Ben', 'Andrew', 'Josh'} 
# # edges={'is brother of', 'is father of', 'is mother of'} 
# # relations={('Ben', 'is brother of', 'Josh'), 
# #           ('Andrew', 'is father of', 'Josh'), 
# #           ('Linda', 'is mother of', 'Josh')}

# EXAMPLE 2: Large text with chunking and clustering
# with open('pg19/test_10146.txt', 'r') as f:
#   large_text = f.read()
  
# Example input text:
# """
# Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
# that uses multiple layers of neural networks. Supervised learning requires training data to learn
# patterns. Machine learning is a type of AI technology that enables computers to learn from data.
# AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
# Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
# many fields of study.
# ...
# """

# graph_2 = kg.generate(
#   input_data=large_text,
#   chunk_size=5000,  # Process text in chunks of 5000 chars
#   cluster=True      # Cluster similar entities and relations
# )
# Output:
# entities={'neural networks', 'deep learning', 'machine learning', 'AI', 'artificial intelligence', 
#          'supervised learning', 'unsupervised learning', 'training data', ...} 
# edges={'is type of', 'requires', 'is subset of', 'uses', 'is related to', ...} 
# relations={('neural networks', 'is type of', 'machine learning'),
#           ('deep learning', 'is subset of', 'machine learning'),
#           ('supervised learning', 'requires', 'training data'),
#           ('machine learning', 'is type of', 'AI'),
#           ('AI', 'is related to', 'artificial intelligence'), ...}
# entity_clusters={
#   'artificial intelligence': {'AI', 'artificial intelligence'},
#   'machine learning': {'machine learning', 'ML'},
#   'neural networks': {'neural networks', 'neural nets', 'NN'}
#   ...
# }
# edge_clusters={
#   'is type of': {'is type of', 'is a type of', 'is a kind of'},
#   'is related to': {'is related to', 'is connected to', 'is associated with'
#  ...}
# }

# # EXAMPLE 3: Messages array
# messages = [
#   {"role": "user", "content": "What is the capital of France?"}, 
#   {"role": "assistant", "content": "The capital of France is Paris."}
# ]
# graph_3 = kg.generate(input_data=messages)
# # Output: 
# # entities={'Paris', 'France'} 
# # edges={'has capital'} 
# # relations={('France', 'has capital', 'Paris')}

# # EXAMPLE 4: Combining multiple graphs
# text1 = "Linda is Joe's mother. Ben is Joe's brother."

# # Input text 2: also goes by Joe."
# text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

# graph4_a = kg.generate(input_data=text1)
# graph4_b = kg.generate(input_data=text2)

# # Combine the graphs
# combined_graph = kg.aggregate([graph4_a, graph4_b])

# # Optionally cluster the combined graph
# clustered_graph = kg.cluster(
#   combined_graph,
#   context="Family relationships"
# )
# # Output:
# # entities={'Linda', 'Ben', 'Andrew', 'Joe', 'Joseph', 'Judy'} 
# # edges={'is mother of', 'is father of', 'is brother of', 'is sister of'} 
# # relations={('Linda', 'is mother of', 'Joe'),
# #           ('Ben', 'is brother of', 'Joe'),
# #           ('Andrew', 'is father of', 'Joe'),
# #           ('Judy', 'is sister of', 'Andrew')}
# # entity_clusters={
# #   'Joe': {'Joe', 'Joseph'},
# #   ...
# # }
# # edge_clusters={ ... }
import json
import os
from utils.kg_gen.models import Graph
from tqdm import tqdm

def extract_contexts_from_items(data):
    """从每个 item 中提取 context 字符串，返回一个字符串列表"""
    # sentences = []
    # for triple in data:
    #     if len(triple) == 3:
    #         s, p, o = triple
    #         sentence = f"({s},{p},{o})"
    #         sentences.append(sentence)
    # context = ".".join(sentences)   
    contexts = []     
    for item in data:
        sentences = []
        for triple in item:
            if len(triple) == 3:
                s, p, o = triple
                sentence = f"({s},{p},{o})"
                sentences.append(sentence)
        context = ".".join(sentences)
        contexts.append(context)
    return contexts


def process_graph_folder(root_dir, output_path):
    result = []

    folders = os.listdir(root_dir)
    for folder_name in tqdm(folders, desc="Processing graphs"):  # 加上进度条
        folder_path = os.path.join(root_dir, folder_name)
        graph_path = os.path.join(folder_path, f"{folder_name}.json")
        
        if os.path.isdir(folder_path) and os.path.exists(graph_path):
            graph = Graph.load_from_cache(graph_path)
            subgraphs = graph.extract_largest_subgraphs(top_k=5)
            subgraph_tripples = []
            rated_subgraph_tripples = []
            # subgraph_logic_lines = []
            for subgraph in subgraphs:
                subgraph_tripples.append(subgraph.to_triples())
                small_subgraphs = subgraph.split_graph(50)
                for small_subgraph in small_subgraphs:
                    rated_subgraph_tripples.append(small_subgraph.to_triples())
                # subgraph_logic_lines.extend(subgraph.find_topk_longest_chains(5))
            
            # # 排序并筛选Top-k
            # subgraph_logic_lines.sort(key=lambda path: (-len(path), path))  # 先按长度降序，再按字典序

            # topk_logic_lines = subgraph_logic_lines[:15]

            # subgraph_contexts = extract_contexts_from_items(subgraph_tripples)
            # graph_logic_contexts = extract_contexts_from_items(topk_logic_lines)
            result.append({
                "id": folder_name,
                "subgraph": subgraph_tripples,
                "rated_subgraph": rated_subgraph_tripples,
                "num_rated_subgraph": len(rated_subgraph_tripples),
                # "logic": graph_logic_contexts
            })

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, ensure_ascii=False, indent=2)

    print(f"Saved {len(result)} items to {output_path}")


def main():
    root_dir = "cache/llama-70b/kg_gen/pg19/now"
    output_path = "cache/llama-70b/kg_gen/pg19/now/pg19_infos.json"
    
    process_graph_folder(root_dir, output_path)

if __name__ == "__main__":
    main()
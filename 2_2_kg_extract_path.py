import json
import networkx as nx
from tqdm import tqdm
import signal
from itertools import islice

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()


def build_graph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    return G

def get_top_degree_nodes(G, top_k):
    degree_scores = {
        node: G.in_degree(node) + G.out_degree(node)
        for node in G.nodes
    }
    # 取度数最大的 top_k 个节点
    top_nodes = sorted(degree_scores, key=degree_scores.get, reverse=True)[:top_k]
    return top_nodes


# 注册超时信号处理器
signal.signal(signal.SIGALRM, timeout_handler)

def find_paths(
    G,
    k,
    max_len=50,
    top_k_sources=10,
    top_k_paths=30,
    max_paths_per_pair=20,
    timeout_per_pair=5  # 每个 source-target 对最多允许计算 3 秒
):
    all_paths = []
    source_nodes = get_top_degree_nodes(G, top_k_sources)
    nodes = list(G.nodes)

    for source in tqdm(source_nodes, desc="  Source nodes", leave=False):
        for target in nodes:
            if source == target:
                continue
            try:
                signal.alarm(timeout_per_pair)  # 设置超时
                paths = islice(
                    nx.all_simple_paths(G, source=source, target=target, cutoff=max_len),
                    max_paths_per_pair
                )
                for path in paths:
                    if len(path) - 1 >= k:
                        edge_path = [
                            (path[i], G[path[i]][path[i+1]]['relation'], path[i+1])
                            for i in range(len(path) - 1)
                        ]
                        all_paths.append(edge_path)
            except TimeoutException:
                print(f"Timeout: skipping path from {source} to {target}")
            except Exception as e:
                print(f"Error: {e}")
            finally:
                signal.alarm(0)  # 关闭超时计时器

    # 计算路径得分：路径中每个节点的入度 + 出度
    def path_score(path):
        nodes_in_path = set()
        for h, _, t in path:
            nodes_in_path.add(h)
            nodes_in_path.add(t)
        return sum(G.in_degree(n) + G.out_degree(n) for n in nodes_in_path)

    # 保留 top_k_paths 条高得分路径
    all_paths.sort(key=path_score, reverse=True)
    return all_paths[:top_k_paths]



def process_json_file(input_file, output_file, k):
    with open(input_file, 'r') as f:
        data = json.load(f)

    output_data = []

    for item in tqdm(data, desc="Processing graphs"):
        if item['id'] not in ['test_10762_processed']:
            continue
        subgraph = item.get("subgraph", [])
        G = build_graph(subgraph[0])
        paths = find_paths(G, k)
        print(f"{item['id']}: {len(paths)} paths found from top-10 degree nodes")
        output_data.append({
            "id": item["id"],
            "paths": paths
        })
        
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    input_file = "/home/xzs/data/experient/benchmark/cache/llama-70b/kg_gen/pg19/pg19_infos_valid.json"  # Replace with your input file path
    output_file = "/home/xzs/data/experient/benchmark/cache/llama-70b/kg_gen/pg19/pg19_infos_valid_path_third.json"  # Replace with your output file path
    min_path_len = 5  # Minimum path length

    process_json_file(input_file, output_file, min_path_len)

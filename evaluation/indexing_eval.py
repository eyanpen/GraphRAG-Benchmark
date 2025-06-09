import os
import igraph as ig
import pandas as pd
import numpy as np

def analyze_graph(g: ig.Graph):
    """
    Analyze a single igraph.Graph object and compute related metrics.

    :param g: igraph.Graph object
    :return: A dictionary containing graph metrics
    """
    num_nodes = g.vcount()
    num_edges = g.ecount()
    average_degree = sum(g.degree()) / num_nodes if num_nodes > 0 else 0
    density = g.density()
    components = g.components()
    num_components = len(components)
    largest_component_size = components.giant().vcount()
    average_clustering_coefficient = g.transitivity_avglocal_undirected()
    diameter = g.diameter() if g.is_connected() else float('inf')

    # Compute component-level statistics excluding isolated nodes
    component_sizes = [len(component) for component in components if len(component) > 1]
    if component_sizes:
        average_component_size = sum(component_sizes) / len(component_sizes)
        median_component_size = np.median(component_sizes)
        num_components_excluding_isolated = len(component_sizes)
        num_components_above_average = sum(1 for size in component_sizes if size > average_component_size)
        num_nodes_excluding_isolated = sum(component_sizes)

        component_sizes_sorted = sorted(component_sizes)
        trimmed_mean_component_size = (
            sum(component_sizes_sorted[1:-1]) / (len(component_sizes_sorted) - 2)
            if len(component_sizes_sorted) > 2 else average_component_size
        )

        geometric_mean_component_size = np.exp(np.mean(np.log(component_sizes)))
        harmonic_mean_component_size = len(component_sizes) / sum(1.0 / size for size in component_sizes)
    else:
        average_component_size = 0
        median_component_size = 0
        num_components_excluding_isolated = 0
        num_components_above_average = 0
        num_nodes_excluding_isolated = 0
        trimmed_mean_component_size = 0
        geometric_mean_component_size = 0
        harmonic_mean_component_size = 0

    degrees = g.degree(mode="all")

    num_isolated_nodes = sum(1 for d in degrees if d == 0)
    num_nodes_excluding_isolated = sum(1 for d in degrees if d > 0)
    num_nodes_degree_above_1 = sum(1 for d in degrees if d > 1)
    num_nodes_degree_above_2 = sum(1 for d in degrees if d > 2)
    num_nodes_degree_above_3 = sum(1 for d in degrees if d > 3)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": average_degree,
        "density": density,
        "num_components": num_components,
        "largest_component_size": largest_component_size,
        "average_clustering_coefficient": average_clustering_coefficient,
        "diameter": diameter,
        "average_component_size": average_component_size,
        "median_component_size": median_component_size,
        "trimmed_mean_component_size": trimmed_mean_component_size,
        "geometric_mean_component_size": geometric_mean_component_size,
        "harmonic_mean_component_size": harmonic_mean_component_size,
        "num_components_excluding_isolated": num_components_excluding_isolated,
        "num_components_above_average": num_components_above_average,
        "num_nodes_excluding_isolated": num_nodes_excluding_isolated,
        "num_isolated_nodes": num_isolated_nodes,
        "num_nodes_degree_above_1": num_nodes_degree_above_1,
        "num_nodes_degree_above_2": num_nodes_degree_above_2,
        "num_nodes_degree_above_3": num_nodes_degree_above_3
    }

# You need to implement this function if using 'entities.parquet' and 'relationships.parquet'
def load_graph_from_parquet(entities_path, relationships_path):
    """
    Load a graph from entity and relationship parquet files.

    :param entities_path: Path to the entities.parquet file
    :param relationships_path: Path to the relationships.parquet file
    :return: An igraph.Graph object
    """
    # Example implementation (customize this as needed)
    df_entities = pd.read_parquet(entities_path)
    df_relationships = pd.read_parquet(relationships_path)

    # Ensure the necessary columns exist
    if 'id' not in df_entities.columns or 'source' not in df_relationships.columns or 'target' not in df_relationships.columns:
        raise ValueError("Missing required columns in parquet files.")

    # Create graph from source-target edges
    g = ig.Graph()
    g.add_vertices(df_entities['id'].astype(str).tolist())
    edges = list(zip(df_relationships['source'].astype(str), df_relationships['target'].astype(str)))
    g.add_edges(edges)

    return g

def process_graphs(base_path, folder_name):
    """
    Traverse subdirectories under the base path, locate graph files, and compute metrics.

    :param base_path: Root directory containing subfolders
    :param folder_name: Folder name where graph files are expected
    :return: List of metric dictionaries, one per graph
    """
    results = []

    for subdir, dirs, files in os.walk(base_path):
        # Example: Microsoft GraphRAG format (entities.parquet + relationships.parquet)
        entities_path = os.path.join(subdir, 'entities.parquet')
        relationships_path = os.path.join(subdir, 'relationships.parquet')
        if os.path.exists(entities_path) and os.path.exists(relationships_path):
            g = load_graph_from_parquet(entities_path, relationships_path)
            result = analyze_graph(g)
            results.append(result)

        # Uncomment the following block for HippoRAG
        # target_folder = os.path.join(subdir, folder_name)
        # if os.path.exists(target_folder):
        #     graph_path = os.path.join(target_folder, 'graph.pickle')
        #     if os.path.exists(graph_path):
        #         g = ig.Graph.Read_Pickle(graph_path)
        #         result = analyze_graph(g)
        #         results.append(result)

        # Uncomment for LightRAG / Fast-GraphRAG
        # graph_path = os.path.join(subdir, 'graph_igraph_data.pklz')
        # if os.path.exists(graph_path):
        #     g = ig.Graph.Read_Picklez(graph_path)
        #     result = analyze_graph(g)
        #     results.append(result)

    return results

def calculate_average(results):
    """
    Compute the average value for each metric across all graphs.

    :param results: List of metric dictionaries
    :return: Dictionary with average values for each metric
    """
    if not results:
        return {}

    avg_results = {key: 0 for key in results[0].keys()}

    for result in results:
        for key, value in result.items():
            avg_results[key] += value

    num_graphs = len(results)
    for key in avg_results:
        avg_results[key] /= num_graphs

    return avg_results

if __name__ == "__main__":
    base_path = "./graphrag_nccn/output"  # Replace with your base path
    folder_name = ""  # Replace if needed

    all_results = process_graphs(base_path, folder_name)
    average_results = calculate_average(all_results)

    print("Average metrics across all graphs:")
    for key, value in average_results.items():
        print(f"  {key}: {value}")

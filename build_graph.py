from utils.utils_nx import *
from utils.utils_common import load_graph

"""
Main pipeline to build and analyze the job-job graph.

Steps:
    1. Build or load the graph:
        - Nodes: Job listings
        - Edges: Connections between jobs via user interactions
        - Weights: Click (low), Purchase (high)

    2. Analyze graph:
        - Basic statistics (nodes, edges, density)
        - Top connected jobs (degree)
        - Centrality analysis (betweenness)
        - Largest connected component
"""

def main(load_saved_files=False):
    """
    Main pipeline to build and analyze a job-job graph based on user interactions.

    Args:
        load_saved_files (bool, optional): 
            If True, load the saved graph. 
            If False, build and save it. Defaults to False.

    Returns:
        None
    """
    # File paths
    user_data_path = "data/user_event_data.csv"
    graph_path = "results/graph.pkl"

    # Load user interaction data
    user_data = load_user_data(user_data_path)

    # Build or load graph
    if not load_saved_files:
        G = create_monopartite_graph(user_data)
        save_graph(G, graph_path)
    else: G = load_graph(graph_path)

    # Basic graph statistics (nodes, edges, density)
    basic_graph_analysis(G)

    # Find top connected jobs (nodes with highest degree)
    get_top_nodes_by_degree(G)

    # Compute betweenness centrality to identify important connector jobs
    betweenness_centrality = compute_centrality(G)  # Takes ~2 minutes

    # Identify the largest connected component of the graph
    largest_cc = get_largest_connected_component(G)

if __name__ == "__main__":
    # Set to True to load saved graph instead of building a new one
    load_saved_files = False

    # Run main pipeline
    main(load_saved_files)

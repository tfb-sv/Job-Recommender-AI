import time
import pickle
import pandas as pd
import networkx as nx
from itertools import combinations

def load_user_data(file_path):
    """
    Load user interaction dataset from a CSV file and remove duplicates.

    Args:
        file_path (str): Path to the CSV file containing user interactions.

    Returns:
        DataFrame: Cleaned DataFrame of user interactions.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.drop_duplicates(keep='first').reset_index(drop=True)

    return df

def create_monopartite_graph(user_data):
    """
    Create a job-job (monopartite) graph based on user interactions.
    Jobs are connected if they are co-clicked or co-purchased by the same user.
    Edge weights are based on interaction type and frequency.

    Args:
        user_data (DataFrame): DataFrame containing user interactions (click/purchase).

    Returns:
        networkx.Graph: Generated job-job graph with weighted edges.
    """
    # Define interaction weights (higher weight for purchases)
    event_weights = {"click": 1, "purchase": 3}
    G = nx.Graph()

    # Map each user to their interacted jobs with weights
    user_clicks = {}
    for _, row in user_data.iterrows():
        user_id = row['client_id']
        item_id = row['item_id']
        event_type = row['event_type']
        weight = event_weights[event_type]  # Weight based on event type
        if user_id not in user_clicks: user_clicks[user_id] = []
        user_clicks[user_id].append((item_id, weight))

    # Connect jobs co-interacted by the same user
    for job_list in user_clicks.values():
        for (job1, w1), (job2, w2) in combinations(job_list, 2):
            total_weight = (w1 + w2) / 2  # Average interaction weight
            if G.has_edge(job1, job2): G[job1][job2]['weight'] += total_weight  # Increment existing edge weight
            else: G.add_edge(job1, job2, weight=total_weight)  # Create new edge
    print("Graph generated!")

    return G

def basic_graph_analysis(G):
    """
    Print basic statistics about the graph.

    Args:
        G (networkx.Graph): Graph object to analyze.

    Returns:
        None
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Measure of how many edges are in the graph relative to max possible
    density = nx.density(G)

    print(f"Total number of nodes: {num_nodes}")
    print(f"Total number of edges: {num_edges}")
    print(f"Graph density: {density:.4f}")

def compute_centrality(G, sample_size=1000):
    """
    Compute betweenness centrality for a subset of nodes (approximate if graph is large).

    Args:
        G (networkx.Graph): Graph to compute centrality on.
        sample_size (int, optional): Number of nodes to sample for approximation. Default is 1000.

    Returns:
        dict: Dictionary of node IDs and their centrality scores.
    """
    start_time = time.time()

    # Approximate betweenness centrality (using sampling if necessary)
    centrality = nx.betweenness_centrality(G, k=sample_size)

    # Calculate and display execution time
    end_time = time.time()
    print(f"\nExecution time for betweenness centrality calculation: {end_time - start_time:.2f} seconds")

    return centrality

def get_largest_connected_component(G):
    """
    Find and return the largest connected component in the graph.

    Args:
        G (networkx.Graph): Graph object to analyze.

    Returns:
        set: Set of nodes in the largest connected component.
    """
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"Size of the largest connected component: {len(largest_cc)}")

    return largest_cc

def get_top_nodes_by_degree(G, top_n=10):
    """
    Print the top-N most connected nodes (highest degree), indicating popular job listings.

    Args:
        G (networkx.Graph): Graph object.
        top_n (int, optional): Number of top nodes to display. Default is 10.

    Returns:
        None
    """
    # Compute degree (number of connections) for each node
    degree_dict = dict(G.degree())
    
    # Sort nodes by degree in descending order
    sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)

    # Check if node is a job listing
    print("\nTop 10 most connected nodes:")
    for node, degree in sorted_degree[:top_n]:
        print(f"Node {node}: {degree} connections")

def save_graph(G, filename):
    """
    Save the graph object to a pickle file for later use.

    Args:
        G (networkx.Graph): Graph object to save.
        filename (str): Path to save the graph file.

    Returns:
        None
    """
    with open(filename, "wb") as f: pickle.dump(G, f)
    print("Graph saved.")

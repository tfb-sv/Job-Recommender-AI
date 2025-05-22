import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def load_graph(file_path):
    """
    Load a NetworkX graph from a pickle file.

    Args:
        file_path (str): Path to the pickle file containing the graph.

    Returns:
        networkx.Graph: Loaded graph object.

    Raises:
        ValueError: If the graph is empty (no nodes or edges).
    """
    with open(file_path, "rb") as f: G = pickle.load(f)
    print(f"Graph loaded.")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Ensure graph is not empty
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError("Graph is empty! Ensure G is properly constructed.")

    return G

def load_job_data(file_path):
    """
    Load job posting dataset from a CSV file and remove duplicates.

    Args:
        file_path (str): Path to the CSV file containing job data.

    Returns:
        pandas.DataFrame: Cleaned DataFrame containing job postings.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.drop_duplicates(keep='first').reset_index(drop=True)

    return df

def get_graph_job_ids(G):
    """
    Extract job IDs (integer nodes) from the graph.

    Args:
        G (networkx.Graph): Graph object containing job and user nodes.

    Returns:
        list: List of job IDs (integers) present as nodes in the graph.
    """
    return [node for node in G.nodes if isinstance(node, int)]

def compute_similarity(embeddings, job_ids, model_type, G):
    """
    Compute cosine similarity matrix for job embeddings.

    Args:
        embeddings (list): List of job embeddings.
        job_ids (list): List of job IDs.
        model_type (str): 'NLP' for text-based, 'GNN' for graph-based embeddings.
        G (networkx.Graph): Graph object, required if model_type is 'GNN'.

    Returns:
        dict: Dictionary containing similarity scores between jobs.
    """
    start_time = time.time()

    # Prepare job embeddings matrix depending on the model type
    if model_type == "NLP":
        job_matrix = np.array(embeddings)
    elif model_type == "GNN":
        job_ids = get_graph_job_ids(G)
        job_index_mapping = {job_id: i for i, job_id in enumerate(job_ids)}
        job_matrix = np.array([embeddings[job_index_mapping[job_id]] for job_id in job_ids])

    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(job_matrix)

    # Build similarity dictionary for each job
    job_similarities = {}
    for i, job_id in enumerate(tqdm(job_ids, desc="Similarities", unit="job")):
        similarities = {
            job_ids[j]: cosine_sim_matrix[i, j]
            for j in range(len(job_ids)) if i != j  # Exclude self-similarity
        }
        job_similarities[job_id] = similarities
    end_time = time.time()

    # Calculate and display execution time
    execution_time = end_time - start_time
    print(f"Cosine similary computation time: {execution_time:.2f} seconds")

    return job_similarities

def get_similar_jobs(job_id, job_similarities, job_titles, top_n):
    """
    Find and return the top-N most similar jobs for a given job ID based on similarity scores.

    Args:
        job_id (int): ID of the job for which similar jobs are to be found.
        job_similarities (dict): Dictionary where each key is a job ID and the value is another dictionary 
                                 of job IDs with their similarity scores.
        job_titles (dict): Mapping of job IDs to their job titles for display purposes.
        top_n (int): Number of similar jobs to retrieve.

    Returns:
        list: List of tuples containing (job_id, similarity score) for the top-N similar jobs.

    Raises:
        str: If the given job_id is not found in job_similarities or if no similar jobs are available.
    """
    if job_id not in job_similarities:
        return f"Job ID {job_id} not found in embeddings."

    # Sort similar jobs by similarity score
    similar_jobs = sorted(job_similarities[job_id].items(), key=lambda x: x[1], reverse=True)
    if not similar_jobs: return f"No similar jobs found for Job ID {job_id}."

    # Display the top-N similar jobs
    print(f"\nTop {top_n} similar jobs for Job ID {job_id} ({job_titles.get(job_id, 'Unknown')})")
    for i, (job, similarity) in enumerate(similar_jobs[:top_n]):
        job_title = job_titles.get(job, "Unknown")
        print(f"  {i + 1} - Job ID {job} ({job_title}) \u27F6 Similarity: {similarity * 100:.2f}%")

    return similar_jobs[:top_n]

def sanity_check(job_embeddings, job_data, model_type, G=None, top_n=10):
    """
    Perform a quick evaluation of embeddings by checking similar jobs for predefined IDs.

    Args:
        job_embeddings (list): List of job embeddings.
        job_data (DataFrame): DataFrame containing job metadata.
        model_type (str): 'NLP' for text-based, 'GNN' for graph-based embeddings.
        G (networkx.Graph, optional): Graph object for GNN models. Defaults to None.
        top_n (int, optional): Number of similar jobs to retrieve. Defaults to 10.

    Returns:
        dict: Dictionary of job similarities.
    """
    job_ids = job_data["item_id"].tolist()
    job_titles = dict(zip(job_ids, job_data["pozisyon_adi"]))

    # Compute similarity matrix
    job_similarities = compute_similarity(job_embeddings, job_ids, model_type, G)

    # Select sample job IDs to check recommendations
    job_ids_to_check = [
        4022460, 4035187, 3442770, 113980, 4025763,
        4031546, 2882551, 3725681, 4028320, 4033707
    ]
    for job_id in job_ids_to_check:
        top_similar_jobs = get_similar_jobs(job_id, job_similarities,
                                            job_titles, top_n)

    return job_similarities

def test_embeddings(test_path, job_embeddings, job_data, model_type,
                    G=None, top_n=10, is_print=False):
    """
    Evaluate embedding quality using a labeled test set.

    Args:
        test_path (str): Path to the labeled test set.
        job_embeddings (list): List of job embeddings.
        job_data (DataFrame): DataFrame with job metadata.
        model_type (str): 'NLP' for text-based, 'GNN' for graph-based embeddings.
        G (networkx.Graph, optional): Graph object if using GNN embeddings. Defaults to None.
        top_n (int, optional): Number of similar jobs to retrieve for evaluation. Defaults to 10.
        is_print (bool, optional): If True, print top-N similar jobs for each source job.
        Defaults to False.

    Returns:
        dict: Dictionary with accuracy scores for each source job.
    """
    test_set = pd.read_csv(test_path)
    all_job_ids = pd.unique(test_set[['source_job_id', 'target_job_id']].values.ravel('K')).tolist()
    source_job_ids = test_set['source_job_id'].unique().tolist()

    # Prepare job embeddings matrix based on model type
    if model_type == "NLP":
        job_matrix = np.array(job_embeddings)
    elif model_type == "GNN":
        job_ids = get_graph_job_ids(G)
        job_index_mapping = {job_id: i for i, job_id in enumerate(job_ids)}

    # Filter jobs to ensure all test set jobs are in embeddings
    filtered_job_ids = []
    for job_id in all_job_ids:
        if job_id not in job_index_mapping:
            print(job_id)  # Log missing job IDs
            continue
        filtered_job_ids.append(job_id)

    # Build filtered embedding matrix
    job_matrix = np.array([job_embeddings[job_index_mapping[job_id]] for job_id in filtered_job_ids])

    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(job_matrix)
    job_titles = dict(zip(job_data["item_id"], job_data["pozisyon_adi"]))

    job_similarities = {}
    # Create similarity dictionary
    for i, job_id in enumerate(tqdm(filtered_job_ids, desc="Similarities", unit="job")):
        similarities = {
            filtered_job_ids[j]: cosine_sim_matrix[i, j]
            for j in range(len(filtered_job_ids)) if i != j
        }
        job_similarities[job_id] = similarities

    # Optional print of top-N similar jobs
    if is_print:
        for job_id in source_job_ids:
            _ = get_similar_jobs(job_id, job_similarities, job_titles, top_n)

    # Calculate accuracy
    accuracies = calculate_accuracy(job_similarities, test_set, source_job_ids)

    return accuracies

def calculate_accuracy(job_similarities, test_set, source_job_ids):
    """
    Calculate top-10 accuracy based on labeled test pairs.

    Args:
        job_similarities (dict): Dictionary with job similarity scores.
        test_set (DataFrame): DataFrame containing labeled job pairs.
        source_job_ids (list): List of source job IDs to evaluate.

    Returns:
        dict: Dictionary of accuracy percentages for each source job.
    """

    # Filter positive (relevant) test pairs
    positive_tests = test_set[test_set['label'] == 1]
    accuracies = {}

    # Iterate over each source job to evaluate its recommendation accuracy
    print("\n")
    for source_id in source_job_ids:
        # Get all positive target jobs for the current source job
        source_positives = positive_tests[positive_tests['source_job_id'] == source_id]
        total_positives = len(source_positives)
        correct_count = 0

        # Check if target jobs appear in top-10 similar jobs
        for _, row in source_positives.iterrows():
            target = row['target_job_id']

            # Retrieve top-10 most similar jobs for current source job
            top_10_similar = sorted(job_similarities[source_id].items(), key=lambda x: x[1], reverse=True)[:10]
            top_10_ids = [job_id for job_id, _ in top_10_similar]

            # Increment correct count if target is in top-10
            if target in top_10_ids: correct_count += 1

        # Calculate accuracy as percentage (handle cases with no positives)
        accuracy = correct_count / total_positives if total_positives > 0 else None
        accuracies[source_id] = round(accuracy * 100, 0)
        print(f"Job ID {source_id}: Accuracy = {accuracy * 100:.0f}%" if accuracy is not None else f"Job ID {source_id}: No positives to evaluate")

    return accuracies

def save_embeddings(job_ids, job_embeddings, output_path):
    """
    Save job embeddings to a CSV file.

    Args:
        job_ids (list or array-like): List of job IDs corresponding to the embeddings.
        job_embeddings (numpy.ndarray): Array of job embeddings where each row represents a job vector.
        output_path (str): Path to the output CSV file where embeddings will be saved.

    Returns:
        None
    """
    # Create DataFrame with job IDs as index
    df = pd.DataFrame(job_embeddings, index=job_ids)
    df.index.name = "item_id"

    # Sort by job ID for consistency
    df = df.sort_index()

    # Save embeddings to CSV
    df.to_csv(output_path, encoding='utf-8')
    print(f"Embeddings saved.")

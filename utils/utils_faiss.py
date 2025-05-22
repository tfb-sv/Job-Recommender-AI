import time
import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def load_embeddings(emb_path):
    """
    Load embeddings from CSV file.

    Args:
        emb_path (str): Path to the embeddings file.

    Returns:
        tuple: (numpy.ndarray of embeddings, numpy.ndarray of item IDs)
    """
    # Load embeddings from CSV
    emb_df = pd.read_csv(emb_path)

    # Extract item IDs
    emb_item_ids = emb_df['item_id'].values

    # Extract embeddings
    emb = emb_df.drop(columns=['item_id']).values

    return emb, emb_item_ids

def concat_embeddings(graph_path, text_path, combined_emb_path,
                      job_ids_path, normalize_embeddings=False):
    """
    Concatenate graph-based and text-based embeddings, align by job IDs, and optionally normalize.

    Args:
        graph_path (str): Path to graph-based embeddings.
        text_path (str): Path to text-based embeddings.
        combined_emb_path (str): Output path for combined embeddings.
        job_ids_path (str): Output path for aligned job IDs.
        normalize_embeddings (bool): Whether to normalize embeddings. Default is False.

    Returns:
        tuple: (numpy.ndarray of combined embeddings, numpy.ndarray of aligned job IDs)
    """
    # Load graph-based and text-based embeddings
    graph_embeddings, graph_item_ids = load_embeddings(graph_path)
    text_embeddings, text_item_ids = load_embeddings(text_path)

    # Get embedding dimensions
    graph_dim = graph_embeddings.shape[1]  # Typically 128
    text_dim = text_embeddings.shape[1]  # Typically 512

    # Merge and align job IDs
    all_item_ids = np.union1d(graph_item_ids, text_item_ids)
    print(f"Total unique job IDs after alignment: {len(all_item_ids)}")

    # Create mapping from job IDs
    graph_id_to_index = {item_id: idx for idx, item_id in enumerate(graph_item_ids)}
    text_id_to_index = {item_id: idx for idx, item_id in enumerate(text_item_ids)}

    # Create zero vectors for jobs missing in either graph or text embeddings
    zero_graph = np.zeros((graph_dim,))
    zero_text = np.zeros((text_dim,))

    # Align embeddings for all job IDs, filling missing ones with zero vectors
    aligned_graph_embeddings = []
    aligned_text_embeddings = []
    for item_id in all_item_ids:
        graph_emb = graph_embeddings[graph_id_to_index[item_id]] if item_id in graph_id_to_index else zero_graph
        text_emb = text_embeddings[text_id_to_index[item_id]] if item_id in text_id_to_index else zero_text
        aligned_graph_embeddings.append(graph_emb)
        aligned_text_embeddings.append(text_emb)

    # Convert lists to numpy arrays
    aligned_graph_embeddings = np.array(aligned_graph_embeddings)
    aligned_text_embeddings = np.array(aligned_text_embeddings)
    
    # Optionally normalize each embedding set to unit vectors
    if normalize_embeddings:
        aligned_graph_embeddings = normalize(aligned_graph_embeddings)
        aligned_text_embeddings = normalize(aligned_text_embeddings)

    # Concatenate graph and text embeddings horizontally (side by side)
    combined_embeddings = np.hstack((aligned_graph_embeddings, aligned_text_embeddings))

    # Optionally normalize the combined embedding vectors
    if normalize_embeddings:
        combined_embeddings = normalize(combined_embeddings)

    print("Combined embeddings shape:", combined_embeddings.shape)

    # Save combined results
    save_combined_embeddings(combined_embeddings, all_item_ids, combined_emb_path, job_ids_path)

    return combined_embeddings, all_item_ids

def create_faiss_index(combined_embeddings, index_path, metric='ip'):
    """
    Create and save a FAISS index from combined embeddings.

    Args:
        combined_embeddings (numpy.ndarray): Combined embeddings matrix.
        index_path (str): Path to save FAISS index.
        metric (str): Similarity metric ('ip' for inner product/cosine, 'l2' for Euclidean).
        Default is 'ip'.

    Returns:
        faiss.Index: Created FAISS index.
    """
    embedding_dim = combined_embeddings.shape[1]

    # Select metric type
    if metric == 'l2':  # euclidean distance
        faiss_index = faiss.IndexFlatL2(embedding_dim)
    elif metric == 'ip':  # cosine similarity
        faiss_index = faiss.IndexFlatIP(embedding_dim)

    # Add embeddings and save index
    faiss_index.add(combined_embeddings)
    print(f"FAISS index created with {faiss_index.ntotal} job embeddings.")
    faiss.write_index(faiss_index, index_path)
    print(f"FAISS index saved.")

    return faiss_index

def load_faiss_index(index_path):
    """
    Load a FAISS index from disk.

    Args:
        index_path (str): Path to FAISS index file.

    Returns:
        faiss.Index: Loaded FAISS index.
    """
    faiss_index = faiss.read_index(index_path)
    print(f"FAISS index loaded.")

    return faiss_index

def save_combined_embeddings(embeddings, item_ids, emb_path, ids_path):
    """
    Save combined embeddings and their item IDs.

    Args:
        embeddings (numpy.ndarray): Combined embeddings matrix.
        item_ids (numpy.ndarray): Corresponding job IDs.
        emb_path (str): Path to save embeddings.
        ids_path (str): Path to save item IDs.
    """
    np.save(emb_path, embeddings)
    np.save(ids_path, item_ids)
    print(f"Combined embeddings saved.")

def load_combined_embeddings(emb_path, item_ids_path):
    """
    Load combined embeddings and their corresponding item IDs.

    Args:
        emb_path (str): Path to combined embeddings file.
        item_ids_path (str): Path to item IDs file.

    Returns:
        tuple: (embeddings array, item IDs array)
    """
    embeddings = np.load(emb_path)
    item_ids = np.load(item_ids_path)
    print(f"Combined embeddings loaded. Shape: {embeddings.shape}")

    return embeddings, item_ids

def recommend_jobs(job_ids, combined_embeddings, faiss_index,
                   job_data, job_id, top_n, is_similarities=True):
    """
    Recommend top-N similar jobs for a given job_id using FAISS index.

    Args:
        job_ids (numpy.ndarray): Array of job IDs aligned with embeddings.
        combined_embeddings (numpy.ndarray): Combined embeddings array.
        faiss_index (faiss.Index): Pre-built FAISS index.
        job_data (DataFrame): DataFrame with job metadata.
        job_id (int): Target job ID for recommendations.
        top_n (int): Number of jobs to recommend.
        is_similarities (bool): Whether to print similarity scores. Default is True.

    Returns:
        list: List of recommended job IDs.
    """
    start_time = time.time()

    # Check if the requested job ID exists
    if job_id not in job_ids:
        return f"Job ID {job_id} not found."

    # Find index and retrieve embedding
    job_index = np.where(job_ids == job_id)[0][0]
    query_vector = np.array([combined_embeddings[job_index]])

    # Search for similar jobs (+1 to exclude itself later)
    distances, indices = faiss_index.search(query_vector, top_n + 1)

    # Normalize distances to percentages (if desired)
    max_dist = distances.max()
    min_dist = 0
    distances = [(d - min_dist) * 100 / (max_dist - min_dist) for d in distances]

    # Prepare results excluding the queried job itself
    similar_jobs = [job_ids[idx] for idx in indices[0] if job_ids[idx] != job_id]
    similar_jobs_with_distances = [
        (job_ids[idx], distances[0][i])
        for i, idx in enumerate(indices[0])
        if job_ids[idx] != job_id
    ][:top_n]

    # Create a mapping from job IDs to job titles
    job_titles = dict(zip(job_data["item_id"], job_data["pozisyon_adi"]))

    # Display the top-N similar jobs
    print(f"\nTop {top_n} similar jobs for Job ID {job_id} ({job_titles.get(job_id, 'Unknown')}):")
    for i, (similar_id, distance) in enumerate(similar_jobs_with_distances):
        job_title = job_titles.get(similar_id, 'Unknown')
        if is_similarities:
            print(f"  {i + 1} - Job ID {similar_id} ({job_title}) \u27F6 Similarity: {distance:.2f}%")
        else:
            print(f"  {i + 1} - Job ID {similar_id} ({job_title})")

    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nJob recommendation computation time: {execution_time:.2f} seconds")

    return similar_jobs

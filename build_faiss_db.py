from utils.utils_faiss import *
from utils.utils_common import load_job_data

"""
Pipeline to combine GNN and NLP embeddings and perform job recommendation using FAISS.

Steps:
    1. Combine embeddings:
        - Merge graph-based (GNN) and text-based (SBERT) embeddings.

    2. Create or load FAISS index:
        - Vector database for fast similarity search.

    3. Recommend similar jobs:
        - Input: Job ID
        - Output: Top-N similar jobs
"""

def main(job_id, top_n=10, load_saved_files=False):
    """
    Main pipeline to combine embeddings, create/load FAISS index, and recommend similar jobs.

    Args:
        job_id (int): ID of the job to get recommendations for.
        top_n (int, optional): Number of similar jobs to recommend. Defaults to 10.
        load_saved_files (bool, optional): 
            If True, load saved combined embeddings and FAISS index. 
            If False, create and save them. Defaults to False.

    Returns:
        None
    """
    # File paths
    job_data_path = "data/item_information.csv"
    graph_emb_path = "results/job_embeddings_GNN.csv"
    text_emb_path = "results/job_embeddings_NLP.csv"
    index_path = "results/faiss_job_embeddings.index"
    combined_emb_path = "results/job_embeddings_combined.npy"
    job_ids_path = "results/job_ids_combined.npy"

    # Load job data
    job_data = load_job_data(job_data_path)
    
    # Create or load combined embeddings and FAISS index
    if not load_saved_files:
        combined_embeddings, job_ids = concat_embeddings(graph_emb_path, text_emb_path,
                                                         combined_emb_path, job_ids_path)
        faiss_index = create_faiss_index(combined_embeddings, index_path)
    else:
        combined_embeddings, job_ids = load_combined_embeddings(combined_emb_path, job_ids_path)
        faiss_index = load_faiss_index(index_path)

    # Recommend similar jobs
    top_similar_jobs = recommend_jobs(job_ids, combined_embeddings, faiss_index,  # Takes ~20 milliseconds
                                      job_data, job_id, top_n)

if __name__ == "__main__":
    # Example input parameteres for job recommendation
    job_id = 4031546
    top_n = 20

    # Set to True to load saved combined embeddings and FAISS Index instead of creating a new one
    load_saved_files = False
    
    # Run main pipeline
    main(job_id, top_n, load_saved_files)

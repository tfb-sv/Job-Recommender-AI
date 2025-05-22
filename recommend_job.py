import argparse
from utils.utils_faiss import *
from utils.utils_common import load_job_data

"""
Pre-built Independent Job Recommendation System.

Features:
    - Uses saved FAISS index and combined embeddings.
    - Input: Job ID (required), Top-N similar jobs (optional, default=10).
    - Output: Top-N most similar job recommendations.
"""

def main(args):
    """
    Main function to load resources and generate job recommendations.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing job_id and top_n.

    Returns:
        None
    """
    job_id = args.job_id
    top_n = args.top_n

    # File paths
    job_data_path = "data/item_information.csv"
    index_path = "results/faiss_job_embeddings.index"
    combined_emb_path = "results/job_embeddings_combined.npy"
    job_ids_path = "results/job_ids_combined.npy"

    # Load job data, combined embeddings, and FAISS index
    job_data = load_job_data(job_data_path)
    combined_embeddings, job_ids = load_combined_embeddings(combined_emb_path, job_ids_path)
    faiss_index = load_faiss_index(index_path)

    # Recommend similar jobs
    top_similar_jobs = recommend_jobs(job_ids, combined_embeddings, faiss_index,
                                      job_data, job_id, top_n, is_similarities=False)

def load_args():
    """
    Load and parse command-line arguments for job recommendation.

    Returns:
        argparse.Namespace: Parsed arguments with job_id and top_n.
    """
    parser = argparse.ArgumentParser(
        description="Job Recommendation System: Input job_id to get similar jobs."
    )
    parser.add_argument(
        "job_id", type=int, help="Job ID for which recommendations are needed (required)."
    )
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="Number of similar jobs to recommend (default=10)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = load_args()

    # Run main pipeline
    main(args)

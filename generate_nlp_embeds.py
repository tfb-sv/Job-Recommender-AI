from utils.utils_NLP import *
from utils.utils_common import *

"""
Pipeline to generate job embeddings using NLP (Sentence-BERT).

Steps:
    1. Preprocess job descriptions:
        - Clean text (stopwords, punctuation, etc.).

    2. Generate embeddings:
        - Method: SBERT (pre-trained model)

    3. Save job embeddings for further use (recommendation, search).
"""

def main(load_saved_files=False):
    """
    Main pipeline to preprocess job title and descriptions, generate SBERT embeddings, and save them.

    Args:
        load_saved_files (bool, optional): 
            If True, load the saved stopwords. 
            If False, create and save it. Defaults to False.

    Returns:
        None
    """
    model_type = "NLP"
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"

    # File paths
    stopwords_path = "helpers/merged_stopwords.txt"
    job_data_path = "data/item_information.csv"
    output_path = f"results/job_embeddings_{model_type}.csv"

    # Create or load stopwords
    if not load_saved_files:
        stopwords = prepare_stopwords("helpers/turkish-stop-words.txt",
                                      "helpers/turkish_stopwords.json",
                                      stopwords_path)
    else: stopwords = load_merged_stopwords(stopwords_path)

    # Load and preprocess job data
    job_data = load_job_data(job_data_path)
    job_data = preprocess_job_descriptions(job_data, stopwords)

    # Generate embeddings using pre-trained model (SBERT)
    job_embeddings = generate_embeddings(job_data, model_name)  # Takes ~3 minutes using GPU without downloading the model

    # Optional: Check similar jobs
    # job_similarities = sanity_check(job_embeddings, job_data, model_type)  # Takes ~6 minutes

    # Save embeddings
    save_embeddings(job_data["item_id"].tolist(), job_embeddings, output_path)

if __name__ == "__main__":
    # Set to True to load saved stopwords instead of creating a new one
    load_saved_files = False

    # Run main pipeline
    main(load_saved_files)

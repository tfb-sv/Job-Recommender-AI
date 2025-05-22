import re
import json
import time
import nltk
import torch
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as nsw
from models.models_NLP import SBERT

# Set random seed for reproducibility
SEED_NO = 0
torch.manual_seed(SEED_NO)

def download_nltk_resources():
    """
    Download necessary NLTK resources (stopwords for Turkish and English).

    Returns:
        tuple: A set of Turkish stopwords and a set of English stopwords.
    """
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    # Load stopwords for both Turkish and English
    tr_stopwords = set(nsw.words('turkish'))
    en_stopwords = set(nsw.words('english'))

    return tr_stopwords, en_stopwords

def prepare_stopwords(txt_file, json_file, output_path):
    """
    Merge stopwords from NLTK, a custom TXT file, and a custom JSON file.

    Args:
        txt_file (str): Path to the TXT file containing stopwords.
        json_file (str): Path to the JSON file containing stopwords.
        output_path (str): File path to save the merged stopwords.

    Returns:
        list: Sorted list of merged stopwords.
    """
    # Download and load NLTK stopwords
    nltk_tr_stopwords, nltk_en_stopwords = download_nltk_resources()

    # Load stopwords from TXT file
    with open(txt_file, "r", encoding="utf-8") as f:
        txt_stopwords = set([line.strip().lower() for line in f])

    # Load stopwords from JSON file and filter them with regex
    with open(json_file, "r", encoding="utf-8") as f:
        json_raw = json.load(f)
        regex_pattern = r"^[a-z\u00E7\u011F\u0131\u00F6\u015F\u00FC]+$"
        json_stopwords = set(
            word for word in json_raw['stopwords']
            if re.match(regex_pattern, word)
        )

    # Merge all stopword lists and sort
    merged_stopwords = nltk_tr_stopwords | txt_stopwords | json_stopwords | nltk_en_stopwords
    merged_stopwords = sorted(merged_stopwords)

    # Save merged stopwords to output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_stopwords))
    print(f"Total {len(merged_stopwords)} stopwords merged and saved!")

    return merged_stopwords

def load_merged_stopwords(file_path):
    """
    Load merged stopwords from a file.

    Args:
        file_path (str): Path to the stopwords file.

    Returns:
        set: Set of stopwords.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())

    return stopwords

def clean_text(text, stopwords):
    """
    Clean job description text by removing stopwords, punctuation, and special characters.

    Args:
        text (str): Raw text data (e.g., job description).
        stopwords (set): Set of stopwords to remove.

    Returns:
        str: Cleaned text.
    """
    # Return empty string if text is NaN
    if pd.isnull(text): return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters and keep only letters (Turkish & English)
    regex_pattern = r"[^a-zA-Z\u00E7\u011F\u0131\u00F6\u015F\u00FC\u00C7\u011E\u0130\u00D6\u015E\u00DC ]"
    text = re.sub(regex_pattern, " ", text)

    # Lowercase and remove extra spaces
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and remove stopwords and single-character tokens
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [word for word in tokens if len(word) > 1]

    return " ".join(tokens)

def preprocess_job_descriptions(job_data, stopwords):
    """
    Clean and preprocess job titles and descriptions in the dataset.

    Args:
        job_data (DataFrame): Job dataset containing 'pozisyon_adi' and 'item_id_aciklama'.
        stopwords (set): Set of stopwords to remove during cleaning.

    Returns:
        DataFrame: DataFrame with added cleaned columns.
    """
    # Apply cleaning to job titles and descriptions
    job_data["cleaned_title"] = job_data["pozisyon_adi"].apply(lambda x: clean_text(x, stopwords))
    job_data["cleaned_description"] = job_data["item_id_aciklama"].apply(lambda x: clean_text(x, stopwords))

    return job_data

def generate_embeddings(job_data, model_name, title_weight=0.5, desc_weight=0.5):
    """
    Generate weighted combined embeddings for job titles and descriptions using SBERT.

    Args:
        job_data (DataFrame): DataFrame containing 'cleaned_titles' and 'cleaned_descriptions'.
        model_name (str): Pre-trained SBERT model name to use.
        title_weight (float): Weight for title embeddings. Default is 0.5.
        desc_weight (float): Weight for description embeddings. Default is 0.5.

    Returns:
        list: List of combined embeddings as Python lists.
    """
    start_time = time.time()

    # Check if GPU is available
    if torch.cuda.is_available(): print("GPU is ready.")

    # Prepare data lists for embedding
    titles = job_data["cleaned_title"].fillna("").tolist()
    descriptions = job_data["cleaned_description"].fillna("").tolist()

    # Generate embeddings for job titles
    print("Generating title embeddings...")
    title_embeddings = SBERT(titles, model_name)

    # Generate embeddings for job descriptions
    print("Generating description embeddings...")
    desc_embeddings = SBERT(descriptions, model_name)

    # Combine embeddings using specified weights
    combined_embeddings = (title_weight * title_embeddings) + (desc_weight * desc_embeddings)

    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Embeddings generation time: {execution_time:.2f} seconds")
    print(f"Embeddings generated!")

    return combined_embeddings.tolist()

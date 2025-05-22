import torch
from sentence_transformers import SentenceTransformer

def SBERT(texts, model_name):
    """
    Generate sentence embeddings using a pre-trained Sentence-BERT (SBERT) model.

    Args:
        texts (list): List of input texts to be embedded.
        model_name (str): Pre-trained SBERT model name.

    Returns:
        numpy.ndarray: Array of sentence embeddings.
    """
    # Select device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained SBERT model
    model = SentenceTransformer(model_name, device=device)

    # Set model to evaluation mode (disables dropout and gradients)
    model.eval()

    # Encode texts into embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,  # Output as NumPy array
        show_progress_bar=True  # Show progress bar during encoding
    )

    return embeddings

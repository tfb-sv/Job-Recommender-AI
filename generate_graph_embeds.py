import itertools
import numpy as np
from utils.utils_GNN import *
from utils.utils_common import *

"""
Main pipeline to generate and evaluate job embeddings using Graph Neural Networks (GCN).

Steps:
    1. Graph to embeddings:
        - Generate vector representations (embeddings) for each job listing.
        - Method: Graph Convolutional Network (GCN)

    2. Embedding evaluation:
        - Evaluate the quality of embeddings.
        - Method: Top-10 accuracy based on cosine similarity

    3. Save embeddings:
        - Save the generated embeddings for further use (recommendation, search).
"""

def part2_process(in_channels, dropout, load_saved_files):
    """
    Process pipeline for GNN embedding generation and evaluation.

    Args:
        in_channels (int): Dimension of input node features.
        dropout (float): Dropout rate used in GNN.
        load_saved_files (bool): 
            If True, load the saved model. 
            If False, train and save it.

    Returns:
        float: Mean accuracy of job recommendations.
    """
    model_type = "GNN"
    out_channels = in_channels  # Output dimension equals input dimension
    hidden_channels = in_channels * 2  # Hidden layer dimension
    epoch_cnt = 1000  # Number of training epochs
    lr = 1e-3  # Learning rate

    # File paths
    graph_path = "results/graph.pkl"
    job_data_path = "data/item_information.csv"
    model_path="results/GNN_model.pth"
    test_path = "data/test.csv"
    output_path = f"results/job_embeddings_{model_type}.csv"

    # Load graph and job data
    G = load_graph(graph_path)
    job_data = load_job_data(job_data_path)

    # Convert graph to PyTorch Geometric (PyG) format
    pyg_data = convert_nx_to_pyg(G, in_channels)

    # Train or load model
    if not load_saved_files:
        trained_model = train_model(  # Takes ~40 seconds using GPU with best hyperparameters
            pyg_data, in_channels, hidden_channels, out_channels,
            epoch_cnt, lr, dropout, model_path
        )
    else:
        trained_model = load_model(
            model_path, in_channels, hidden_channels, out_channels, dropout
        )

    # Generate embeddings using trained model (GNN)
    job_embeddings = extract_embeddings(trained_model, pyg_data)

    # Optional: Check similar jobs
    # job_similarities = sanity_check(job_embeddings, job_data, model_type, G)  # Takes ~6 minutes

    # Evaluate embeddings using test set
    accuracies = test_embeddings(test_path, job_embeddings, job_data, model_type, G)
    mean_accuracy = np.mean(list(accuracies.values()))

    # Save embeddings
    job_ids = get_graph_job_ids(G)
    save_embeddings(job_ids, job_embeddings, output_path)

    # Reset GPU and clear memory
    reset_model()

    return mean_accuracy

def main(in_channels_list, dropout_list, load_saved_files=False):
    """
    Run GNN embedding generation and evaluation for multiple hyperparameter combinations.

    Args:
        in_channels_list (list): List of input feature dimensions to try.
        dropout_list (list): List of dropout rates to try.
        load_saved_files (bool, optional): 
            If True, load the saved model. 
            If False, train and save it. Defaults to False.

    Returns:
        None
    """
    # Create all combinations of in_channels and dropout rates
    param_combinations = list(itertools.product(in_channels_list, dropout_list))

    # Run the pipeline for each combination
    results = {}
    for in_channels, dropout in param_combinations:
        mean_accuracy = part2_process(in_channels, dropout, load_saved_files)
        results[(in_channels, dropout)] = mean_accuracy

    # Display final results
    print("\nFinal Results:")
    for params, acc in results.items():
        print(f"in_channels={params[0]}, dropout={params[1]} -> Mean Accuracy: {acc:.1f}%")

if __name__ == "__main__":
    # Hyperparameter options for tuning
    # in_channels_list = [64, 128, 256, 512]  # Feature dimension
    # dropout_list = [0.3, 0.5, 0.7]  # Dropout rate

    # Best hyperparameters (single run)
    in_channels_list = [128]
    dropout_list = [0.5]

    # Set to True to load saved model instead of training a new one
    load_saved_files = False

    # Hyperparameter search check
    if load_saved_files:
        assert_msg = "Hyperparameter lists must have length 1 when using a saved model!"
        assert len(in_channels_list) == 1 and len(dropout_list) == 1, assert_msg

    # Run main pipeline
    main(in_channels_list, dropout_list, load_saved_files)

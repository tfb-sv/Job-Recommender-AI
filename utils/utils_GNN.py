import gc
import time
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from models.models_GNN import GCN

# Set random seed for reproducibility
SEED_NO = 0
torch.manual_seed(SEED_NO)

def convert_nx_to_pyg(G, in_channels=128):
    """
    Convert a NetworkX graph to PyTorch Geometric (PyG) Data format.

    Args:
        G (networkx.Graph): Input NetworkX graph.
        in_channels (int): Dimension of initial node features. Default is 128.

    Returns:
        torch_geometric.data.Data: PyG graph data object containing nodes, edges, and edge attributes.
    """
    # Map each node to a unique index
    node_mapping = {node: i for i, node in enumerate(G.nodes)}

    # Convert edge list to tensor format
    edge_index = torch.tensor(
        [[node_mapping[src], node_mapping[dst]] for src, dst in G.edges],
        dtype=torch.long
    ).t().contiguous()  # Transpose to match PyG format (2, num_edges)

    # Extract edge weights as tensor
    edge_weight = torch.tensor(
        [G[u][v]['weight'] for u, v in G.edges],
        dtype=torch.float
    )

    # Initialize random node features
    num_nodes = len(G.nodes)
    x = torch.rand((num_nodes, in_channels))

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    print(f"Graph converted to PyTorch Geometric format! Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")

    return data

def train_model(data, in_channels, hidden_channels,
                out_channels, epochs, lr, dropout, model_path):
    """
    Train a GCN model on the provided graph data.

    Args:
        data (torch_geometric.data.Data): Graph data object.
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden layer dimension.
        out_channels (int): Output embedding dimension.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        dropout (float): Dropout rate.
        model_path (str): Path to save the trained model.

    Returns:
        torch.nn.Module: Trained GCN model.
    """
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print("GPU is ready.")

    # Move data to device
    data = data.to(device)

    # Initialize GCN model
    model = GCN(in_channels, hidden_channels,
                out_channels, dropout).to(device)
    print(model)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop
    print(f"Training on {device} for {epochs} epochs...")
    model.train()
    start_time = time.time()
    with tqdm(range(epochs), desc="Epochs") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            out = model(data)  # Forward pass
            loss = criterion(out, data.x)  # Reconstruction loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimization step
            pbar.set_postfix_str(f"MSE Loss: {loss.item():.4f}")
    print("Model training complete!")

    # Save trained model
    save_model(model, model_path)

    # Calculate and display execution time
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    print(f"Execution time for training: {execution_time:.2f} seconds")

    return model

def save_model(model, model_path):
    """
    Save a trained PyTorch model.

    Args:
        model (torch.nn.Module): Trained model to save.
        model_path (str): Path to save the model.
    
    Returns:
        None
    """
    # Save model weights
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

def load_model(model_path, in_channels, hidden_channels, out_channels, dropout):
    """
    Load a trained GCN model from disk.

    Args:
        model_path (str): Path to the saved model.
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden layer dimension.
        out_channels (int): Output embedding dimension.
        dropout (float): Dropout rate used in the model.

    Returns:
        torch.nn.Module: Loaded GCN model in evaluation mode.
    """
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize GCN model
    model = GCN(in_channels, hidden_channels,
                out_channels, dropout).to(device)

    # Load saved model weights
    model.load_state_dict(torch.load(model_path))

    # Set model to evaluation mode (disables dropout and gradients)
    model.eval()
    print(f"Loaded model.")

    return model

def extract_embeddings(model, pyg_data):
    """
    Extract node embeddings from the trained GCN model.

    Args:
        model (torch.nn.Module): Trained GCN model.
        pyg_data (torch_geometric.data.Data): Graph data object.

    Returns:
        list: List of node embeddings as Python lists.
    """
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move graph data and model to device
    pyg_data = pyg_data.to(device)
    model.to(device)

    # Set model to evaluation mode (disables dropout and gradients)
    model.eval()

    # Generate embeddings without gradient computation
    with torch.no_grad(): embeddings = model(pyg_data)

    # Convert to CPU and convert to list format for later use
    embeddings = embeddings.cpu().numpy().tolist()
    print(f"Embeddings generated!")

    return embeddings

def reset_model():
    """
    Clear GPU memory and reset cache. Useful to prevent memory overflow in multiple training sessions.

    Returns:
        None
    """
    gc.collect()
    torch.cuda.empty_cache()

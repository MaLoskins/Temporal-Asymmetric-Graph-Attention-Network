"""
TAGAN Usage Example

This script demonstrates how to use the Temporal Asymmetric Geometric Attention Network (TAGAN)
for temporal graph analysis. It includes data loading, model configuration, training,
evaluation, and visualization of attention patterns.
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from src.tagan.model import TAGAN
from src.tagan.utils.config import TAGANConfig
from src.tagan.training.trainer import TAGANTrainer
from src.tagan.data.data_loader import TemporalGraphDataLoader, TemporalGraphDataset
from src.tagan.visualization.attention_vis import plot_attention_patterns, plot_graph_with_attention
from src.tagan.utils.metrics import calculate_metrics


def create_synthetic_data(num_samples=100, seq_length=5, num_nodes=10, feature_dim=16):
    """Create synthetic temporal graph data for demonstration purposes."""
    
    # Create dataset container
    dataset = []
    
    for i in range(num_samples):
        # Create a sequence of graphs
        graph_sequence = []
        
        for t in range(seq_length):
            # Create random node features
            num_active_nodes = np.random.randint(5, num_nodes + 1)
            node_features = torch.randn(num_active_nodes, feature_dim)
            
            # Create random edge indices (connections between nodes)
            edge_index = torch.randint(0, num_active_nodes, (2, num_active_nodes * 2))
            
            # Create random edge features
            edge_attr = torch.randn(edge_index.size(1), 8)
            
            # Create node IDs (some nodes may disappear and reappear)
            node_ids = np.random.choice(num_nodes, num_active_nodes, replace=False).tolist()
            # Add to sequence - use a dictionary format to match what preprocessing.py expects
            # Use tuple format which is what the model expects
            graph_sequence.append((node_features, edge_index, edge_attr, node_ids))
        
        # Create random labels (binary classification)
        label = torch.randint(0, 2, (1,)).float()
        
        # Add to dataset
        dataset.append((graph_sequence, label))
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    return train_data, val_data, test_data


def main():
    """Main function demonstrating TAGAN usage."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)
    
    # Step 1: Create synthetic data
    print("\n===== DEBUGGING: Starting synthetic data creation =====")
    train_data, val_data, test_data = create_synthetic_data()
    
    # Debug info about created data
    print(f"DEBUGGING: Data sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"DEBUGGING: First train sample structure: {type(train_data[0])}")
    print(f"DEBUGGING: First train sample graph sequence length: {len(train_data[0][0])}")
    print(f"DEBUGGING: First train sample label: {train_data[0][1]}")
    print("===== DEBUGGING: Finished synthetic data creation =====\n")
    
    # Step 2: Create data loaders
    batch_size = 4  # Reduced batch size
    
    # Extract sequences and labels from data tuples
    train_sequences = [seq for seq, _ in train_data]
    train_labels = [label.item() for _, label in train_data]
    
    val_sequences = [seq for seq, _ in val_data]
    val_labels = [label.item() for _, label in val_data]
    
    test_sequences = [seq for seq, _ in test_data]
    test_labels = [label.item() for _, label in test_data]
    
    # Wrap raw data in TemporalGraphDataset instances
    train_dataset = TemporalGraphDataset(data=train_sequences, labels=train_labels)
    val_dataset = TemporalGraphDataset(data=val_sequences, labels=val_labels)
    test_dataset = TemporalGraphDataset(data=test_sequences, labels=test_labels)
    
    # Create data loaders from datasets
    print("\n===== DEBUGGING: Creating data loaders =====")
    train_loader = TemporalGraphDataLoader(train_dataset, batch_size=batch_size)
    val_loader = TemporalGraphDataLoader(val_dataset, batch_size=batch_size)
    test_loader = TemporalGraphDataLoader(test_dataset, batch_size=batch_size)
    
    # Debug info about loaders
    print(f"DEBUGGING: TemporalGraphDataLoader batch size: {batch_size}")
    print(f"DEBUGGING: Train loader length: {len(train_loader)}")
    print(f"DEBUGGING: Val loader length: {len(val_loader)}")
    print(f"DEBUGGING: Test loader length: {len(test_loader)}")
    # Check first batch
    first_batch = next(iter(train_loader))
    print(f"DEBUGGING: First batch type: {type(first_batch)}")
    if isinstance(first_batch, tuple) and len(first_batch) == 2:
        sequences, labels = first_batch
        print(f"DEBUGGING: First batch sequences type: {type(sequences)}")
        print(f"DEBUGGING: First batch labels shape: {labels.shape}")
    print("===== DEBUGGING: Finished creating data loaders =====\n")
    
    # Step 3: Configure model
    print("Configuring model...")
    config = TAGANConfig(
        node_feature_dim=16,      # Dimension of node features
        edge_feature_dim=8,       # Dimension of edge features
        hidden_dim=64,            # Hidden dimension size
        output_dim=1,             # Binary classification
        num_heads=4,              # Number of attention heads
        num_layers=2,             # Number of GNN layers
        dropout=0.1,              # Dropout probability
        learning_rate=0.001,      # Learning rate
        weight_decay=1e-5,        # Weight decay for regularization
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Device to use
        loss_type='bce',          # Loss function type
        edge_importance=True,     # Whether to use edge features
        use_layer_norm=True,      # Whether to use layer normalization
        memory_decay_factor=0.9,  # Decay factor for memory bank
        gru_bias=True,            # Whether to use bias in GRU
        temporal_attention_dim=64, # Dimension for temporal attention
        leaky_relu_slope=0.2,     # Slope for LeakyReLU activation
        use_edge_features=True,   # Use edge features (since edge_feature_dim=8)
        
        # Temporal propagation options
        time_aware=True,          # Use time-aware processing
        bidirectional=False,      # Unidirectional temporal processing
        use_skip_connection=True, # Use skip connections
        use_gating=True,          # Use gating mechanisms
        temporal_window_size=3,   # Window size for skip connections
        aggregation_method='mean',# Method for aggregating skip connections
        use_residual=True         # Use residual connections
    )
    
    # Step 4: Create model
    print(f"Creating model (running on {config.device})...")
    model = TAGAN(config)
    model.to(config.device)
    
    # Step 5: Create trainer
    print("Setting up trainer...")
    trainer = TAGANTrainer(
        model=model,
        config=config,
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    )
    
    # Step 6: Train model
    print("\n===== DEBUGGING: Starting model training =====")
    # Override the number of epochs to a smaller value for demo purposes
    num_epochs = 5  # Reduced number of epochs
    print(f"DEBUGGING: Training for {num_epochs} epochs")
    print(f"DEBUGGING: Train loader length: {len(train_loader)}")
    print(f"DEBUGGING: Val loader length: {len(val_loader)}")
    
    # Get a sample batch to check structure
    for i, sample_batch in enumerate(train_loader):
        if i == 0:
            if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                sequences, labels = sample_batch
                print(f"DEBUGGING: Sample batch - sequences type: {type(sequences)}")
                print(f"DEBUGGING: Sample batch - labels shape: {labels.shape}")
            else:
                print(f"DEBUGGING: Unexpected batch structure: {type(sample_batch)}")
            break
            
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        validate_every=1,
        save_best=True
    )
    print("===== DEBUGGING: Finished model training =====\n")
    
    # Step 7: Test model
    print("\n===== DEBUGGING: Starting model testing =====")
    test_results = trainer.test(test_loader, model_path='./checkpoints/best_model.pt')
    
    print(f"Test results: Loss = {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {test_results['metrics']['precision']:.4f}")
    print(f"Recall: {test_results['metrics']['recall']:.4f}")
    print(f"F1 Score: {test_results['metrics']['f1']:.4f}")
    print(f"All metrics: {test_results['metrics']}")
    print("===== DEBUGGING: Finished model testing =====\n")
    
    # Step 8: Visualize attention patterns
    print("Visualizing attention patterns...")
    # Get a sample from test data
    sample_sequence, sample_label = test_data[0]
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('./visualizations', exist_ok=True)
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        attention_output = model.infer_with_attention(sample_sequence)
        
        # Print the attention weight structures for debugging
        print("Visualizing attention patterns...")
        
        # Visualize geometric attention
        if 'geometric_attention_weights' in attention_output:
            geo_attn = attention_output['geometric_attention_weights']
            
            # Debug information
            if geo_attn:
                if isinstance(geo_attn[0], torch.Tensor):
                    print(f"Geometric attention shape: {geo_attn[0].shape}")
                else:
                    print(f"Geometric attention type: {type(geo_attn[0])}")
            
            # Use the plot_attention_patterns function
            plot_attention_patterns(
                geometric_attention=geo_attn,
                title="Geometric Attention Weights",
                figsize=(10, 8),
                save_path='./visualizations/geometric_attention.png'
            )
            
        # Visualize temporal attention
        if 'temporal_attention_weights' in attention_output:
            temp_attn = attention_output['temporal_attention_weights']
            
            # Debug information
            if isinstance(temp_attn, torch.Tensor):
                print(f"Temporal attention shape: {temp_attn.shape}")
                
                # Process multi-dimensional temporal attention
                if temp_attn.ndim == 4:
                    # For temporal attention with shape [batch_size, num_heads, seq_len, seq_len]
                    # Average over batch and heads
                    temp_attn_2d = temp_attn.mean(dim=(0, 1)).cpu().numpy()
                elif temp_attn.ndim == 3:
                    # Assume it's [num_heads, seq_len, seq_len] and average over heads
                    temp_attn_2d = temp_attn.mean(dim=0).cpu().numpy()
                else:
                    temp_attn_2d = temp_attn.cpu().numpy()
                
                # Create a simple plot of the temporal attention
                plt.figure(figsize=(10, 8))
                plt.imshow(temp_attn_2d, cmap='plasma', aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.title("Temporal Attention Weights")
                plt.xlabel('Timestep')
                plt.ylabel('Node')
                plt.savefig('./visualizations/temporal_attention.png')
            else:
                print(f"Temporal attention type: {type(temp_attn)}")
    
    print("Done! Check the visualizations folder for attention patterns.")


if __name__ == "__main__":
    main()
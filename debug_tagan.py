"""
TAGAN Debugging Script

This script helps identify and fix issues in the TAGAN model when working with our
synthetic social media data. It executes each step of the pipeline with additional
debugging outputs to diagnose problems related to:
1. Data format consistency
2. Temporal propagation handling
3. Batch size and node ID management
4. Memory leaks or excessive resource usage
"""
import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import pickle
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Import custom modules
from synthetic_social_media_data import SocialMediaDataGenerator
from preprocess_social_media import SocialMediaGraphProcessor

# Import TAGAN components
from src.tagan.model import TAGAN
from src.tagan.utils.config import TAGANConfig
from src.tagan.training.trainer import TAGANTrainer
from src.tagan.data.data_loader import TemporalGraphDataLoader, TemporalGraphDataset
from src.tagan.utils.debug_utils import enable_debugging, get_debugger, plot_attention_patterns, plot_temporal_graph_attention


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        './data',
        './data/raw',
        './data/processed',
        './checkpoints',
        './logs',
        './visualizations',
        './debug_output'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def inspect_snapshot(snapshot, idx=0, max_nodes=5, max_edges=5):
    """Inspect a graph snapshot and return a formatted description."""
    if isinstance(snapshot, dict):
        x = snapshot.get('x')
        edge_index = snapshot.get('edge_index')
        edge_attr = snapshot.get('edge_attr', None)
        node_ids = snapshot.get('node_ids', [])
        timestep = snapshot.get('timestep', None)
    elif isinstance(snapshot, tuple) and len(snapshot) >= 4:
        x, edge_index, edge_attr, node_ids = snapshot[:4]
        timestep = None
    else:
        return f"Unsupported snapshot type: {type(snapshot)}"
    
    description = [f"Snapshot {idx}:"]
    
    # Node features
    if x is not None:
        description.append(f"  x: tensor of shape {x.shape}")
        if x.shape[0] > 0:
            description.append(f"  First {min(max_nodes, x.shape[0])} node features:")
            for i in range(min(max_nodes, x.shape[0])):
                feat_str = ", ".join(f"{v:.4f}" for v in x[i][:5].tolist())
                description.append(f"    Node {i}: [{feat_str}...]")
    else:
        description.append("  x: None")
    
    # Edge indices
    if edge_index is not None:
        description.append(f"  edge_index: tensor of shape {edge_index.shape}")
        if edge_index.shape[1] > 0:
            description.append(f"  First {min(max_edges, edge_index.shape[1])} edges:")
            for i in range(min(max_edges, edge_index.shape[1])):
                description.append(f"    Edge {i}: {edge_index[0, i]} -> {edge_index[1, i]}")
    else:
        description.append("  edge_index: None")
    
    # Edge attributes
    if edge_attr is not None:
        description.append(f"  edge_attr: tensor of shape {edge_attr.shape}")
        if edge_attr.shape[0] > 0:
            description.append(f"  First {min(max_edges, edge_attr.shape[0])} edge attributes:")
            for i in range(min(max_edges, edge_attr.shape[0])):
                attr_str = ", ".join(f"{v:.4f}" for v in edge_attr[i].tolist())
                description.append(f"    Edge {i} attr: [{attr_str}]")
    else:
        description.append("  edge_attr: None")
    
    # Node IDs
    if node_ids:
        description.append(f"  node_ids: {len(node_ids)} ids")
        if len(node_ids) > 0:
            description.append(f"  First {min(max_nodes, len(node_ids))} node IDs:")
            for i in range(min(max_nodes, len(node_ids))):
                description.append(f"    Node {i} ID: {node_ids[i]}")
    else:
        description.append("  node_ids: None or empty")
    
    # Timestep (if available)
    if timestep is not None:
        description.append(f"  timestep: {timestep}")
    
    return "\n".join(description)


def debug_dataset_batch(dataset, loader, max_sequences=2, max_snapshots=3):
    """Debug dataset and batch formation."""
    debugger = get_debugger()
    debugger.log("\n===== DEBUGGING DATASET AND BATCH FORMATION =====", level="INFO")
    
    # Dataset inspection
    debugger.log(f"Dataset type: {type(dataset)}", level="INFO")
    debugger.log(f"Dataset length: {len(dataset)}", level="INFO")
    
    # Examine a few sequences
    for i in range(min(max_sequences, len(dataset))):
        sequence, label = dataset[i]
        debugger.log(f"\nSequence {i} (label: {label}):", level="INFO")
        debugger.log(f"  Length: {len(sequence)} snapshots", level="INFO")
        
        # Examine snapshots in this sequence
        for j in range(min(max_snapshots, len(sequence))):
            snapshot = sequence[j]
            debugger.log(inspect_snapshot(snapshot, j), level="INFO")
    
    # Batch formation
    debugger.log("\nExamining batch formation:", level="INFO")
    
    batch_iter = iter(loader)
    try:
        batch = next(batch_iter)
        debugger.log(f"Batch type: {type(batch)}", level="INFO")
        
        if isinstance(batch, tuple) and len(batch) >= 2:
            sequences, labels = batch[:2]
            debugger.log(f"Batch contains {len(sequences)} sequences, labels shape: {labels.shape}", level="INFO")
            
            # Check first sequence in batch
            if sequences and len(sequences) > 0:
                first_seq = sequences[0]
                debugger.log(f"First sequence in batch has {len(first_seq)} snapshots", level="INFO")
                
                # Check structure consistency
                debugger.log("Checking snapshot structure consistency...", level="INFO")
                
                # Check 1: Node feature dimensions
                node_feat_dims = set()
                edge_feat_dims = set()
                
                for seq in sequences:
                    for snapshot in seq:
                        if isinstance(snapshot, dict):
                            x = snapshot.get('x')
                            edge_attr = snapshot.get('edge_attr')
                        elif isinstance(snapshot, tuple) and len(snapshot) >= 3:
                            x, _, edge_attr = snapshot[:3]
                        else:
                            continue
                            
                        if x is not None and hasattr(x, 'shape') and len(x.shape) > 1:
                            node_feat_dims.add(x.shape[1])
                            
                        if edge_attr is not None and hasattr(edge_attr, 'shape') and len(edge_attr.shape) > 1:
                            edge_feat_dims.add(edge_attr.shape[1])
                
                if len(node_feat_dims) > 1:
                    debugger.log(f"WARNING: Inconsistent node feature dimensions: {node_feat_dims}", level="WARNING")
                else:
                    debugger.log(f"Consistent node feature dimensions: {node_feat_dims}", level="INFO")
                    
                if len(edge_feat_dims) > 1:
                    debugger.log(f"WARNING: Inconsistent edge feature dimensions: {edge_feat_dims}", level="WARNING")
                else:
                    debugger.log(f"Consistent edge feature dimensions: {edge_feat_dims}", level="INFO")
        else:
            debugger.log(f"Unexpected batch format: {type(batch)}", level="WARNING")
    except StopIteration:
        debugger.log("Empty dataloader, no batches available", level="WARNING")
    except Exception as e:
        debugger.log(f"Error inspecting batch: {str(e)}", level="ERROR")
        debugger.log(traceback.format_exc(), level="ERROR")


def debug_model_forward(model, sequence, labels=None):
    """Debug model forward pass."""
    debugger = get_debugger()
    debugger.log("\n===== DEBUGGING MODEL FORWARD PASS =====", level="INFO")
    
    # Inspect the input sequence
    debugger.inspect_sequence(sequence)
    
    # Profile the forward pass
    debugger.start_timer("forward_pass")
    try:
        # Set model to eval mode for debugging
        model.eval()
        
        # Run forward pass with attention weights
        with torch.no_grad():
            outputs = model(sequence, labels=labels, return_attention_weights=True)
        
        debugger.log("Forward pass completed successfully", level="INFO")
        
        # Log output details
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                debugger.check_tensor(value, name=f"outputs['{key}']")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                debugger.log(f"outputs['{key}']: list of {len(value)} tensors", level="TENSOR")
                if len(value) > 0:
                    debugger.check_tensor(value[0], name=f"outputs['{key}'][0]")
            elif value is None:
                debugger.log(f"outputs['{key}']: None", level="INFO")
            else:
                debugger.log(f"outputs['{key}']: {type(value)}", level="INFO")
        
        # Visualize attention weights if available
        if 'geometric_attention_weights' in outputs:
            debugger.log("Visualizing geometric attention weights", level="INFO")
            plot_attention_patterns(
                outputs['geometric_attention_weights'],
                title="Geometric Attention Patterns",
                save_path="./visualizations/geometric_attention.png"
            )
            
        if 'temporal_attention_weights' in outputs:
            debugger.log("Visualizing temporal attention weights", level="INFO")
            debugger.visualize_attention(
                outputs['temporal_attention_weights'],
                title="Temporal Attention Weights",
                save_path="./visualizations/temporal_attention.png"
            )
        
        return outputs
        
    except Exception as e:
        debugger.log(f"Error in forward pass: {str(e)}", level="ERROR")
        debugger.log(traceback.format_exc(), level="ERROR")
        return None
    finally:
        duration = debugger.end_timer("forward_pass")
        debugger.log(f"Forward pass took {duration:.4f} seconds", level="INFO")
        debugger.check_memory_usage()


def profile_model_performance(model, dataset, loader, num_batches=2):
    """Profile model performance metrics."""
    debugger = get_debugger()
    debugger.log("\n===== PROFILING MODEL PERFORMANCE =====", level="INFO")
    
    # Initialize tracking variables
    timings = []
    memory_usages = []
    
    # Baseline memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()
    else:
        baseline_memory = 0
    
    # Set model to evaluation mode
    model.eval()
    
    # Process several batches
    debugger.log(f"Processing {num_batches} batches for performance profiling", level="INFO")
    
    batch_iter = iter(loader)
    for i in range(num_batches):
        try:
            # Get next batch
            batch = next(batch_iter)
            
            if isinstance(batch, tuple) and len(batch) >= 2:
                sequences, labels = batch[:2]
                
                # Measure forward pass time and memory
                debugger.start_timer(f"batch_{i}")
                
                # Clear memory before forward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                with torch.no_grad():
                    _ = model(sequences, labels=labels)
                
                # Collect timing
                duration = debugger.end_timer(f"batch_{i}")
                timings.append(duration)
                
                # Collect memory usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() - baseline_memory
                    memory_usages.append(current_memory / 1024**2)  # Convert to MB
                else:
                    memory_usages.append(0)
                
                debugger.log(f"Batch {i}: {duration:.4f} s, Memory: {memory_usages[-1]:.2f} MB", level="PROFILE")
            
        except StopIteration:
            debugger.log(f"Ran out of batches after {i} iterations", level="WARNING")
            break
        except Exception as e:
            debugger.log(f"Error processing batch {i}: {str(e)}", level="ERROR")
            debugger.log(traceback.format_exc(), level="ERROR")
    
    # Report performance statistics
    if timings:
        avg_time = sum(timings) / len(timings)
        max_time = max(timings)
        min_time = min(timings)
        
        debugger.log(f"Performance Summary:", level="PROFILE")
        debugger.log(f"  Average batch time: {avg_time:.4f} s", level="PROFILE")
        debugger.log(f"  Min/Max batch time: {min_time:.4f} s / {max_time:.4f} s", level="PROFILE")
        
        if memory_usages:
            avg_memory = sum(memory_usages) / len(memory_usages)
            max_memory = max(memory_usages)
            debugger.log(f"  Average memory usage: {avg_memory:.2f} MB", level="PROFILE")
            debugger.log(f"  Peak memory usage: {max_memory:.2f} MB", level="PROFILE")
    else:
        debugger.log("No timing data collected", level="WARNING")


def diagnose_temporal_asymmetric_attention(model, sequence, time_steps=5):
    """
    Diagnose the temporal asymmetric attention mechanism.
    
    This function specifically tests and visualizes how the model handles
    asymmetric temporal dependencies.
    """
    debugger = get_debugger()
    debugger.log("\n===== DIAGNOSING TEMPORAL ASYMMETRIC ATTENTION =====", level="INFO")
    
    # Check time steps
    if not sequence or len(sequence) < 2:
        debugger.log("Need at least 2 time steps to diagnose temporal attention", level="WARNING")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Create sequence for testing with variable time steps
    test_sequence = sequence[:min(time_steps, len(sequence))]
    
    # Define a test function to capture attention weights at each layer
    def run_with_attention_capture():
        with torch.no_grad():
            outputs = model(test_sequence, return_attention_weights=True)
            
            # Extract attention weights
            attn_weights = {}
            
            if 'geometric_attention_weights' in outputs:
                attn_weights['geometric'] = outputs['geometric_attention_weights']
                
            if 'temporal_attention_weights' in outputs:
                attn_weights['temporal'] = outputs['temporal_attention_weights']
                
            return outputs, attn_weights
    
    # Run the model and capture attention weights
    debugger.start_timer("attention_capture")
    try:
        outputs, attn_weights = run_with_attention_capture()
        
        # Analyze asymmetric temporal attention patterns
        if 'temporal' in attn_weights:
            temporal_attn = attn_weights['temporal']
            
            # Visualize the temporal attention patterns
            debugger.log("Visualizing temporal attention patterns...", level="INFO")
            
            # Create a more sophisticated visualization
            if isinstance(temporal_attn, torch.Tensor):
                # If tensor, shape is typically [batch, heads, seq_len, seq_len]
                if temporal_attn.dim() >= 3:
                    # Check for asymmetry in the attention weights
                    # In asymmetric attention, the upper and lower triangles should be different
                    
                    # Get a single attention matrix (average over batches and heads if needed)
                    if temporal_attn.dim() == 4:
                        attn_matrix = temporal_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                    elif temporal_attn.dim() == 3:
                        attn_matrix = temporal_attn[0].cpu().numpy()  # [seq_len, seq_len]
                    else:
                        attn_matrix = temporal_attn.cpu().numpy()
                    
                    # Calculate asymmetry score (difference between upper and lower triangles)
                    triu_indices = np.triu_indices(attn_matrix.shape[0], k=1)
                    tril_indices = np.tril_indices(attn_matrix.shape[0], k=-1)
                    
                    upper_weights = attn_matrix[triu_indices]
                    lower_weights = attn_matrix[tril_indices]
                    
                    # Transpose lower weights to match upper weights for comparison
                    lower_weights_t = np.array([attn_matrix[j, i] for i, j in zip(*tril_indices)])
                    
                    asymmetry_score = np.mean(np.abs(upper_weights - lower_weights_t))
                    normalized_asymmetry = asymmetry_score / (np.mean(upper_weights) + 1e-8)
                    
                    debugger.log(f"Temporal attention asymmetry score: {asymmetry_score:.4f}", level="ANALYSIS")
                    debugger.log(f"Normalized asymmetry (relative to mean attention weight): {normalized_asymmetry:.4f}", level="ANALYSIS")
                    
                    # Visualize the asymmetry pattern
                    plt.figure(figsize=(14, 4))
                    
                    # Plot 1: Full attention matrix
                    plt.subplot(1, 3, 1)
                    plt.imshow(attn_matrix, cmap='viridis')
                    plt.colorbar(label='Attention weight')
                    plt.title('Temporal Attention Matrix')
                    plt.xlabel('Target Timestep')
                    plt.ylabel('Source Timestep')
                    
                    # Plot 2: Asymmetry visualization (abs difference between upper and lower triangles)
                    asymmetry_matrix = np.zeros_like(attn_matrix)
                    for (i, j), (j_t, i_t) in zip(zip(*triu_indices), zip(*tril_indices)):
                        diff = abs(attn_matrix[i, j] - attn_matrix[j_t, i_t])
                        asymmetry_matrix[i, j] = diff
                        asymmetry_matrix[j_t, i_t] = diff
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(asymmetry_matrix, cmap='hot')
                    plt.colorbar(label='Asymmetry magnitude')
                    plt.title('Attention Asymmetry')
                    plt.xlabel('Timestep')
                    plt.ylabel('Timestep')
                    
                    # Plot 3: Past vs. Future attention comparison
                    timesteps = attn_matrix.shape[0]
                    mid_idx = timesteps // 2
                    if mid_idx > 0 and mid_idx < timesteps:
                        # For a middle timestep, compare attention to past vs. future
                        past_attention = attn_matrix[mid_idx, :mid_idx]
                        future_attention = attn_matrix[mid_idx, mid_idx+1:] if mid_idx < timesteps-1 else np.array([])
                        
                        plt.subplot(1, 3, 3)
                        
                        x_past = np.arange(mid_idx)
                        x_future = np.arange(mid_idx+1, timesteps) if mid_idx < timesteps-1 else np.array([])
                        
                        if len(past_attention) > 0:
                            plt.plot(x_past, past_attention, 'b-o', label='Attention to Past')
                        if len(future_attention) > 0:
                            plt.plot(x_future, future_attention, 'r-o', label='Attention to Future')
                        
                        plt.axvline(x=mid_idx, color='green', linestyle='--', label='Current Timestep')
                        plt.title(f'Past vs. Future Attention (from t={mid_idx})')
                        plt.xlabel('Timestep')
                        plt.ylabel('Attention Weight')
                        plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig('./visualizations/temporal_asymmetry_analysis.png', dpi=300)
                    debugger.log("Temporal asymmetry analysis saved to ./visualizations/temporal_asymmetry_analysis.png", level="INFO")
                    plt.close()
            
            # If we have a sequence of graphs, visualize the temporal attention over the graph structure
            if 'geometric' in attn_weights and test_sequence:
                try:
                    plot_temporal_graph_attention(
                        attn_weights['geometric'], 
                        test_sequence,
                        title="Temporal-Geometric Attention Patterns",
                        save_path="./visualizations/temporal_geometric_attention.png",
                        show_plot=False
                    )
                    debugger.log("Temporal-geometric attention visualization saved", level="INFO")
                except Exception as e:
                    debugger.log(f"Error creating temporal-geometric attention plot: {str(e)}", level="ERROR")

    except Exception as e:
        debugger.log(f"Error diagnosing temporal attention: {str(e)}", level="ERROR")
        debugger.log(traceback.format_exc(), level="ERROR")
    finally:
        duration = debugger.end_timer("attention_capture")
        debugger.log(f"Temporal attention diagnosis completed in {duration:.4f} seconds", level="INFO")


def main():
    """Main debug function."""
    # Setup debugging output
    enable_debugging('./debug_output/tagan_debug.log')
    debugger = get_debugger()
    
    # Make sure all required directories exist
    ensure_directories()
    
    debugger.log("Starting TAGAN debugging script", level="INFO")
    
    # Load configuration for debugging
    config = TAGANConfig(
        node_feature_dim=2,  # Match the actual node feature dimension from the data
        edge_feature_dim=1,  # Match the actual edge feature dimension from the data
        hidden_dim=128,
        output_dim=2,  # Binary classification for example
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_layer_norm=True,
        bidirectional=False,  # Use unidirectional to emphasize asymmetric dependencies
        causal_attention=True,  # Enable causal masking in attention
        asymmetric_temporal_bias=True,  # Enable asymmetric bias in temporal attention
        time_aware=True,  # Make propagation time-aware
        use_skip_connection=True,
        window_size=5,  # Temporal window size
        loss_type='bce',  # Use bce (Binary Cross Entropy) for binary classification
        aggregation_method='mean',
        learnable_distance=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    debugger.log("Loaded model configuration", level="INFO")
    
    try:
        # Attempt to create a model
        debugger.start_timer("model_creation")
        model = TAGAN(config)
        debugger.end_timer("model_creation")
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        debugger.log(f"Created TAGAN model (on {device})", level="INFO")
        
        # Generate synthetic data for debugging
        debugger.start_timer("data_generation")
        
        # Simple synthetic data generation - replace with your actual data loading
        data_generator = SocialMediaDataGenerator(
            num_users=50,
            num_threads=20,
            max_posts_per_thread=15,
            max_replies_per_post=5,
            time_span_days=5,
            controversial_ratio=0.3
        )
        raw_data = data_generator.generate_data()
        
        # Create data directories
        os.makedirs("./data/raw", exist_ok=True)
        os.makedirs("./data/processed", exist_ok=True)
        
        # Save raw data to files expected by the processor
        raw_data.to_csv("./data/raw/social_media_data.csv", index=False)
        
        # Generate user profiles
        users = data_generator._generate_users()
        with open("./data/raw/user_profiles.json", "w") as f:
            json.dump(users, f, indent=2)
        
        # Generate thread labels
        thread_labels = pd.DataFrame({
            "thread_id": raw_data["thread_id"].unique(),
            "controversial": [raw_data[raw_data["thread_id"] == tid]["controversial"].iloc[0] for tid in raw_data["thread_id"].unique()]
        })
        thread_labels.to_csv("./data/raw/thread_labels.csv", index=False)
        
        # Process into graph format
        processor = SocialMediaGraphProcessor(
            raw_data_dir="./data/raw",
            processed_data_dir="./data/processed",
            text_embedding_dim=64
        )
        train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = processor.process_data()
        
        # Create dataset and loader using training data
        dataset = TemporalGraphDataset(train_sequences, train_labels)
        loader = TemporalGraphDataLoader(dataset, batch_size=4, shuffle=True)
        
        debugger.end_timer("data_generation")
        debugger.log("Generated synthetic data for debugging", level="INFO")
        
        # Debug dataset and batch formation
        debug_dataset_batch(dataset, loader)
        
        # Get a sample sequence for debugging
        sample_sequence, sample_label = dataset[0]
        sample_labels = torch.tensor([sample_label], device=device)
        
        # Debug forward pass
        outputs = debug_model_forward(model, sample_sequence, sample_labels)
        
        # Diagnose temporal asymmetric attention
        diagnose_temporal_asymmetric_attention(model, sample_sequence)
        
        # Profile model performance
        profile_model_performance(model, dataset, loader)
        
        debugger.log("TAGAN debugging completed successfully", level="INFO")
        
    except Exception as e:
        debugger.log(f"Error during debugging: {str(e)}", level="ERROR")
        debugger.log(traceback.format_exc(), level="ERROR")


if __name__ == "__main__":
    main()
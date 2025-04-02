"""
TAGAN Social Media Analysis Pipeline

This script runs the entire pipeline for temporal graph analysis on social media data:
1. Generate synthetic social media data
2. Preprocess raw data into temporal graph format
3. Train and evaluate the TAGAN model
4. Visualize attention patterns and results
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# Import our custom modules
from synthetic_social_media_data import SocialMediaDataGenerator
from preprocess_social_media import SocialMediaGraphProcessor

# Import TAGAN components
from src.tagan.model import TAGAN
from src.tagan.utils.config import TAGANConfig
from src.tagan.training.trainer import TAGANTrainer
from src.tagan.data.data_loader import TemporalGraphDataLoader, TemporalGraphDataset
from src.tagan.visualization.attention_vis import plot_attention_patterns, plot_graph_with_attention


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        './data',
        './data/raw',
        './data/processed',
        './checkpoints',
        './logs',
        './visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def generate_data():
    """
    Generate synthetic social media data.
    
    Returns:
        DataFrame containing the generated data
    """
    print("\n==== STEP 1: GENERATING SYNTHETIC SOCIAL MEDIA DATA ====")
    
    # Initialize generator with parameters
    generator = SocialMediaDataGenerator(
        num_users=50,
        num_threads=100,
        max_posts_per_thread=20,
        max_replies_per_post=5,
        time_span_days=7,
        controversial_ratio=0.3,
        output_dir="./data/raw"
    )
    
    # Generate data
    df = generator.generate_data()
    
    # Print statistics
    generator.analyze_data(df)
    
    print("==== COMPLETED DATA GENERATION ====\n")
    
    return df


def preprocess_data():
    """
    Preprocess raw social media data into temporal graph format.
    
    Returns:
        Tuple of datasets and loaders
    """
    print("\n==== STEP 2: PREPROCESSING RAW DATA INTO TEMPORAL GRAPHS ====")
    
    # Initialize processor with parameters
    processor = SocialMediaGraphProcessor(
        raw_data_dir="./data/raw",
        processed_data_dir="./data/processed",
        text_embedding_dim=16,
        snapshot_duration=6,  # hours
        max_snapshots=10,
        min_nodes_per_snapshot=3
    )
    
    # Process data
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = processor.process_data()
    
    # Create datasets
    print("\nCreating TemporalGraphDataset instances...")
    train_dataset = TemporalGraphDataset(data=train_sequences, labels=train_labels)
    val_dataset = TemporalGraphDataset(data=val_sequences, labels=val_labels)
    test_dataset = TemporalGraphDataset(data=test_sequences, labels=test_labels)
    
    # Print dataset statistics
    print("\nTraining dataset statistics:")
    train_stats = train_dataset.get_statistics()
    for key, value in train_stats.items():
        if isinstance(value, (int, float, np.number)):
            print(f"  {key}: {value}")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Create data loaders
    batch_size = 4
    train_loader = TemporalGraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TemporalGraphDataLoader(val_dataset, batch_size=batch_size)
    test_loader = TemporalGraphDataLoader(test_dataset, batch_size=batch_size)
    
    print(f"\nCreated data loaders with batch size {batch_size}:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print("==== COMPLETED DATA PREPROCESSING ====\n")
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def configure_and_train_model(train_loader, val_loader, test_loader, test_dataset):
    """
    Configure, train and evaluate the TAGAN model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        test_dataset: Test dataset for visualization
        
    Returns:
        Tuple of (model, trainer, test_results)
    """
    print("\n==== STEP 3: CONFIGURING AND TRAINING TAGAN MODEL ====")
    
    # Create model configuration
    # Sample the first batch to get feature dimensions
    first_batch = next(iter(train_loader))
    first_sequence = first_batch[0]
    
    if first_sequence:
        # Get dimensions from the first snapshot
        first_snapshot = first_sequence[0]
        node_feature_dim = first_snapshot['x'].size(1)
        
        if 'edge_attr' in first_snapshot and first_snapshot['edge_attr'].size(0) > 0:
            edge_feature_dim = first_snapshot['edge_attr'].size(1)
        else:
            edge_feature_dim = 1
            
        print(f"Detected dimensions - Node features: {node_feature_dim}, Edge features: {edge_feature_dim}")
    else:
        print("WARNING: Could not detect dimensions from data, using defaults")
        node_feature_dim = 16
        edge_feature_dim = 2
    
    config = TAGANConfig(
        # Feature dimensions
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=64,
        output_dim=1,  # Binary classification (controversial or not)
        
        # Attention parameters
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        
        # Training parameters
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        loss_type='bce',
        
        # Model components
        edge_importance=True,
        use_layer_norm=True,
        memory_decay_factor=0.9,
        gru_bias=True,
        temporal_attention_dim=64,
        leaky_relu_slope=0.2,
        use_edge_features=True,
        
        # Temporal propagation options
        time_aware=True,
        bidirectional=False,
        use_skip_connection=True,
        use_gating=True,
        temporal_window_size=3,
        aggregation_method='mean',
        use_residual=True
    )
    
    print(f"Model will run on: {config.device}")
    
    # Create model
    model = TAGAN(config)
    model.to(config.device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Patch model's forward method to convert labels to float type
    original_forward = model.forward
    
    def patched_forward(self, graph_sequence, labels=None, return_attention_weights=False):
        # Convert labels to float for BCE loss
        if labels is not None and labels.dtype != torch.float:
            print(f"Converting labels from {labels.dtype} to float for BCE loss")
            labels = labels.float()
        
        # Call original forward
        return original_forward(graph_sequence, labels, return_attention_weights)
    
    # Apply the patch
    model.forward = patched_forward.__get__(model, TAGAN)
    print("Applied patch to convert labels to float type")
    
    # Create trainer
    trainer = TAGANTrainer(
        model=model,
        config=config,
        checkpoint_dir='./checkpoints',
        log_dir='./logs'
    )
    
    # Train model
    print("\nTraining model...")
    num_epochs = 10
    
    try:
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            validate_every=1,
            save_best=True
        )
        
        # Plot training results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_results['train_losses'], label='Train Loss')
        if training_results['val_losses']:  # Check if validation losses exist
            plt.plot(training_results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        
        # Check if we have validation metrics
        if training_results['val_metrics']:
            # Extract accuracy and F1 values across epochs
            val_accuracy = [metrics_dict.get('accuracy', 0) for metrics_dict in training_results['val_metrics']]
            val_f1 = [metrics_dict.get('f1', 0) for metrics_dict in training_results['val_metrics']]
            
            plt.plot(val_accuracy, label='Accuracy')
            plt.plot(val_f1, label='F1 Score')
        else:
            # Use training metrics if no validation
            train_accuracy = [metrics_dict.get('accuracy', 0) for metrics_dict in training_results['train_metrics']]
            train_f1 = [metrics_dict.get('f1', 0) for metrics_dict in training_results['train_metrics']]
            
            plt.plot(train_accuracy, label='Train Accuracy')
            plt.plot(train_f1, label='Train F1 Score')
            
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Model Performance Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./visualizations/training_results.png')
        
        print(f"Training completed for {num_epochs} epochs")
        
        # Report best validation loss if available
        if training_results['val_losses']:
            print(f"Best validation loss: {min(training_results['val_losses']):.4f}")
        else:
            print(f"Best training loss: {min(training_results['train_losses']):.4f}")
            
        # Report accuracy metrics if available
        if training_results['val_metrics'] and len(training_results['val_metrics']) > 0:
            # Extract accuracy from each epoch's metrics dictionary
            val_accuracy = [metrics_dict.get('accuracy', 0) for metrics_dict in training_results['val_metrics']]
            if val_accuracy:
                print(f"Best validation accuracy: {max(val_accuracy):.4f}")
        else:
            # Use training metrics if no validation
            train_accuracy = [metrics_dict.get('accuracy', 0) for metrics_dict in training_results['train_metrics']]
            if train_accuracy:
                print(f"Best training accuracy: {max(train_accuracy):.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test model
    print("\nTesting model...")
    try:
        # Handle checkpoint loading errors
        try:
            test_results = trainer.test(test_loader, model_path='./checkpoints/best_model.pt')
            
            print(f"Test results:")
            print(f"  Loss: {test_results['loss']:.4f}")
            print(f"  Accuracy: {test_results['metrics']['accuracy']:.4f}")
            print(f"  F1 Score: {test_results['metrics']['f1']:.4f}")
            
            # Save test results
            with open('./logs/test_results.pkl', 'wb') as f:
                pickle.dump(test_results, f)
                
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("Model checkpoint dimensions don't match current model. Using current model without loading checkpoint.")
                # Run test without loading checkpoint
                test_results = trainer.test(test_loader, model_path=None)
                print(f"Test results (using current model):")
                print(f"  Loss: {test_results['loss']:.4f}")
                print(f"  Accuracy: {test_results['metrics']['accuracy']:.4f}")
                print(f"  F1 Score: {test_results['metrics']['f1']:.4f}")
            else:
                raise e
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        test_results = None
    
    print("==== COMPLETED MODEL TRAINING AND EVALUATION ====\n")
    
    return model, trainer, test_results


def visualize_attention(model, test_dataset):
    """
    Visualize attention patterns from the TAGAN model.
    
    Args:
        model: Trained TAGAN model
        test_dataset: Test dataset containing sequences
    """
    print("\n==== STEP 4: VISUALIZING ATTENTION PATTERNS ====")
    
    try:
        # Get a sample sequence
        sample_idx = 0
        sample_sequence, sample_label = test_dataset[sample_idx]
        
        # Set model to evaluation mode
        model.eval()
        
        # Get attention weights
        with torch.no_grad():
            attention_output = model.infer_with_attention(sample_sequence)
            
            # Visualize geometric attention
            if 'geometric_attention_weights' in attention_output:
                geo_attn = attention_output['geometric_attention_weights']
                
                # Debug information
                if geo_attn:
                    if isinstance(geo_attn[0], torch.Tensor):
                        print(f"Geometric attention shape: {geo_attn[0].shape}")
                    else:
                        print(f"Geometric attention type: {type(geo_attn[0])}")
                
                # Handle different geometric attention formats
                processed_geo_attn = []
                for attn in geo_attn:
                    if isinstance(attn, dict) and 'attention' in attn:
                        if isinstance(attn['attention'], torch.Tensor):
                            processed_geo_attn.append(attn['attention'])
                        else:
                            processed_geo_attn.append(attn['attention'])
                    else:
                        processed_geo_attn.append(attn)
                
                # Create enhanced visualization using the better plot_temporal_graph_attention function if available
                try:
                    # Extract relevant graph data for visualization
                    graph_data = []
                    for i, snapshot in enumerate(sample_sequence):
                        graph_data.append({
                            'edge_index': snapshot['edge_index'],
                            'node_ids': snapshot['node_ids'] if 'node_ids' in snapshot else list(range(snapshot['x'].size(0)))
                        })
                    
                    # Create network visualization with attention weights
                    plot_temporal_graph_attention(
                        geometric_attention=processed_geo_attn,
                        graph_data=graph_data,
                        title=f"Geometric Attention Network (Label: {'Controversial' if sample_label == 1 else 'Non-controversial'})",
                        figsize=(16, 10),
                        save_path='./visualizations/geometric_attention.png',
                        show_plot=False
                    )
                except Exception as e:
                    print(f"Could not create network visualization: {str(e)}")
                    # Fallback to basic visualization
                    plot_attention_patterns(
                        geometric_attention=processed_geo_attn,
                        title=f"Geometric Attention Weights (Label: {'Controversial' if sample_label == 1 else 'Non-controversial'})",
                        figsize=(12, 8),
                        save_path='./visualizations/geometric_attention.png',
                        show_plot=False
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
                    
                    # Create an enhanced temporal attention visualization
                    fig, ax = plt.subplots(figsize=(12, 9))
                    im = ax.imshow(temp_attn_2d, cmap='viridis', aspect='auto')
                    
                    # Add colorbar
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Attention Weight', fontsize=12)
                    
                    # Add arrows to show asymmetric attention
                    n_timesteps = temp_attn_2d.shape[0]
                    for i in range(n_timesteps):
                        for j in range(n_timesteps):
                            if i != j and temp_attn_2d[i, j] > 0.5:  # Only show strong connections
                                ax.annotate('',
                                    xy=(j, i), xycoords='data',
                                    xytext=(i, j), textcoords='data',
                                    arrowprops=dict(arrowstyle="->", color='white', alpha=0.6*temp_attn_2d[i, j]),
                                )
                    
                    # Add better labels
                    ax.set_xticks(np.arange(n_timesteps))
                    ax.set_yticks(np.arange(n_timesteps))
                    ax.set_xticklabels([f'T{i}' for i in range(n_timesteps)], fontsize=10)
                    ax.set_yticklabels([f'T{i}' for i in range(n_timesteps)], fontsize=10)
                    
                    plt.title(f"Temporal Attention Patterns (Label: {'Controversial' if sample_label == 1 else 'Non-controversial'})", fontsize=14)
                    plt.xlabel('Target Timestep', fontsize=12)
                    plt.ylabel('Source Timestep', fontsize=12)
                    plt.tight_layout()
                    plt.savefig('./visualizations/temporal_attention.png', dpi=300)
                    plt.close()
                else:
                    print(f"Temporal attention type: {type(temp_attn)}")
            
            # Also visualize the prediction
            if 'logits' in attention_output:
                logits = attention_output['logits']
                print(f"Logits shape: {logits.shape}")
                
                # Handle batch of predictions - select the first example for visualization
                if logits.numel() > 1:
                    print(f"Multiple predictions detected (batch size: {logits.size(0)}). Selecting first example for visualization.")
                    # Get the first prediction
                    single_logit = logits[0]
                    pred_prob = torch.sigmoid(single_logit).item()
                else:
                    # Single prediction
                    pred_prob = torch.sigmoid(logits).item()
                    
                    # Use 0.7 threshold to align with our bias correction and metrics calculation
                    pred_label = 1 if pred_prob > 0.7 else 0
                
                print(f"Sample sequence (true label: {sample_label}):")
                print(f"  Predicted probability: {pred_prob:.4f}")
                print(f"  Predicted label: {pred_label} (Controversial: {'Yes' if pred_label == 1 else 'No'})")
                
                # Create an enhanced prediction visualization
                fig, ax = plt.subplots(figsize=(10, 7))
                
                # Create a more visually appealing bar chart
                bars = ax.bar(
                    ['Non-controversial', 'Controversial'],
                    [1-pred_prob, pred_prob],
                    color=['#3498db', '#e74c3c'] if sample_label == 1 else ['#3498db', '#e74c3c'],
                    alpha=0.8,
                    width=0.5
                )
                
                # Add a horizontal line at 0.5 for decision threshold
                ax.axhline(y=0.5, color='#7f8c8d', linestyle='--', alpha=0.7, label='Decision Threshold')
                
                # Highlight the true label
                true_idx = sample_label
                bars[true_idx].set_color('#2ecc71' if pred_label == sample_label else '#e74c3c')
                bars[true_idx].set_edgecolor('black')
                bars[true_idx].set_linewidth(2)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.02,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
                
                # Add indication of correct/incorrect prediction
                prediction_status = "✓ Correct" if pred_label == sample_label else "✗ Incorrect"
                ax.text(
                    0.5, 0.92,
                    prediction_status,
                    transform=ax.transAxes,
                    ha='center',
                    fontsize=14,
                    color='#2ecc71' if pred_label == sample_label else '#e74c3c',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        alpha=0.8
                    )
                )
                
                ax.set_ylim(0, 1.1)
                ax.set_title(f"Model Prediction\nTrue Label: {'Controversial' if sample_label == 1 else 'Non-controversial'}", fontsize=14)
                ax.set_ylabel('Probability', fontsize=12)
                ax.legend()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('./visualizations/prediction.png', dpi=300)
                plt.close()
    
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nAttention pattern visualizations saved to ./visualizations/")
    print("==== COMPLETED VISUALIZATION ====\n")


def main():
    """Main function to run the complete pipeline."""
    print("Starting TAGAN Social Media Analysis Pipeline")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create required directories
    ensure_directories()
    
    # Generate data if it doesn't exist
    if not os.path.exists("./data/raw/social_media_data.csv"):
        generate_data()
    else:
        print("\n==== USING EXISTING RAW DATA ====")
    
    # Preprocess data if it doesn't exist
    if not os.path.exists("./data/processed/processed_social_media_data.pkl"):
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = preprocess_data()
    else:
        print("\n==== LOADING PREPROCESSED DATA ====")
        # Load preprocessed data
        with open("./data/processed/processed_social_media_data.pkl", 'rb') as f:
            processed_data = pickle.load(f)
        
        train_sequences = processed_data['train_sequences']
        train_labels = processed_data['train_labels']
        val_sequences = processed_data['val_sequences']
        val_labels = processed_data['val_labels']
        test_sequences = processed_data['test_sequences']
        test_labels = processed_data['test_labels']
        
        # Create datasets and loaders
        train_dataset = TemporalGraphDataset(data=train_sequences, labels=train_labels)
        val_dataset = TemporalGraphDataset(data=val_sequences, labels=val_labels)
        test_dataset = TemporalGraphDataset(data=test_sequences, labels=test_labels)
        
        batch_size = 4
        train_loader = TemporalGraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TemporalGraphDataLoader(val_dataset, batch_size=batch_size)
        test_loader = TemporalGraphDataLoader(test_dataset, batch_size=batch_size)
        
        print(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    # Train and evaluate model
    model, trainer, test_results = configure_and_train_model(
        train_loader, val_loader, test_loader, test_dataset
    )
    
    # Visualize attention patterns
    visualize_attention(model, test_dataset)
    
    print("\n==== PIPELINE COMPLETED SUCCESSFULLY ====")
    if test_results:
        print(f"Final test accuracy: {test_results['metrics']['accuracy']:.4f}")
        print(f"Final test F1 score: {test_results['metrics']['f1']:.4f}")
    print("Visualizations saved to ./visualizations/")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    runtime = end_time - start_time
    print(f"\nTotal runtime: {runtime}")
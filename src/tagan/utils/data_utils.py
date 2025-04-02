"""
Data utility functions for TAGAN.

This module provides utilities for creating synthetic data and
other data manipulation tasks.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
from sklearn.model_selection import train_test_split

def create_synthetic_data(
    num_samples: int = 100,
    num_nodes_range: Tuple[int, int] = (5, 20),
    num_edges_per_node: int = 2,
    node_feature_dim: int = 10,
    edge_feature_dim: int = 5,
    sequence_length: int = 4,
    num_classes: int = 2,
    balanced: bool = True
) -> List[Tuple]:
    """
    Create synthetic temporal graph data for testing.
    
    Args:
        num_samples: Number of graph sequences to generate
        num_nodes_range: Range of number of nodes (min, max)
        num_edges_per_node: Average number of edges per node
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        sequence_length: Number of time steps in each sequence
        num_classes: Number of classes for labels
        balanced: Whether to create balanced classes
        
    Returns:
        List of (graph_sequence, label) tuples
    """
    data = []
    
    # Create balanced labels if requested
    if balanced and num_classes > 0:
        labels_per_class = num_samples // num_classes
        labels = []
        for i in range(num_classes):
            labels.extend([i] * labels_per_class)
        # Add any remaining samples to the last class
        remaining = num_samples - len(labels)
        if remaining > 0:
            labels.extend([num_classes - 1] * remaining)
        # Shuffle the labels
        np.random.shuffle(labels)
    else:
        # Random labels
        labels = np.random.randint(0, max(1, num_classes), size=num_samples)
    
    # Don't convert the entire array to boolean - we need to index into it
    # We'll convert individual labels when creating the output
    
    for i in range(num_samples):
        # Generate a graph sequence
        graph_sequence = []
        
        # Randomly choose the number of nodes for this graph
        num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1] + 1)
        
        # Create a base graph structure using NetworkX
        G = nx.barabasi_albert_graph(num_nodes, min(num_edges_per_node, num_nodes - 1))
        
        # Get edge indices
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        
        # Make it bidirectional by adding reversed edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create unique node IDs
        node_ids = list(range(num_nodes))
        
        # Create a sequence of snapshots with minor variations
        for t in range(sequence_length):
            # Generate node features with some temporal variation
            node_features = torch.randn(num_nodes, node_feature_dim)
            
            # Add temporal signal related to the label
            # Check if this is a positive class (label=1 for binary)
            is_positive = (labels[i] == 1) if num_classes == 2 else (labels[i] > 0)
            
            if is_positive:  # Positive class
                # Add a positive signal to some nodes
                signal_strength = 0.5 + 0.2 * t / sequence_length  # Increasing signal over time
                node_features[:, 0] += signal_strength
            else:  # Negative class
                # Add a negative signal to some nodes
                signal_strength = -0.5 - 0.2 * t / sequence_length  # Decreasing signal over time
                node_features[:, 0] += signal_strength
            
            # Add random variation
            node_features += 0.1 * torch.randn(num_nodes, node_feature_dim)
            
            # Generate edge features
            if edge_feature_dim > 0:
                edge_features = torch.randn(edge_index.size(1), edge_feature_dim)
            else:
                edge_features = None
            
            # Add to the sequence
            snapshot = (node_features, edge_index, edge_features, node_ids)
            graph_sequence.append(snapshot)
        # Convert label with appropriate data type for the loss function
        if num_classes == 2:
            # For binary classification with BCE loss, use float
            label = torch.tensor(float(labels[i]), dtype=torch.float)
        else:
            # For multi-class classification, use long
            label = torch.tensor(labels[i], dtype=torch.long)
        
        # Add to data
        data.append((graph_sequence, label))
    
    return data
    return data
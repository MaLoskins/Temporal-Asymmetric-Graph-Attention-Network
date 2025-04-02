"""
TAGAN Model implementation.

This module provides the main Temporal Asymmetric Geometric Attention Network (TAGAN) model,
integrating all components into a cohesive architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from .layers.geometric_attention import GeometricAttention
from .layers.graph_attention import TAGANGraphAttention
from .layers.temporal_attention import TemporalAttention, AsymmetricTemporalAttention
from .layers.temporal_propagation import TemporalPropagation, AsymmetricTemporalCell
from .layers.classification import ClassificationModule, RegressionModule, TemporalLossModule
from .layers.classification import ClassificationModule, RegressionModule, TemporalLossModule
from .utils.config import TAGANConfig


class TAGAN(nn.Module):
    """
    Temporal Asymmetric Geometric Attention Network (TAGAN).
    
    This model combines geometric attention for graph structure with
    temporal attention for capturing asymmetric temporal dependencies.
    
    Attributes:
        config (TAGANConfig): Configuration parameters
        node_embedding (nn.Linear): Node feature embedding
        edge_embedding (nn.Linear): Edge feature embedding (if used)
        geometric_attention_layers (nn.ModuleList): List of geometric attention layers
        temporal_propagation (TemporalPropagation): Temporal propagation module
        temporal_attention (TemporalAttention): Temporal attention module
        classification_head (ClassificationHead): Classification head
        loss_fn (TAGANLoss): Loss function
    """
    
    def __init__(
        self,
        config: TAGANConfig
    ):
        """
        Initialize the TAGAN model.
        
        Args:
            config: Configuration parameters
        """
        super(TAGAN, self).__init__()
        
        self.config = config
        
        # Input embeddings
        self.node_embedding = nn.Linear(config.node_feature_dim, config.hidden_dim)
        
        if config.edge_feature_dim > 0:
            self.edge_embedding = nn.Linear(config.edge_feature_dim, config.hidden_dim)
        else:
            self.edge_embedding = None
        
        # Geometric attention layers (encoder)
        self.geometric_attention_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            self.geometric_attention_layers.append(
                TAGANGraphAttention(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    distance_metric="scaled_dot_product" if config.learnable_distance else "euclidean",
                    use_layer_norm=config.use_layer_norm,
                    learnable_distance=config.learnable_distance
                )
            )
        
        # Temporal propagation
        self.temporal_propagation = TemporalPropagation(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            time_aware=config.time_aware,
            bidirectional=config.bidirectional,
            use_layer_norm=config.use_layer_norm,
            use_skip_connection=config.use_skip_connection,
            use_gating=config.use_gating,
            window_size=config.temporal_window_size,
            aggregation=config.aggregation_method,
            residual=config.use_residual
        )
        
        # Temporal attention with enhanced asymmetric capabilities
        self.temporal_attention = AsymmetricTemporalAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            causal=config.causal_attention,  # Configurable causal attention
            time_aware=True,
            use_layer_norm=config.use_layer_norm,
            asymmetric_window_size=config.window_size,
            relative_position_bias=config.asymmetric_temporal_bias
        )
        
        # Classification head
        self.classification_head = ClassificationModule(
            hidden_dim=config.hidden_dim,
            task_configs={
                'output_dim': config.output_dim,
                'task_type': config.loss_type
            },
            multi_task=False,
            num_layers=2,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm
        )
        
        # Loss function
        self.loss_fn = TemporalLossModule(
            task_configs={'default': {
                'task_type': config.loss_type,
                'output_dim': config.output_dim
            }},
            loss_config={
                'reduction': 'mean',
                'focal_alpha': config.focal_alpha,
                'focal_gamma': config.focal_gamma
            }
        )
        
        # Skip connections and layer normalization
        if config.use_layer_norm:
            self.skip_layer_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.skip_layer_norm = None
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize input embeddings
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)
        
        if self.edge_embedding is not None:
            nn.init.xavier_uniform_(self.edge_embedding.weight)
            nn.init.zeros_(self.edge_embedding.bias)
    
    def forward(
        self,
        graph_sequence: List[Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]]],
        labels: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for TAGAN.
        
        Args:
            graph_sequence: List of graph snapshots for each time step in the sequence
                          Each snapshot can be either:
                          - A dictionary with keys 'x', 'edge_index', 'edge_attr', 'node_ids'
                          - A tuple of (node_features, edge_index, edge_attr, node_ids)
            labels: Optional ground truth labels for training
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs, including:
            - predictions: Model predictions
            - loss: Loss value (if labels provided)
            - attention_weights: Attention weights (if requested)
        """
        # Process each graph in the sequence
        node_embeddings_seq = []
        node_masks_seq = []
        
        # Track all unique node IDs
        all_node_ids = set()
        for snapshot in graph_sequence:
            # Handle both dictionary and tuple formats
            if isinstance(snapshot, dict):
                node_ids = snapshot['node_ids']
            elif isinstance(snapshot, tuple):
                # Handle tuples of varying length
                if len(snapshot) >= 4:
                    # Standard format (node_features, edge_index, edge_attr, node_ids)
                    node_ids = snapshot[3]
                else:
                    raise ValueError(f"Snapshot tuple has incorrect format. Expected at least 4 elements, got {len(snapshot)}")
            else:
                raise ValueError(f"Unsupported snapshot type: {type(snapshot)}")
            all_node_ids.update(node_ids)
        
        # Convert to sorted list for consistent indexing
        all_node_ids = sorted(list(all_node_ids))
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        
        # Process each time step
        temporal_attention_inputs = []
        attn_weights_seq = []
        
        # Get the device of the model
        device = next(self.parameters()).device
        
        for t, snapshot in enumerate(graph_sequence):
            # Handle both dictionary and tuple formats
            if isinstance(snapshot, dict):
                node_features = snapshot['x'].to(device)
                edge_index = snapshot['edge_index'].to(device)
                edge_attr = snapshot['edge_attr'].to(device) if 'edge_attr' in snapshot else None
                node_ids = snapshot['node_ids']
            else:
                # Handle tuples of varying length
                if len(snapshot) >= 4:
                    # Standard format (node_features, edge_index, edge_attr, node_ids)
                    node_features, edge_index, edge_attr, node_ids = snapshot
                    node_features = node_features.to(device)
                    edge_index = edge_index.to(device)
                    if edge_attr is not None:
                        edge_attr = edge_attr.to(device)
                else:
                    raise ValueError(f"Snapshot tuple has incorrect format. Expected at least 4 elements, got {len(snapshot)}")
            
            # Embed node features
            x = self.node_embedding(node_features)
            
            # Embed edge features if provided
            if edge_attr is not None and self.edge_embedding is not None:
                edge_features = self.edge_embedding(edge_attr)
            else:
                edge_features = None
            
            # Apply geometric attention layers with skip connections
            skip_features = x
            
            for i, attn_layer in enumerate(self.geometric_attention_layers):
                if return_attention_weights:
                    x, attn_weights = attn_layer(
                        x, edge_index, edge_features, 
                        return_attention_weights=True
                    )
                    
                    # Store attention weights for first layer
                    if i == 0:
                        attn_weights_seq.append(attn_weights)
                else:
                    x = attn_layer(x, edge_index, edge_features)
                
                # Apply skip connection
                if i == 0:  # First layer
                    if self.skip_layer_norm is not None:
                        x = x + self.skip_layer_norm(skip_features)
                    else:
                        x = x + skip_features
            
            # Apply temporal propagation - each snapshot is processed individually first
            # then temporal relationships are captured when all snapshots are processed
            temporal_input = [x]  # Wrap current snapshot embedding in a list
            
            # Process through temporal propagation
            x_tuple = self.temporal_propagation(temporal_input, [node_ids])
            
            # Unpack the tuple - temporal_propagation returns (output_features, memory_bank)
            if isinstance(x_tuple, tuple):
                # Extract the processed node embeddings
                if isinstance(x_tuple[0], list) and len(x_tuple[0]) > 0:
                    x = x_tuple[0][0]  # Get the first (and only) output tensor
                else:
                    x = x_tuple[0]  # Get the output_features directly
            
            # Store node embeddings for temporal attention
            temporal_attention_inputs.append(x)
            
            # Create mask for this time step - use the first tensor in x if it's a list/sequence
            device = x[0].device if isinstance(x, list) and len(x) > 0 else x.device
            mask = torch.zeros(len(all_node_ids), dtype=torch.bool, device=device)
            for node_id in node_ids:
                mask[node_id_to_idx[node_id]] = True
            
            node_masks_seq.append(mask)
        # Apply temporal attention to aggregate information across time
        # Apply temporal attention to capture asymmetric relationships across time
        attention_mask = None
        if len(node_masks_seq) > 0:
            # Create an attention mask from node masks
            attention_mask = torch.stack(node_masks_seq, dim=0)
        
        if return_attention_weights:
            x, temp_attn_weights = self.temporal_attention(
                temporal_attention_inputs,
                attention_mask=attention_mask,
                return_attention_weights=True
            )
        else:
            x = self.temporal_attention(
                temporal_attention_inputs,
                attention_mask=attention_mask
            )
        
        # Pool node features back to graph-level features correctly based on node masks
        seq_len = len(graph_sequence)
        batch_size = 1  # Default for single graph sequence
        
        # If labels provided, use their shape to determine batch size
        if labels is not None:
            if labels.dim() > 0:
                batch_size = labels.shape[0]
        
        # Get dimensions from the temporal attention output
        if isinstance(x, torch.Tensor):
            hidden_dim = x.shape[-1]
        else:
            # Handle case where x is a list
            hidden_dim = x[0].shape[-1] if isinstance(x, list) and len(x) > 0 else 128
        
        # Create graph features tensor with proper dimensions
        graph_features = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
        
        # Use node masks to properly pool node features for each graph
        for b in range(batch_size):
            for t in range(seq_len):
                if t < len(graph_sequence):
                    # Get node IDs for this graph at this timestep
                    snapshot = graph_sequence[t]
                    if isinstance(snapshot, dict):
                        node_ids = snapshot['node_ids']
                    else:
                        node_ids = snapshot[3]
                    
                    # Get node features for these nodes
                    node_features = []
                    # In a snapshot, the node_ids should correspond to rows in x in order
                    # So we use the position in node_ids list as the index into x
                    for i, node_id in enumerate(node_ids):
                        if isinstance(x, list):
                            # x is a list of tensors (one per timestep)
                            if t < len(x):
                                node_features.append(x[t][i])  # Use position i instead of global idx
                        else:
                            # x is a tensor with all nodes for current snapshot
                            node_features.append(x[i])  # Use position i instead of global idx
                    
                    # Stack and pool node features
                    if node_features:
                        node_tensor = torch.stack(node_features)
                        # Properly handle the mean operation across nodes
                        mean_features = torch.mean(node_tensor, dim=0)
                        
                        # Check dimensions and ensure we have the right shape for assignment
                        if mean_features.dim() > 1:
                            # If mean_features has too many dimensions, take mean along first dimension
                            mean_features = torch.mean(mean_features, dim=0)
                        
                        # Now assign the properly shaped features
                        graph_features[b, t] = mean_features
        
        # Apply classification head to the properly pooled graph representation
        logits = self.classification_head(graph_features)
        
        # Compute loss if labels provided
        if labels is not None:
            # Convert boolean labels to integers if necessary
            if labels.dtype == torch.bool:
                labels = labels.long()
                
            # For multi-class, ensure labels and logits have compatible shapes
            if self.config.output_dim > 1 and labels.dim() == 1:
                # Labels are class indices, use CrossEntropyLoss
                loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                # Use the configured loss function
                loss = self.loss_fn(logits, labels)
        else:
            loss = None
        
        # Convert logits to predictions
        if self.config.output_dim == 1:  # Binary classification
            predictions = torch.sigmoid(logits)
        else:  # Multi-class classification
            predictions = F.softmax(logits, dim=1)
        
        # Create output dictionary
        outputs = {
            'logits': logits,
            'predictions': predictions,
            'loss': loss
        }
        
        # Add attention weights if requested
        if return_attention_weights:
            outputs['geometric_attention_weights'] = attn_weights_seq
            outputs['temporal_attention_weights'] = temp_attn_weights
        
        return outputs
    
    def infer(
        self,
        graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]],
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Inference pass for TAGAN.
        
        Args:
            graph_sequence: List of (node_features, edge_index, edge_attr, node_ids) tuples
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary containing model outputs
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass without computing loss
        with torch.no_grad():
            outputs = self.forward(graph_sequence, return_attention_weights=False)
        
        # Get predictions
        if return_probs:
            # Return class probabilities
            predictions = outputs['predictions']
        else:
            # Return class predictions
            if self.config.output_dim == 1:  # Binary classification
                predictions = (outputs['predictions'] > 0.5).float()
            else:  # Multi-class classification
                predictions = torch.argmax(outputs['predictions'], dim=1)
        
        return {
            'predictions': predictions,
            'logits': outputs['logits']
        }
    
    @torch.no_grad()
    def infer_with_attention(
        self,
        graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]]
    ) -> Dict[str, Any]:
        """
        Inference with attention weights for visualization.
        
        Args:
            graph_sequence: List of (node_features, edge_index, edge_attr, node_ids) tuples
            
        Returns:
            Dictionary containing model outputs and attention weights
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass with attention weights
        outputs = self.forward(graph_sequence, return_attention_weights=True)
        
        return outputs
    
    def encode_graph_sequence(
        self,
        graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]]
    ) -> torch.Tensor:
        """
        Encode a graph sequence into a fixed-size representation.
        
        This method processes the sequence through the encoder part of the model
        and returns the final representation.
        
        Args:
            graph_sequence: List of (node_features, edge_index, edge_attr, node_ids) tuples
            
        Returns:
            Encoded representation of the graph sequence
        """
        # Set model to evaluation mode
        self.eval()
        
        # Process the sequence
        with torch.no_grad():
            # Process each graph in the sequence
            node_embeddings_seq = []
            node_masks_seq = []
            
            # Track all unique node IDs
            all_node_ids = set()
            for _, _, _, node_ids in graph_sequence:
                all_node_ids.update(node_ids)
            
            # Convert to sorted list for consistent indexing
            all_node_ids = sorted(list(all_node_ids))
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            
            # Process each time step
            temporal_attention_inputs = []
            
            for node_features, edge_index, edge_attr, node_ids in graph_sequence:
                # Embed node features
                x = self.node_embedding(node_features)
                
                # Embed edge features if provided
                if edge_attr is not None and self.edge_embedding is not None:
                    edge_features = self.edge_embedding(edge_attr)
                else:
                    edge_features = None
                
                # Apply geometric attention layers with skip connections
                skip_features = x
                
                for i, attn_layer in enumerate(self.geometric_attention_layers):
                    x = attn_layer(x, edge_index, edge_features)
                    
                    # Apply skip connection
                    if i == 0:  # First layer
                        if self.skip_layer_norm is not None:
                            x = x + self.skip_layer_norm(skip_features)
                        else:
                            x = x + skip_features
                
                # Apply temporal propagation
                x = self.temporal_propagation(x, node_ids)
                
                # Store node embeddings for temporal attention
                temporal_attention_inputs.append(x)
                
                # Create mask for this time step
                mask = torch.zeros(len(all_node_ids), dtype=torch.bool, device=x.device)
                for node_id in node_ids:
                    mask[node_id_to_idx[node_id]] = True
                
                node_masks_seq.append(mask)
            
            # Apply temporal attention to aggregate information across time
            encoded_representation = self.temporal_attention(
                temporal_attention_inputs,
                node_masks_seq
            )
        
        return encoded_representation
    
    def reset_temporal_state(self):
        """Reset the temporal state of the model."""
        self.temporal_propagation.reset_memory()
    
    def save_temporal_state(self, filepath: str):
        """
        Save the temporal state of the model.
        
        Args:
            filepath: Path to save the temporal state
        """
        self.temporal_propagation.save_memory_state(filepath)
    
    def load_temporal_state(self, filepath: str):
        """
        Load the temporal state of the model.
        
        Args:
            filepath: Path to load the temporal state from
        """
        self.temporal_propagation.load_memory_state(filepath)
    
    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], TAGANConfig]) -> 'TAGAN':
        """
        Create a TAGAN model from a configuration.
        
        Args:
            config: Configuration dictionary or TAGANConfig object
            
        Returns:
            TAGAN model
        """
        if isinstance(config, dict):
            config = TAGANConfig.from_dict(config)
        
        return cls(config)
    
    def extra_repr(self) -> str:
        """Return a string representation of the model configuration."""
        return f"config={self.config}"


class TAGANEncoder(nn.Module):
    """
    Encoder part of the TAGAN model.
    
    This module extracts node representations from graph sequences.
    
    Attributes:
        config (TAGANConfig): Configuration parameters
        node_embedding (nn.Linear): Node feature embedding
        edge_embedding (nn.Linear): Edge feature embedding (if used)
        geometric_attention_layers (nn.ModuleList): List of geometric attention layers
        temporal_propagation (TemporalPropagation): Temporal propagation module
        temporal_attention (TemporalAttention): Temporal attention module
    """
    
    def __init__(
        self,
        config: TAGANConfig
    ):
        """
        Initialize the TAGAN encoder.
        
        Args:
            config: Configuration parameters
        """
        super(TAGANEncoder, self).__init__()
        
        self.config = config
        
        # Input embeddings
        self.node_embedding = nn.Linear(config.node_feature_dim, config.hidden_dim)
        
        if config.edge_feature_dim > 0:
            self.edge_embedding = nn.Linear(config.edge_feature_dim, config.hidden_dim)
        else:
            self.edge_embedding = None
        
        # Geometric attention layers
        self.geometric_attention_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            self.geometric_attention_layers.append(
                TAGANGraphAttention(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    distance_metric="scaled_dot_product" if config.learnable_distance else "euclidean",
                    use_layer_norm=config.use_layer_norm,
                    learnable_distance=config.learnable_distance
                )
            )
        
        # Temporal propagation
        self.temporal_propagation = TemporalPropagation(
            hidden_dim=config.hidden_dim,
            input_dim=config.hidden_dim,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
            decay_factor=config.memory_decay_factor,
            gru_bias=config.gru_bias,
            device=torch.device(config.device)
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=config.hidden_dim,
            attention_dim=config.temporal_attention_dim,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
            temporal_bias=True,
            num_heads=config.num_heads
        )
        
        # Skip connections and layer normalization
        if config.use_layer_norm:
            self.skip_layer_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.skip_layer_norm = None
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize input embeddings
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)
        
        if self.edge_embedding is not None:
            nn.init.xavier_uniform_(self.edge_embedding.weight)
            nn.init.zeros_(self.edge_embedding.bias)
    
    def forward(
        self,
        graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]],
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass for TAGAN encoder.
        
        Args:
            graph_sequence: List of (node_features, edge_index, edge_attr, node_ids) tuples
            return_attention_weights: Whether to return attention weights
            
        Returns:
            - If return_attention_weights is False: Encoded representation
            - If return_attention_weights is True: Tuple of (encoded representation, attention weights)
        """
        # Process each graph in the sequence
        node_embeddings_seq = []
        node_masks_seq = []
        
        # Track all unique node IDs
        all_node_ids = set()
        for _, _, _, node_ids in graph_sequence:
            all_node_ids.update(node_ids)
        
        # Convert to sorted list for consistent indexing
        all_node_ids = sorted(list(all_node_ids))
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        
        # Process each time step
        temporal_attention_inputs = []
        attn_weights_seq = []
        
        # Get the device of the model
        device = next(self.parameters()).device
        
        for node_features, edge_index, edge_attr, node_ids in graph_sequence:
            # Move input tensors to the correct device
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)
                
            # Embed node features
            x = self.node_embedding(node_features)
            
            # Embed edge features if provided
            if edge_attr is not None and self.edge_embedding is not None:
                edge_features = self.edge_embedding(edge_attr)
            else:
                edge_features = None
            
            # Apply geometric attention layers with skip connections
            skip_features = x
            
            for i, attn_layer in enumerate(self.geometric_attention_layers):
                if return_attention_weights:
                    x, attn_weights = attn_layer(
                        x, edge_index, edge_features, 
                        return_attention_weights=True
                    )
                    
                    # Store attention weights for first layer
                    if i == 0:
                        attn_weights_seq.append(attn_weights)
                else:
                    x = attn_layer(x, edge_index, edge_features)
                
                # Apply skip connection
                if i == 0:  # First layer
                    if self.skip_layer_norm is not None:
                        x = x + self.skip_layer_norm(skip_features)
                    else:
                        x = x + skip_features
            
            # Apply temporal propagation
            x = self.temporal_propagation(x, node_ids)
            
            # Store node embeddings for temporal attention
            temporal_attention_inputs.append(x)
            
            # Create mask for this time step
            mask = torch.zeros(len(all_node_ids), dtype=torch.bool, device=x.device)
            for node_id in node_ids:
                mask[node_id_to_idx[node_id]] = True
            
            node_masks_seq.append(mask)
        
        # Apply temporal attention to aggregate information across time
        if return_attention_weights:
            encoded, temp_attn_weights = self.temporal_attention(
                temporal_attention_inputs,
                node_masks_seq,
                return_attention_weights=True
            )
            
            attention_weights = {
                'geometric_attention_weights': attn_weights_seq,
                'temporal_attention_weights': temp_attn_weights
            }
            
            return encoded, attention_weights
        else:
            encoded = self.temporal_attention(
                temporal_attention_inputs,
                node_masks_seq
            )
            
            return encoded
    
    def reset_temporal_state(self):
        """Reset the temporal state of the encoder."""
        self.temporal_propagation.reset_memory()
    
    def extra_repr(self) -> str:
        """Return a string representation of the encoder configuration."""
        return f"config={self.config}"


class TAGANDecoder(nn.Module):
    """
    Decoder part of the TAGAN model.
    
    This module takes encoded representations and makes predictions.
    
    Attributes:
        config (TAGANConfig): Configuration parameters
        classification_head (ClassificationHead): Classification head
        loss_fn (TAGANLoss): Loss function
    """
    
    def __init__(
        self,
        config: TAGANConfig
    ):
        """
        Initialize the TAGAN decoder.
        
        Args:
            config: Configuration parameters
        """
        super(TAGANDecoder, self).__init__()
        
        self.config = config
        
        # Classification head
        self.classification_head = ClassificationHead(
            input_dim=config.hidden_dim,
            output_dim=config.output_dim,
            hidden_dims=[config.hidden_dim * 2, config.hidden_dim],
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm
        )
        
        # Loss function
        self.loss_fn = TAGANLoss(
            loss_type=config.loss_type,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )
    
    def forward(
        self,
        encoded_representation: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for TAGAN decoder.
        
        Args:
            encoded_representation: Encoded representation from encoder
            labels: Optional ground truth labels for training
            
        Returns:
            Dictionary containing model outputs
        """
        # Apply classification head
        logits = self.classification_head(encoded_representation)
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        
        # Convert logits to predictions
        if self.config.output_dim == 1:  # Binary classification
            predictions = torch.sigmoid(logits)
        else:  # Multi-class classification
            predictions = F.softmax(logits, dim=1)
        
        # Create output dictionary
        outputs = {
            'logits': logits,
            'predictions': predictions,
            'loss': loss
        }
        
        return outputs
    
    def extra_repr(self) -> str:
        """Return a string representation of the decoder configuration."""
        return f"config={self.config}"


class TAGANEncoderDecoder(nn.Module):
    """
    Complete TAGAN model with separate encoder and decoder.
    
    This model combines the encoder and decoder components.
    
    Attributes:
        encoder (TAGANEncoder): Encoder component
        decoder (TAGANDecoder): Decoder component
        config (TAGANConfig): Configuration parameters
    """
    
    def __init__(
        self,
        config: TAGANConfig
    ):
        """
        Initialize the TAGAN encoder-decoder model.
        
        Args:
            config: Configuration parameters
        """
        super(TAGANEncoderDecoder, self).__init__()
        
        self.config = config
        
        # Create encoder and decoder
        self.encoder = TAGANEncoder(config)
        self.decoder = TAGANDecoder(config)
    
    def forward(
        self,
        graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]],
        labels: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for TAGAN encoder-decoder.
        
        Args:
            graph_sequence: List of (node_features, edge_index, edge_attr, node_ids) tuples
            labels: Optional ground truth labels for training
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode input sequence
        if return_attention_weights:
            encoded, attention_weights = self.encoder(
                graph_sequence, 
                return_attention_weights=True
            )
        else:
            encoded = self.encoder(graph_sequence)
        
        # Decode and get predictions
        outputs = self.decoder(encoded, labels)
        
        # Add attention weights if requested
        if return_attention_weights:
            outputs.update(attention_weights)
        
        return outputs
    
    def infer(
        self,
        graph_sequence: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]],
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Inference pass for TAGAN.
        
        Args:
            graph_sequence: List of (node_features, edge_index, edge_attr, node_ids) tuples
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary containing model outputs
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass without computing loss
        with torch.no_grad():
            outputs = self.forward(graph_sequence, return_attention_weights=False)
        
        # Get predictions
        if return_probs:
            # Return class probabilities
            predictions = outputs['predictions']
        else:
            # Return class predictions
            if self.config.output_dim == 1:  # Binary classification
                predictions = (outputs['predictions'] > 0.5).float()
            else:  # Multi-class classification
                predictions = torch.argmax(outputs['predictions'], dim=1)
        
        return {
            'predictions': predictions,
            'logits': outputs['logits']
        }
    
    def reset_temporal_state(self):
        """Reset the temporal state of the model."""
        self.encoder.reset_temporal_state()
    
    def extra_repr(self) -> str:
        """Return a string representation of the model configuration."""
        return f"config={self.config}"
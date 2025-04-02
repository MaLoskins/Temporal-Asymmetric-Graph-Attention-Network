"""
Graph attention adapter for TAGAN.

This module provides an adapter between graph data and the geometric attention mechanism,
handling dimension transformations and graph-specific operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union

from .geometric_attention import GeometricAttention


class TAGANGraphAttention(nn.Module):
    """
    Adapter class that connects graph data with geometric attention.
    
    This class handles the dimension adaptation between graph data format
    and the format expected by the geometric attention module.
    
    Attributes:
        geometric_attention (GeometricAttention): The underlying geometric attention module
        hidden_dim (int): Hidden dimension size
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        distance_metric: str = "scaled_dot_product",
        use_layer_norm: bool = True,
        learnable_distance: bool = False
    ):
        """
        Initialize the graph attention adapter.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            distance_metric: Distance metric to use (default: "scaled_dot_product")
            use_layer_norm: Whether to use layer normalization (default: True)
            learnable_distance: Whether to use learnable distance parameters (default: False)
        """
        super(TAGANGraphAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Create geometric attention module
        self.geometric_attention = GeometricAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            distance_metric=distance_metric,
            use_layer_norm=use_layer_norm,
            learnable_distance=learnable_distance
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for graph attention.
        
        Adapts graph data format to the format expected by geometric attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim] (default: None)
            return_attention_weights: Whether to return attention weights (default: False)
            
        Returns:
            Updated node features [num_nodes, hidden_dim] and optionally attention weights
        """
        # Get device
        device = x.device
        
        # Handle the dimension adaptation
        # For graph data, we treat it as a batch_size=1, seq_len=num_nodes
        batch_size = 1
        num_nodes = x.size(0)
        
        # Reshape x to [batch_size, seq_len, hidden_dim]
        x_reshaped = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # Create attention mask based on edge_index
        # This turns the edge_index into a dense adjacency matrix to use as an attention mask
        attention_mask = None
        if edge_index is not None:
            # Convert edge_index to adjacency matrix
            adj = torch.zeros(num_nodes, num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1
            
            # Add self-loops (optional, but usually helpful)
            adj = adj + torch.eye(num_nodes, device=device)
            
            # Convert to attention mask format [batch_size, seq_len, seq_len]
            attention_mask = adj.unsqueeze(0)  # [1, num_nodes, num_nodes]
        
        # Handle edge features if provided
        geometric_bias = None
        if edge_attr is not None:
            # Here we could compute a geometric bias based on edge features
            # For now, we'll leave this as None
            pass
        
        # Apply geometric attention
        if return_attention_weights:
            # For returning attention weights, we'd need to modify the geometric attention
            # class to return them. For now, this is a placeholder.
            out = self.geometric_attention(x_reshaped, attention_mask, geometric_bias)
            
            # Extract attention weights from the attention module (if implemented to return them)
            attention_weights = {"node_attention": None}  # placeholder
            
            # Reshape output back to graph format
            out = out.squeeze(0)  # [num_nodes, hidden_dim]
            
            return out, attention_weights
        else:
            out = self.geometric_attention(x_reshaped, attention_mask, geometric_bias)
            
            # Reshape output back to graph format
            out = out.squeeze(0)  # [num_nodes, hidden_dim]
            
            return out
    
    def extra_repr(self) -> str:
        """Return a string representation of the adapter configuration."""
        return f"hidden_dim={self.hidden_dim}"
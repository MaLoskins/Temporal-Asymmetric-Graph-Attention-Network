"""
Geometric attention mechanisms for TAGAN.

This module provides attention layers that operate on spatial/geometric data,
with custom distance metrics to capture geometric relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import math


class DistanceMetric:
    """
    Distance metrics for geometric attention.
    
    This class provides various distance metrics to measure similarity
    between vectors in geometric space.
    """
    
    @staticmethod
    def euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Euclidean distances [batch_size, ...]
        """
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
    
    @staticmethod
    def squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distance between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Squared Euclidean distances [batch_size, ...]
        """
        return torch.sum((x - y) ** 2, dim=-1)
    
    @staticmethod
    def manhattan(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Manhattan (L1) distance between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Manhattan distances [batch_size, ...]
        """
        return torch.sum(torch.abs(x - y), dim=-1)
    
    @staticmethod
    def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Cosine similarities [batch_size, ...]
        """
        # Compute cosine similarity
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        
        # Add epsilon to avoid division by zero
        x_norm = torch.where(x_norm == 0, torch.ones_like(x_norm) * 1e-8, x_norm)
        y_norm = torch.where(y_norm == 0, torch.ones_like(y_norm) * 1e-8, y_norm)
        
        similarity = torch.sum(x * y, dim=-1) / (x_norm * y_norm).squeeze(-1)
        
        # Clip to [-1, 1] to handle floating point errors
        similarity = torch.clamp(similarity, -1.0, 1.0)
        
        return similarity
    
    @staticmethod
    def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine distance (1 - cosine similarity) between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Cosine distances [batch_size, ...]
        """
        return 1.0 - DistanceMetric.cosine_similarity(x, y)
    
    @staticmethod
    def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Dot products [batch_size, ...]
        """
        return torch.sum(x * y, dim=-1)
    
    @staticmethod
    def scaled_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot product between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            
        Returns:
            Scaled dot products [batch_size, ...]
        """
        dim = x.size(-1)
        return torch.sum(x * y, dim=-1) / math.sqrt(dim)
    
    @staticmethod
    def mahalanobis(x: torch.Tensor, y: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            cov_inv: Inverse covariance matrix [dim, dim]
            
        Returns:
            Mahalanobis distances [batch_size, ...]
        """
        diff = x - y  # [batch_size, ..., dim]
        
        # Reshape diff to 2D for matrix multiplication
        diff_shape = diff.shape
        diff_2d = diff.view(-1, diff_shape[-1])  # [batch_size * ..., dim]
        
        # Compute Mahalanobis distance
        # (diff_2d @ cov_inv) @ diff_2d.T
        mahal = torch.sum(diff_2d @ cov_inv * diff_2d, dim=-1)
        
        # Reshape back to original shape (except last dim)
        mahal = mahal.view(*diff_shape[:-1])
        
        return torch.sqrt(mahal + 1e-8)
    
    @staticmethod
    def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Compute Gaussian kernel between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            sigma: Kernel bandwidth (default: 1.0)
            
        Returns:
            Kernel values [batch_size, ...]
        """
        squared_dist = DistanceMetric.squared_euclidean(x, y)
        return torch.exp(-squared_dist / (2 * sigma ** 2))
    
    @staticmethod
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """
        Compute RBF kernel between batches of vectors.
        
        Args:
            x: First tensor [batch_size, ..., dim]
            y: Second tensor [batch_size, ..., dim]
            gamma: Kernel parameter (default: 1.0)
            
        Returns:
            Kernel values [batch_size, ...]
        """
        squared_dist = DistanceMetric.squared_euclidean(x, y)
        return torch.exp(-gamma * squared_dist)
    
    @staticmethod
    def get_metric(metric_name: str) -> Callable:
        """
        Get distance metric function by name.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Metric function
        """
        if metric_name == "euclidean":
            return DistanceMetric.euclidean
        elif metric_name == "squared_euclidean":
            return DistanceMetric.squared_euclidean
        elif metric_name == "manhattan":
            return DistanceMetric.manhattan
        elif metric_name == "cosine_similarity":
            return DistanceMetric.cosine_similarity
        elif metric_name == "cosine_distance":
            return DistanceMetric.cosine_distance
        elif metric_name == "dot_product":
            return DistanceMetric.dot_product
        elif metric_name == "scaled_dot_product":
            return DistanceMetric.scaled_dot_product
        elif metric_name == "gaussian_kernel":
            return DistanceMetric.gaussian_kernel
        elif metric_name == "rbf_kernel":
            return DistanceMetric.rbf_kernel
        else:
            raise ValueError(f"Unknown distance metric: {metric_name}")


class GeometricAttention(nn.Module):
    """
    Base class for geometric attention mechanisms.
    
    This module provides attention operations based on
    geometric relationships between entities.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        distance_metric (str): Distance metric to use
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
        Initialize the geometric attention module.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            distance_metric: Distance metric to use (default: "scaled_dot_product")
            use_layer_norm: Whether to use layer normalization (default: True)
            learnable_distance: Whether to use learnable distance parameters (default: False)
        """
        super(GeometricAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.distance_metric = distance_metric
        self.use_layer_norm = use_layer_norm
        self.learnable_distance = learnable_distance
        
        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for queries, keys, and values
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Learnable distance parameters
        if learnable_distance:
            if distance_metric in ["gaussian_kernel", "rbf_kernel"]:
                # Learnable bandwidth/gamma parameter per head
                self.distance_param = nn.Parameter(torch.ones(num_heads))
            elif distance_metric == "mahalanobis":
                # Learnable inverse covariance matrices for each head
                # We use a low-rank approximation for efficiency
                rank = min(16, hidden_dim // 4)  # Low-rank approximation
                self.cov_factors = nn.Parameter(
                    torch.zeros(num_heads, rank, self.head_dim)
                )
                nn.init.xavier_uniform_(self.cov_factors)
        
        # Distance metric function
        self.distance_fn = DistanceMetric.get_metric(distance_metric)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        nn.init.zeros_(self.q_linear.bias)
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.v_linear.bias)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize learnable distance parameters
        if self.learnable_distance:
            if self.distance_metric in ["gaussian_kernel", "rbf_kernel"]:
                # Initialize to reasonable defaults
                if self.distance_metric == "gaussian_kernel":
                    nn.init.constant_(self.distance_param, 1.0)  # sigma = 1.0
                else:  # rbf_kernel
                    nn.init.constant_(self.distance_param, 0.1)  # gamma = 0.1
    
    def _get_attention_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights based on distance metric.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optional mask [batch_size, seq_len, seq_len] (default: None)
            
        Returns:
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, num_heads, seq_len, head_dim = q.size()
        
        if self.distance_metric == "scaled_dot_product":
            # Standard scaled dot-product attention
            # [batch_size, num_heads, seq_len, seq_len]
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        elif self.distance_metric in ["dot_product", "cosine_similarity"]:
            # Pre-defined similarity metrics
            # Compute for each head separately
            attention_scores = torch.zeros(
                batch_size, num_heads, seq_len, seq_len,
                device=q.device
            )
            
            for h in range(num_heads):
                for i in range(seq_len):
                    # [batch_size, head_dim]
                    q_i = q[:, h, i]
                    
                    # [batch_size, seq_len, head_dim]
                    k_all = k[:, h]
                    
                    # [batch_size, seq_len]
                    scores = self.distance_fn(q_i.unsqueeze(1), k_all)
                    
                    # Store scores
                    attention_scores[:, h, i] = scores
        
        elif self.distance_metric in ["euclidean", "squared_euclidean", "manhattan", "cosine_distance"]:
            # Distance-based metrics (convert to similarity)
            # Compute for each head separately
            attention_scores = torch.zeros(
                batch_size, num_heads, seq_len, seq_len,
                device=q.device
            )
            
            for h in range(num_heads):
                for i in range(seq_len):
                    # [batch_size, head_dim]
                    q_i = q[:, h, i]
                    
                    # [batch_size, seq_len, head_dim]
                    k_all = k[:, h]
                    
                    # [batch_size, seq_len]
                    distances = self.distance_fn(q_i.unsqueeze(1), k_all)
                    
                    # Convert distances to similarities (negative distances)
                    scores = -distances
                    
                    # Store scores
                    attention_scores[:, h, i] = scores
        
        elif self.distance_metric in ["gaussian_kernel", "rbf_kernel"]:
            # Kernel-based similarity
            # Compute for each head separately
            attention_scores = torch.zeros(
                batch_size, num_heads, seq_len, seq_len,
                device=q.device
            )
            
            for h in range(num_heads):
                # Get kernel parameter for this head
                if self.learnable_distance:
                    if self.distance_metric == "gaussian_kernel":
                        sigma = self.distance_param[h]
                        kernel_fn = lambda x, y: DistanceMetric.gaussian_kernel(x, y, sigma)
                    else:  # rbf_kernel
                        gamma = self.distance_param[h]
                        kernel_fn = lambda x, y: DistanceMetric.rbf_kernel(x, y, gamma)
                else:
                    kernel_fn = self.distance_fn
                
                for i in range(seq_len):
                    # [batch_size, head_dim]
                    q_i = q[:, h, i]
                    
                    # [batch_size, seq_len, head_dim]
                    k_all = k[:, h]
                    
                    # [batch_size, seq_len]
                    similarities = kernel_fn(q_i.unsqueeze(1), k_all)
                    
                    # Store scores
                    attention_scores[:, h, i] = similarities
        
        elif self.distance_metric == "mahalanobis":
            # Mahalanobis distance (with learnable covariance)
            # Compute for each head separately
            attention_scores = torch.zeros(
                batch_size, num_heads, seq_len, seq_len,
                device=q.device
            )
            
            for h in range(num_heads):
                # Compute inverse covariance matrix for this head
                if self.learnable_distance:
                    # Using low-rank approximation: factors @ factors.T
                    # This ensures the matrix is positive semi-definite
                    factors = self.cov_factors[h]  # [rank, head_dim]
                    cov_inv = factors.T @ factors  # [head_dim, head_dim]
                else:
                    # Default: identity matrix (equivalent to scaled Euclidean)
                    cov_inv = torch.eye(head_dim, device=q.device)
                
                for i in range(seq_len):
                    # [batch_size, head_dim]
                    q_i = q[:, h, i]
                    
                    # [batch_size, seq_len, head_dim]
                    k_all = k[:, h]
                    
                    # [batch_size, seq_len]
                    distances = DistanceMetric.mahalanobis(q_i.unsqueeze(1), k_all, cov_inv)
                    
                    # Convert distances to similarities (negative distances)
                    scores = -distances
                    
                    # Store scores
                    attention_scores[:, h, i] = scores
        
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
            expanded_mask = mask.unsqueeze(1)
            
            # Apply mask (set masked positions to negative infinity)
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.attn_dropout(attention_weights)
        
        return attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        geometric_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for geometric attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional mask [batch_size, seq_len, seq_len] (default: None)
            geometric_bias: Optional geometric bias [batch_size, seq_len, seq_len] (default: None)
            
        Returns:
            Attended tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Store original for residual connection
        identity = x
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            x = self.layer_norm1(x)
        
        # Compute queries, keys, and values
        # [batch_size, seq_len, hidden_dim]
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        # [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention weights based on distance metric
        # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = self._get_attention_weights(q, k, attention_mask)
        
        # Add geometric bias if provided
        if geometric_bias is not None:
            # Expand bias for multi-head attention
            # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
            geometric_bias = geometric_bias.unsqueeze(1)
            
            # Add to attention weights and re-normalize
            attention_weights = attention_weights + geometric_bias
            attention_weights = F.softmax(attention_weights, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, v)
        
        # Transpose back and reshape
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, hidden_dim]
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        # Apply output projection
        context = self.output_proj(context)
        
        # Apply dropout
        context = self.output_dropout(context)
        
        # Apply residual connection
        out = context + identity
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            out = self.layer_norm2(out)
        
        return out
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'num_heads={self.num_heads}, '
                f'distance_metric={self.distance_metric}, '
                f'learnable_distance={self.learnable_distance}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'dropout={self.dropout_prob}')


class GeometricDistanceModule(nn.Module):
    """
    Module for computing geometric distances between entities.
    
    This module computes pairwise distances between entities based on
    their features, coordinates, or other attributes.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        distance_metric (str): Distance metric to use
    """
    
    def __init__(
        self,
        hidden_dim: int,
        distance_metric: str = "euclidean",
        trainable: bool = True,
        feature_based: bool = True,
        coordinate_dim: int = 2,
        temperature: float = 1.0
    ):
        """
        Initialize the geometric distance module.
        
        Args:
            hidden_dim: Hidden dimension size
            distance_metric: Distance metric to use (default: "euclidean")
            trainable: Whether distances are trainable (default: True)
            feature_based: Whether to use feature-based or coordinate-based distances (default: True)
            coordinate_dim: Dimension of coordinates (default: 2)
            temperature: Temperature parameter for scaling distances (default: 1.0)
        """
        super(GeometricDistanceModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.distance_metric = distance_metric
        self.trainable = trainable
        self.feature_based = feature_based
        self.coordinate_dim = coordinate_dim
        self.temperature = temperature
        
        # Get distance function
        self.distance_fn = DistanceMetric.get_metric(distance_metric)
        
        # Feature projection for distance computation
        if feature_based and trainable:
            self.distance_projection = nn.Linear(hidden_dim, hidden_dim // 2)
        elif not feature_based and trainable:
            # Project to coordinates
            self.coordinate_projection = nn.Linear(hidden_dim, coordinate_dim)
        
        # Learnable temperature
        if trainable:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        if self.feature_based and self.trainable:
            nn.init.xavier_uniform_(self.distance_projection.weight)
            nn.init.zeros_(self.distance_projection.bias)
        elif not self.feature_based and self.trainable:
            nn.init.xavier_uniform_(self.coordinate_projection.weight)
            nn.init.zeros_(self.coordinate_projection.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute geometric distances between entities.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            coordinates: Optional coordinates [batch_size, seq_len, coordinate_dim] (default: None)
            
        Returns:
            Distance matrix [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        if self.feature_based:
            # Use features for distance computation
            if self.trainable:
                # Project features
                features = self.distance_projection(x)
            else:
                features = x
            
            # Compute pairwise distances
            distances = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            
            for b in range(batch_size):
                for i in range(seq_len):
                    # Get features for entity i
                    # [hidden_dim] or [hidden_dim // 2]
                    feat_i = features[b, i]
                    
                    # Get features for all entities
                    # [seq_len, hidden_dim] or [seq_len, hidden_dim // 2]
                    feat_all = features[b]
                    
                    # Compute distances
                    # [seq_len]
                    entity_dists = self.distance_fn(feat_i.unsqueeze(0), feat_all)
                    
                    # Store distances
                    distances[b, i] = entity_dists
        else:
            # Use coordinates for distance computation
            if coordinates is None:
                if self.trainable:
                    # Project features to coordinates
                    coordinates = self.coordinate_projection(x)
                else:
                    raise ValueError("Coordinates must be provided when feature_based=False and trainable=False")
            
            # Compute pairwise distances
            distances = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            
            for b in range(batch_size):
                for i in range(seq_len):
                    # Get coordinates for entity i
                    # [coordinate_dim]
                    coord_i = coordinates[b, i]
                    
                    # Get coordinates for all entities
                    # [seq_len, coordinate_dim]
                    coord_all = coordinates[b]
                    
                    # Compute distances
                    # [seq_len]
                    entity_dists = self.distance_fn(coord_i.unsqueeze(0), coord_all)
                    
                    # Store distances
                    distances[b, i] = entity_dists
        
        # Scale distances by temperature
        if self.trainable:
            temperature = torch.exp(self.log_temperature)
        else:
            temperature = self.temperature
        
        scaled_distances = distances / temperature
        
        return scaled_distances
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'distance_metric={self.distance_metric}, '
                f'feature_based={self.feature_based}, '
                f'trainable={self.trainable}, '
                f'coordinate_dim={self.coordinate_dim}, '
                f'temperature={self.temperature}')


class SpatialPositionEncoding(nn.Module):
    """
    Spatial position encoding for geometric attention.
    
    This module provides positional encodings for spatial coordinates,
    supporting both absolute and relative positions.
    
    Attributes:
        d_model (int): Model dimension for encoding
        encoding_type (str): Type of encoding to use
    """
    
    def __init__(
        self,
        d_model: int,
        encoding_type: str = 'sinusoidal',
        max_spatial_distance: float = 100.0,
        num_bases: int = 16,
        learnable: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize the spatial position encoding module.
        
        Args:
            d_model: Model dimension for encoding
            encoding_type: Type of encoding (default: 'sinusoidal')
                         Options: 'sinusoidal', 'linear', 'learned', 'basis'
            max_spatial_distance: Maximum spatial distance (default: 100.0)
            num_bases: Number of basis functions for 'basis' encoding (default: 16)
            learnable: Whether encodings are learnable (default: False)
            dropout: Dropout probability (default: 0.1)
        """
        super(SpatialPositionEncoding, self).__init__()
        
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.max_spatial_distance = max_spatial_distance
        self.num_bases = num_bases
        self.learnable = learnable
        self.dropout = nn.Dropout(p=dropout)
        
        # Create encoding based on specified type
        if encoding_type == 'sinusoidal':
            # Create sinusoidal encoding for spatial positions
            self.register_buffer('freq_bands', torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            if learnable:
                self.freq_bands = nn.Parameter(self.freq_bands)
        
        elif encoding_type == 'linear':
            # Linear encoding of space
            pass  # No pre-computed values needed
        
        elif encoding_type == 'learned':
            # Fully learned encoding
            spatial_range = torch.linspace(0, max_spatial_distance, 1000)
            
            # Initialize with sinusoidal encoding
            pos_enc = torch.zeros(len(spatial_range), d_model)
            
            position = spatial_range.unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            
            # Make learnable
            self.pos_enc = nn.Parameter(pos_enc)
            self.spatial_range = spatial_range
        
        elif encoding_type == 'basis':
            # RBF-like basis functions
            self.basis_mu = nn.Parameter(torch.linspace(0, 1, num_bases))
            self.basis_sigma = nn.Parameter(torch.ones(num_bases) * 0.1)
            self.basis_proj = nn.Linear(num_bases, d_model)
            
            # Initialize
            nn.init.xavier_uniform_(self.basis_proj.weight)
            nn.init.zeros_(self.basis_proj.bias)
    
    def _get_sinusoidal_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal encoding for positions.
        
        Args:
            positions: Positions to encode [batch_size, ...]
            
        Returns:
            Encoded positions [batch_size, ..., d_model]
        """
        # Normalize positions to [0, 1]
        pos_norm = positions / self.max_spatial_distance
        pos_norm = torch.clamp(pos_norm, 0.0, 1.0)
        
        # Reshape for encoding
        orig_shape = pos_norm.shape
        pos_flat = pos_norm.view(-1, 1)  # [batch_size * ..., 1]
        
        # Apply sinusoidal encoding
        # [batch_size * ..., d_model]
        enc = torch.zeros(pos_flat.size(0), self.d_model, device=positions.device)
        
        enc[:, 0::2] = torch.sin(pos_flat * torch.exp(self.freq_bands).unsqueeze(0))
        enc[:, 1::2] = torch.cos(pos_flat * torch.exp(self.freq_bands).unsqueeze(0))
        
        # Reshape back to original shape
        # [batch_size, ..., d_model]
        enc = enc.view(*orig_shape, self.d_model)
        
        return enc
    
    def _get_linear_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute linear encoding for positions.
        
        Args:
            positions: Positions to encode [batch_size, ...]
            
        Returns:
            Encoded positions [batch_size, ..., d_model]
        """
        # Normalize positions to [0, 1]
        pos_norm = positions / self.max_spatial_distance
        pos_norm = torch.clamp(pos_norm, 0.0, 1.0)
        
        # Reshape for encoding
        orig_shape = pos_norm.shape
        
        # Repeat across dimension to get [batch_size, ..., d_model]
        pos_expanded = pos_norm.unsqueeze(-1).expand(*orig_shape, self.d_model)
        
        return pos_expanded
    
    def _get_learned_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get learned encoding for positions.
        
        Args:
            positions: Positions to encode [batch_size, ...]
            
        Returns:
            Encoded positions [batch_size, ..., d_model]
        """
        # Normalize positions to [0, max_spatial_distance]
        pos_norm = torch.clamp(positions, 0.0, self.max_spatial_distance)
        
        # Reshape for interpolation
        orig_shape = pos_norm.shape
        pos_flat = pos_norm.view(-1)  # [batch_size * ...]
        
        # Interpolate from learned table
        # Convert to indices in spatial_range
        pos_idx = pos_flat / self.max_spatial_distance * (len(self.spatial_range) - 1)
        
        # Get integer indices for interpolation
        idx_low = torch.floor(pos_idx).long()
        idx_high = torch.ceil(pos_idx).long()
        
        # Ensure within bounds
        idx_low = torch.clamp(idx_low, 0, len(self.spatial_range) - 1)
        idx_high = torch.clamp(idx_high, 0, len(self.spatial_range) - 1)
        
        # Get weights for interpolation
        weight_high = pos_idx - idx_low.float()
        weight_low = 1.0 - weight_high
        
        # Get encodings for low and high indices
        enc_low = self.pos_enc[idx_low]  # [batch_size * ..., d_model]
        enc_high = self.pos_enc[idx_high]  # [batch_size * ..., d_model]
        
        # Interpolate
        enc = weight_low.unsqueeze(-1) * enc_low + weight_high.unsqueeze(-1) * enc_high
        
        # Reshape back to original shape
        # [batch_size, ..., d_model]
        enc = enc.view(*orig_shape, self.d_model)
        
        return enc
    
    def _get_basis_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute basis function encoding for positions.
        
        Args:
            positions: Positions to encode [batch_size, ...]
            
        Returns:
            Encoded positions [batch_size, ..., d_model]
        """
        # Normalize positions to [0, 1]
        pos_norm = positions / self.max_spatial_distance
        pos_norm = torch.clamp(pos_norm, 0.0, 1.0)
        
        # Reshape for encoding
        orig_shape = pos_norm.shape
        pos_flat = pos_norm.view(-1, 1)  # [batch_size * ..., 1]
        
        # Compute RBF values
        # [batch_size * ..., num_bases]
        basis_values = torch.exp(
            -((pos_flat - self.basis_mu) ** 2) / (2 * self.basis_sigma ** 2)
        )
        
        # Project to dimension
        # [batch_size * ..., d_model]
        enc = self.basis_proj(basis_values)
        
        # Reshape back to original shape
        # [batch_size, ..., d_model]
        enc = enc.view(*orig_shape, self.d_model)
        
        return enc
    
    def forward(
        self,
        positions: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute positional encoding for spatial positions.
        
        Args:
            positions: Spatial positions to encode [batch_size, ...]
            x: Optional tensor to add encoding to [batch_size, ..., d_model] (default: None)
            
        Returns:
            Positional encoding or input with encoding added [batch_size, ..., d_model]
        """
        # Compute encoding based on type
        if self.encoding_type == 'sinusoidal':
            encoding = self._get_sinusoidal_encoding(positions)
        elif self.encoding_type == 'linear':
            encoding = self._get_linear_encoding(positions)
        elif self.encoding_type == 'learned':
            encoding = self._get_learned_encoding(positions)
        elif self.encoding_type == 'basis':
            encoding = self._get_basis_encoding(positions)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Apply dropout
        encoding = self.dropout(encoding)
        
        # Add to input if provided
        if x is not None:
            return x + encoding
        else:
            return encoding
    
    def extra_repr(self) -> str:
        """Return a string representation of the encoding configuration."""
        return (f'd_model={self.d_model}, '
                f'encoding_type={self.encoding_type}, '
                f'max_spatial_distance={self.max_spatial_distance}, '
                f'num_bases={self.num_bases if self.encoding_type == "basis" else "N/A"}, '
                f'learnable={self.learnable}')


class GeometricTemporalAttention(nn.Module):
    """
    Combined geometric and temporal attention module.
    
    This module integrates geometric and temporal attention mechanisms,
    allowing the model to capture both spatial and temporal dependencies.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        geometric_attention (GeometricAttention): Geometric attention module
        temporal_attention (nn.Module): Temporal attention module
        fusion_type (str): How to fuse geometric and temporal attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        distance_metric: str = "euclidean",
        fusion_type: str = "parallel",
        geometric_weight: float = 0.5,
        learnable_fusion: bool = True
    ):
        """
        Initialize the geometric-temporal attention module.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            use_layer_norm: Whether to use layer normalization (default: True)
            distance_metric: Distance metric for geometric attention (default: "euclidean")
            fusion_type: How to fuse geometric and temporal attention (default: "parallel")
                       Options: "parallel", "sequential", "gated"
            geometric_weight: Weight for geometric attention in fusion (default: 0.5)
            learnable_fusion: Whether fusion weights are learnable (default: True)
        """
        super(GeometricTemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.use_layer_norm = use_layer_norm
        self.distance_metric = distance_metric
        self.fusion_type = fusion_type
        self.geometric_weight = geometric_weight
        self.learnable_fusion = learnable_fusion
        
        # Geometric attention
        self.geometric_attention = GeometricAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            distance_metric=distance_metric,
            use_layer_norm=use_layer_norm,
            learnable_distance=True
        )
        
        # Import temporal attention here to avoid circular imports
        from tagan.layers.temporal_attention import AsymmetricTemporalAttention
        
        # Temporal attention
        self.temporal_attention = AsymmetricTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,
            time_aware=True,
            use_layer_norm=use_layer_norm,
            asymmetric_window_size=5,
            future_discount=0.8
        )
        
        # Fusion mechanism
        if fusion_type == "parallel":
            if learnable_fusion:
                # Learnable weights for fusion
                self.fusion_weights = nn.Parameter(torch.tensor([geometric_weight, 1.0 - geometric_weight]))
            else:
                # Fixed weights
                self.register_buffer('fusion_weights', torch.tensor([geometric_weight, 1.0 - geometric_weight]))
        
        elif fusion_type == "gated":
            # Gating mechanism for fusion
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Output layer norm if needed
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        if self.fusion_type == "gated" and self.learnable_fusion:
            nn.init.xavier_uniform_(self.gate_net[0].weight)
            nn.init.zeros_(self.gate_net[0].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        time_stamps: Optional[torch.Tensor] = None,
        coordinates: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        geometric_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for geometric-temporal attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            time_stamps: Optional time stamps [batch_size, seq_len] (default: None)
            coordinates: Optional coordinates [batch_size, seq_len, coordinate_dim] (default: None)
            attention_mask: Optional attention mask [batch_size, seq_len, seq_len] (default: None)
            geometric_bias: Optional geometric bias [batch_size, seq_len, seq_len] (default: None)
            
        Returns:
            Attended tensor [batch_size, seq_len, hidden_dim]
        """
        # Store original for residual connection
        identity = x
        
        if self.fusion_type == "parallel":
            # Apply geometric and temporal attention in parallel
            geometric_out = self.geometric_attention(x, attention_mask, geometric_bias)
            temporal_out = self.temporal_attention(x, time_stamps, attention_mask)
            
            # Fuse outputs with weights
            weights = F.softmax(self.fusion_weights, dim=0)
            output = weights[0] * geometric_out + weights[1] * temporal_out
        
        elif self.fusion_type == "sequential":
            # Apply geometric attention then temporal attention
            geometric_out = self.geometric_attention(x, attention_mask, geometric_bias)
            output = self.temporal_attention(geometric_out, time_stamps, attention_mask)
        
        elif self.fusion_type == "gated":
            # Apply both attentions
            geometric_out = self.geometric_attention(x, attention_mask, geometric_bias)
            temporal_out = self.temporal_attention(x, time_stamps, attention_mask)
            
            # Compute gate values
            gate_input = torch.cat([geometric_out, temporal_out], dim=-1)
            gates = self.gate_net(gate_input)
            
            # Apply gating
            output = gates * geometric_out + (1 - gates) * temporal_out
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Apply dropout
        output = self.dropout(output)
        
        # Apply residual connection
        output = output + identity
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'num_heads={self.num_heads}, '
                f'distance_metric={self.distance_metric}, '
                f'fusion_type={self.fusion_type}, '
                f'geometric_weight={self.geometric_weight}, '
                f'learnable_fusion={self.learnable_fusion}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'dropout={self.dropout_prob}')
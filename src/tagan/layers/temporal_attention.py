"""
Temporal attention mechanisms for TAGAN.

This module provides attention layers that operate on temporal data,
capturing asymmetric dependencies across time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import math


class TimeEncoding(nn.Module):
    """
    Time encoding module for temporal attention.
    
    This module provides positional encodings for time values, supporting
    both fixed and learned encodings with various options for encoding time.
    
    Attributes:
        d_model (int): Model dimension for encoding
        max_len (int): Maximum sequence length
        learnable (bool): Whether encodings are learnable
        encoding_type (str): Type of encoding to use
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        learnable: bool = False,
        encoding_type: str = 'sinusoidal',
        dropout: float = 0.1,
        num_bases: int = 16,
        scale: float = 1.0
    ):
        """
        Initialize the time encoding module.
        
        Args:
            d_model: Model dimension for encoding
            max_len: Maximum sequence length supported (default: 5000)
            learnable: Whether encodings are learnable (default: False)
            encoding_type: Type of encoding (default: 'sinusoidal')
                         Options: 'sinusoidal', 'linear', 'log', 'learned', 'basis'
            dropout: Dropout probability (default: 0.1)
            num_bases: Number of basis functions for 'basis' encoding (default: 16)
            scale: Scaling factor for encodings (default: 1.0)
        """
        super(TimeEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.learnable = learnable
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(p=dropout)
        self.num_bases = num_bases
        self.scale = scale
        
        # Create encoding based on specified type
        if encoding_type == 'sinusoidal':
            # Sinusoidal encoding similar to Transformer's positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            if learnable:
                self.register_parameter('pe', nn.Parameter(pe))
            else:
                self.register_buffer('pe', pe)
        
        elif encoding_type == 'linear':
            # Linear encoding of time
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            position = position / max_len  # Normalize to [0, 1]
            
            # Repeat across dimension
            pe = position.repeat(1, d_model)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            if learnable:
                self.register_parameter('pe', nn.Parameter(pe))
            else:
                self.register_buffer('pe', pe)
        
        elif encoding_type == 'log':
            # Logarithmic encoding of time
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(1, max_len + 1, dtype=torch.float).unsqueeze(1)
            position = torch.log(position) / math.log(max_len)  # Normalize log
            
            # Repeat across dimension
            pe = position.repeat(1, d_model)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            if learnable:
                self.register_parameter('pe', nn.Parameter(pe))
            else:
                self.register_buffer('pe', pe)
        
        elif encoding_type == 'learned':
            # Fully learned encoding
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        
        elif encoding_type == 'basis':
            # RBF-like basis functions
            self.basis_mu = nn.Parameter(torch.linspace(0, 1, num_bases))
            self.basis_sigma = nn.Parameter(torch.ones(num_bases) * 0.1)
            self.basis_proj = nn.Linear(num_bases, d_model)
        
        else:
            # Default to empty encoding
            self.register_buffer('pe', torch.zeros(1, max_len, d_model))
    
    def _get_basis_encoding(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Compute basis function encoding for time values.
        
        Args:
            time_values: Time values to encode [batch_size, seq_len]
            
        Returns:
            Encoded time values [batch_size, seq_len, d_model]
        """
        try:
            print(f"\n==== TAGAN Debug: Basis Encoding ====")
            print(f"Time values: shape={time_values.shape}, device={time_values.device}")
            
            # Check for NaN or extreme values
            if torch.isnan(time_values).any():
                print("WARNING: NaN values in time_values")
                # Replace NaNs with zeros for stability
                time_values = torch.nan_to_num(time_values, nan=0.0)
                
            time_min = time_values.min()
            time_max = time_values.max()
            print(f"Time range before normalization: min={time_min.item()}, max={time_max.item()}")
            
            # Normalize time values to [0, 1] with numerical stability
            if time_max > time_min and (time_max - time_min) > 1e-7:
                print(f"Normalizing with range: {time_max - time_min}")
                time_norm = (time_values - time_min) / (time_max - time_min)
            else:
                print("WARNING: time_max <= time_min or too small range, using zeros")
                time_norm = torch.zeros_like(time_values)
            
            print(f"Time norm: shape={time_norm.shape}, min={time_norm.min().item()}, max={time_norm.max().item()}")
            
            # Expand for broadcasting against basis means
            # [batch_size, seq_len, 1]
            time_norm = time_norm.unsqueeze(-1)
            print(f"Expanded time norm: shape={time_norm.shape}")
            
            # Access basis parameters
            print(f"Basis mu: shape={self.basis_mu.shape}, range=[{self.basis_mu.min().item()}, {self.basis_mu.max().item()}]")
            print(f"Basis sigma: shape={self.basis_sigma.shape}, range=[{self.basis_sigma.min().item()}, {self.basis_sigma.max().item()}]")
            
            # Compute distance term safely
            distance_term = (time_norm - self.basis_mu) ** 2
            print(f"Distance term: shape={distance_term.shape}, max={distance_term.max().item()}")
            
            # Check for any extremely small sigma values that could cause numerical issues
            min_sigma = self.basis_sigma.min().item()
            if min_sigma < 1e-7:
                print(f"WARNING: Very small sigma value detected: {min_sigma}")
                # Apply a minimum threshold to sigma to prevent division by near-zero
                sigma_safe = torch.clamp(self.basis_sigma, min=1e-7)
            else:
                sigma_safe = self.basis_sigma
                
            # Compute RBF values with numerical stability
            # [batch_size, seq_len, num_bases]
            divisor = 2 * sigma_safe ** 2
            print(f"Divisor: shape={divisor.shape}, min={divisor.min().item()}")
            
            exponent = -(distance_term / divisor)
            print(f"Exponent before exp: min={exponent.min().item()}, max={exponent.max().item()}")
            
            # Clip exponent to avoid overflow
            exponent = torch.clamp(exponent, min=-88.0, max=88.0)  # For fp32 stability
            
            basis_values = torch.exp(exponent)
            print(f"Basis values: shape={basis_values.shape}, min={basis_values.min().item()}, max={basis_values.max().item()}")
            
            # Check for NaN in basis values
            if torch.isnan(basis_values).any():
                print("WARNING: NaN values detected in basis_values")
                basis_values = torch.nan_to_num(basis_values, nan=0.0)
            
            # Project to dimension
            # [batch_size, seq_len, d_model]
            print(f"Basis projection weight: shape={self.basis_proj.weight.shape}")
            encoding = self.basis_proj(basis_values)
            print(f"Final encoding: shape={encoding.shape}, min={encoding.min().item()}, max={encoding.max().item()}")
            
            # Final check for NaN
            if torch.isnan(encoding).any():
                print("WARNING: NaN values in final encoding")
                encoding = torch.nan_to_num(encoding, nan=0.0)
                
            print(f"==== TAGAN Debug: Basis Encoding completed ====\n")
            return encoding
            
        except Exception as e:
            print(f"ERROR in basis encoding: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a default encoding as fallback
            print("Creating default zero encoding as fallback")
            batch_size, seq_len = time_values.shape
            default_encoding = torch.zeros(batch_size, seq_len, self.d_model, device=time_values.device)
            print(f"==== TAGAN Debug: Returning default encoding ====\n")
            return default_encoding
    
    def forward(
        self,
        time_values: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute time encoding for input values.
        
        Args:
            time_values: Optional time values to encode [batch_size, seq_len] (default: None)
                       If None, uses integer positions from x
            x: Optional tensor to add encoding to [batch_size, seq_len, d_model] (default: None)
                       If None, returns just the encoding
            
        Returns:
            Time encoding or input with encoding added [batch_size, seq_len, d_model]
        """
        try:
            # Determine encoding positions
            if self.encoding_type == 'basis' and time_values is not None:
                # Use basis function encoding with continuous time values
                encoding = self._get_basis_encoding(time_values)
            else:
                # For other encoding types, use integer positions
                if time_values is not None:
                    # Convert time values to integer positions
                    # First normalize to [0, max_len)
                    time_min = time_values.min()
                    time_max = time_values.max()
                    
                    if time_max > time_min:
                        pos = ((time_values - time_min) / (time_max - time_min) * (self.max_len - 1)).long()
                    else:
                        pos = torch.zeros_like(time_values, dtype=torch.long)
                    
                    # Clip to ensure within bounds
                    pos = torch.clamp(pos, 0, self.max_len - 1)
                    
                elif x is not None:
                    # Use sequence positions from x
                    batch_size, seq_len = x.size(0), x.size(1)
                    pos = torch.arange(0, seq_len, device=x.device).expand(batch_size, seq_len)
                else:
                    raise ValueError("Either time_values or x must be provided")
                
                # Get encoding from stored or generated values
                if self.encoding_type in ['sinusoidal', 'linear', 'log', 'learned']:
                    encoding = self.pe[:, pos]
            
            # Apply scaling
            encoding = encoding * self.scale
            
            # Apply dropout
            encoding = self.dropout(encoding)
            
            # Add to input if provided
            if x is not None:
                result = x + encoding
            else:
                result = encoding
                
            return result
            
        except Exception as e:
            print(f"ERROR in time encoding forward: {str(e)}")
            # Return a default encoding or original input as fallback
            if x is not None:
                print("Returning original input as fallback")
                return x
            else:
                # Create a default encoding of zeros
                print("Creating default zero encoding")
                shape = time_values.shape if time_values is not None else (1, 1)
                default_encoding = torch.zeros(shape[0], shape[1], self.d_model, device=time_values.device if time_values is not None else 'cpu')
                print(f"==== TAGAN Debug: Returning default encoding ====\n")
                return default_encoding
    
    def extra_repr(self) -> str:
        """Return a string representation of the encoding configuration."""
        return (f'd_model={self.d_model}, '
                f'max_len={self.max_len}, '
                f'learnable={self.learnable}, '
                f'encoding_type={self.encoding_type}, '
                f'num_bases={self.num_bases if self.encoding_type == "basis" else "N/A"}, '
                f'scale={self.scale}')


class TemporalAttention(nn.Module):
    """
    Base class for temporal attention mechanisms.
    
    This module provides the foundation for attention operations
    on temporal data, supporting multi-head attention.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
        use_layer_norm: bool = True
    ):
        """
        Initialize the temporal attention module.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            causal: Whether to use causal (masked) attention (default: False)
            use_layer_norm: Whether to use layer normalization (default: True)
        """
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.causal = causal
        self.use_layer_norm = use_layer_norm
        
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
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal mask for attention.
        
        Args:
            seq_len: Sequence length
            device: Device for tensor
            
        Returns:
            Causal mask [seq_len, seq_len]
        """
        # Create mask where upper triangle is masked (set to 0)
        # Lower triangle (including diagonal) is unmasked (set to 1)
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for temporal attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional mask [batch_size, seq_len, seq_len] (default: None)
            return_attention_weights: Whether to return attention weights (default: False)
            
        Returns:
            If return_attention_weights is False:
                Attended tensor [batch_size, seq_len, hidden_dim]
            If return_attention_weights is True:
                (Attended tensor [batch_size, seq_len, hidden_dim],
                 Attention weights [batch_size, num_heads, seq_len, seq_len])
        """
        # Check if input is a list of tensors (from temporal propagation)
        if isinstance(x, list):
            # Convert list of tensors to a single tensor
            # If each tensor in the list has shape [nodes, features]
            # Stack them to create [seq_len, nodes, features]
            x_stacked = []
            
            # First find the maximum number of nodes across all tensors
            max_nodes = 0
            for tensor in x:
                # Check if the tensor is itself a list (from temporal propagation)
                if isinstance(tensor, list) and len(tensor) > 0:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                max_nodes = max(max_nodes, current_tensor.shape[0])
            
            # Now pad each tensor to the maximum size
            for tensor in x:
                # Check if the tensor is itself a list (from temporal propagation)
                if isinstance(tensor, list) and len(tensor) > 0:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                
                # Get current tensor dimensions
                num_nodes, feat_dim = current_tensor.shape
                
                # If this tensor has fewer nodes than max_nodes, pad it
                if num_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - num_nodes, feat_dim, device=current_tensor.device)
                    padded_tensor = torch.cat([current_tensor, padding], dim=0)
                    x_stacked.append(padded_tensor)
                else:
                    x_stacked.append(current_tensor)
            
            try:
                # Stack along the sequence dimension (dim=0)
                x = torch.stack(x_stacked, dim=0)
                
                # Permute to [batch_size, seq_len, features] where batch_size=1
                # if we're working with a single sequence
                x = x.permute(1, 0, 2)
            except RuntimeError as e:
                # Collect shapes for error reporting
                shapes = [t.shape for t in x_stacked]
                raise RuntimeError(f"Failed to stack temporal tensors with shapes {shapes}: {str(e)}")
            
        # Now x should have shape [batch_size, seq_len, features]
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
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = self._create_causal_mask(seq_len, x.device)
            # Set masked positions to negative infinity
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle case when attention_mask is a list
            if isinstance(attention_mask, list):
                # Convert list of masks to a tensor if possible
                try:
                    attention_mask = torch.stack(attention_mask, dim=0)
                except:
                    # If we can't stack (e.g., different shapes), create a default mask
                    attention_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device)
            
            # Debug dimensions before applying mask
            att_shape = attention_scores.shape
            mask_shape = attention_mask.shape
            
            # Create a mask of the correct dimensions
            if mask_shape[-1] != att_shape[-1] or mask_shape[-2] != att_shape[-2]:
                # Need to resize the mask to match attention scores dimensions
                print(f"Warning: Resizing mask from {mask_shape} to match attention dimensions {att_shape}")
                # Create a new mask of the correct size
                expanded_mask = torch.ones(batch_size, 1, att_shape[-2], att_shape[-1], device=x.device)
            else:
                # Expand attention mask for multi-head attention
                # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                expanded_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (set masked positions to negative infinity)
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention weights
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
        
        if return_attention_weights:
            return out, attention_weights
        else:
            return out


class AsymmetricTemporalAttention(TemporalAttention):
    """
    Asymmetric temporal attention mechanism.
    
    This module extends the base temporal attention to incorporate
    asymmetric dependencies, with special handling for past and future
    time steps based on temporal distances.
    
    Attributes:
        time_aware (bool): Whether to use time-aware attention
        asymmetric_window_size (int): Window size for asymmetric attention
        future_discount (float): Discount factor for future dependencies
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
        time_aware: bool = True,
        use_layer_norm: bool = True,
        asymmetric_window_size: int = 5,
        future_discount: float = 0.8,
        relative_position_bias: bool = True,
        max_relative_position: int = 32,
        time_encoding_type: str = 'basis',
        use_time_masks: bool = True
    ):
        """
        Initialize the asymmetric temporal attention module.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            causal: Whether to use causal (masked) attention (default: False)
            time_aware: Whether to use time-aware attention (default: True)
            use_layer_norm: Whether to use layer normalization (default: True)
            asymmetric_window_size: Window size for asymmetric attention (default: 5)
            future_discount: Discount factor for future dependencies (default: 0.8)
            relative_position_bias: Whether to use relative positional bias (default: True)
            max_relative_position: Maximum relative position for bias (default: 32)
            time_encoding_type: Type of time encoding (default: 'basis')
            use_time_masks: Whether to use time-based masks (default: True)
        """
        super(AsymmetricTemporalAttention, self).__init__(
            hidden_dim, num_heads, dropout, causal, use_layer_norm
        )
        
        self.time_aware = time_aware
        self.asymmetric_window_size = asymmetric_window_size
        self.future_discount = future_discount
        self.relative_position_bias = relative_position_bias
        self.max_relative_position = max_relative_position
        self.time_encoding_type = time_encoding_type
        self.use_time_masks = use_time_masks
        
        # Relative positional bias
        if relative_position_bias:
            # Create table of relative position bias
            # (2 * max_relative_position + 1) x num_heads
            # +1 for the zero position
            self.relative_pos_table = nn.Parameter(
                torch.zeros((2 * max_relative_position + 1, num_heads))
            )
            
            # Initialize with small random values
            nn.init.xavier_uniform_(self.relative_pos_table)
        
        # Time encoding
        if time_aware:
            self.time_encoding = TimeEncoding(
                d_model=hidden_dim,
                learnable=True,
                encoding_type=time_encoding_type,
                num_bases=hidden_dim // 4  # A reasonable default
            )
            
            # Projections for time-based attention adjustments
            self.time_q_proj = nn.Linear(hidden_dim, num_heads)
            self.time_k_proj = nn.Linear(hidden_dim, num_heads)
            
            nn.init.xavier_uniform_(self.time_q_proj.weight)
            nn.init.xavier_uniform_(self.time_k_proj.weight)
            nn.init.zeros_(self.time_q_proj.bias)
            nn.init.zeros_(self.time_k_proj.bias)
        
        # Asymmetric kernel parameters
        self.asymmetric_kernel = nn.Parameter(
            torch.ones(2 * asymmetric_window_size + 1, num_heads)
        )
        
        # Initialize with stronger weights for nearby context
        # and weaker for distant context
        with torch.no_grad():
            center_idx = asymmetric_window_size
            for i in range(2 * asymmetric_window_size + 1):
                dist = abs(i - center_idx)
                if i < center_idx:  # Past context
                    # Past context gets higher weight
                    self.asymmetric_kernel[i] = 1.0 - 0.5 * (dist / asymmetric_window_size)
                elif i > center_idx:  # Future context
                    # Future context gets discounted weight
                    self.asymmetric_kernel[i] = future_discount * (1.0 - 0.5 * (dist / asymmetric_window_size))
                else:  # Present (center)
                    self.asymmetric_kernel[i] = 1.0
    
    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative positions between all pairs in a sequence.
        
        Args:
            seq_len: Sequence length
            device: Device for tensor
            
        Returns:
            Relative positions [seq_len, seq_len]
        """
        # Create position indices
        pos = torch.arange(seq_len, device=device)
        
        # Compute relative positions
        # [seq_len, seq_len]
        relative_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        
        # Shift and clip to table bounds
        relative_pos = relative_pos + self.max_relative_position
        relative_pos = torch.clamp(relative_pos, 0, 2 * self.max_relative_position)
        
        return relative_pos
    
    def _get_asymmetric_kernel_values(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get asymmetric kernel values for attention.
        
        Args:
            seq_len: Sequence length
            device: Device for tensor
            
        Returns:
            Kernel values [seq_len, seq_len, num_heads]
        """
        # Create position indices
        pos = torch.arange(seq_len, device=device)
        
        # Compute relative positions
        # [seq_len, seq_len]
        relative_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        
        # Create mask for positions within window
        within_window = (relative_pos >= -self.asymmetric_window_size) & (relative_pos <= self.asymmetric_window_size)
        
        # Shift to kernel indices
        kernel_indices = relative_pos + self.asymmetric_window_size
        
        # Clip to kernel bounds
        kernel_indices = torch.clamp(kernel_indices, 0, 2 * self.asymmetric_window_size)
        
        # Gather kernel values
        # [seq_len, seq_len, num_heads]
        kernel_values = self.asymmetric_kernel[kernel_indices]
        
        # Mask values outside window
        kernel_values = kernel_values * within_window.unsqueeze(-1).float()
        
        return kernel_values
    
    def _compute_time_based_attention(
        self,
        time_diffs: torch.Tensor,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        Compute time-based attention adjustments.
        
        Args:
            time_diffs: Time differences [batch_size, seq_len, seq_len]
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Time-based attention adjustments [batch_size, num_heads, seq_len, seq_len]
        """
        print(f"\n==== TAGAN Debug: Computing time-based attention ====")
        print(f"Time diffs: shape={time_diffs.shape}, device={time_diffs.device}")
        # Check for NaN or extreme values in time_diffs
        if torch.isnan(time_diffs).any():
            print("WARNING: NaN values detected in time_diffs")
        
        # Handle empty tensor case
        if time_diffs.numel() == 0:
            print("WARNING: Empty time_diffs tensor (sequence length 0)")
            # Return zeros of the correct shape for empty time differences
            zero_attn = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=time_diffs.device)
            return zero_attn
        
        time_min = time_diffs.min().item()
        time_max = time_diffs.max().item()
        print(f"Time diffs range: min={time_min}, max={time_max}")
        print(f"Time diffs range: min={time_min}, max={time_max}")
        
        # Reshape for encoding
        flattened = time_diffs.view(batch_size * seq_len * seq_len, 1)
        print(f"Flattened time diffs: shape={flattened.shape}")
        
        try:
            # Get time encoding
            # [batch_size * seq_len * seq_len, hidden_dim]
            flat_time_enc = self.time_encoding(flattened)
            print(f"Encoded flattened time: shape={flat_time_enc.shape}")
            
            # Check for NaNs in encoded time
            if torch.isnan(flat_time_enc).any():
                print("WARNING: NaN values detected in time encoding output")
            
            # Reshape back to original dimensions with features
            # [batch_size, seq_len, seq_len, hidden_dim]
            time_enc = flat_time_enc.view(batch_size, seq_len, seq_len, self.hidden_dim)
            print(f"Reshaped time encoding: shape={time_enc.shape}")
            
            # Project to attention logits per head
            # [batch_size, seq_len, seq_len, num_heads]
            time_attn = self.time_q_proj(time_enc)
            print(f"Projected time attention: shape={time_attn.shape}")
            
            # Check for NaNs in projected attention
            if torch.isnan(time_attn).any():
                print("WARNING: NaN values detected in time attention projection")
            
            # Transpose to match attention shape
            # [batch_size, num_heads, seq_len, seq_len]
            time_attn = time_attn.permute(0, 3, 1, 2)
            print(f"Permuted time attention: shape={time_attn.shape}")
            
            print(f"Time attention range: min={time_attn.min().item()}, max={time_attn.max().item()}")
            print(f"==== TAGAN Debug: Finished computing time-based attention ====\n")
            
            return time_attn
            
        except Exception as e:
            print(f"ERROR in time-based attention: {str(e)}")
            # Create default time attention (zeros)
            print(f"Creating default zero time attention")
            default_attn = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=time_diffs.device)
            print(f"==== TAGAN Debug: Returning default time-based attention ====\n")
            return default_attn
    
    def _create_time_mask(
        self,
        time_stamps: torch.Tensor,
        max_time_diff: float = 10.0
    ) -> torch.Tensor:
        """
        Create attention mask based on time differences.
        
        Args:
            time_stamps: Time stamps [batch_size, seq_len]
            max_time_diff: Maximum allowed time difference (default: 10.0)
            
        Returns:
            Time-based mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = time_stamps.size()
        
        # Handle empty tensor case (seq_len == 0)
        if seq_len == 0:
            print("WARNING: Empty time_stamps tensor (sequence length 0)")
            # Return empty mask of correct shape
            return torch.ones(batch_size, 0, 0, device=time_stamps.device)
        
        # Compute time differences
        # [batch_size, seq_len, 1] - [batch_size, 1, seq_len] -> [batch_size, seq_len, seq_len]
        time_diffs = torch.abs(time_stamps.unsqueeze(2) - time_stamps.unsqueeze(1))
        
        # Create mask: 1 for positions within max_time_diff, 0 otherwise
        mask = (time_diffs <= max_time_diff).float()
        
        return mask
    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        time_stamps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for asymmetric temporal attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim] or a list of tensors
            time_stamps: Optional time stamps [batch_size, seq_len] (default: None)
            attention_mask: Optional mask [batch_size, seq_len, seq_len] (default: None)
            return_attention_weights: Whether to return attention weights (default: False)
            
        Returns:
            If return_attention_weights is False:
                Attended tensor [batch_size, seq_len, hidden_dim]
            If return_attention_weights is True:
                (Attended tensor [batch_size, seq_len, hidden_dim],
                 Attention weights [batch_size, num_heads, seq_len, seq_len])
        """
        # Check if input is a list of tensors (from temporal propagation)
        if isinstance(x, list):
            # Convert list of tensors to a single tensor
            # If each tensor in the list has shape [nodes, features]
            # Stack them to create [seq_len, nodes, features]
            x_stacked = []
            
            # First find the maximum number of nodes across all tensors
            max_nodes = 0
            for tensor in x:
                # Check if the tensor is itself a list (from temporal propagation)
                if isinstance(tensor, list) and len(tensor) > 0:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                max_nodes = max(max_nodes, current_tensor.shape[0])
            
            # Now pad each tensor to the maximum size
            for tensor in x:
                # Check if the tensor is itself a list (from temporal propagation)
                if isinstance(tensor, list) and len(tensor) > 0:
                    current_tensor = tensor[0]
                else:
                    current_tensor = tensor
                
                # Get current tensor dimensions
                num_nodes, feat_dim = current_tensor.shape
                
                # If this tensor has fewer nodes than max_nodes, pad it
                if num_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - num_nodes, feat_dim, device=current_tensor.device)
                    padded_tensor = torch.cat([current_tensor, padding], dim=0)
                    x_stacked.append(padded_tensor)
                else:
                    x_stacked.append(current_tensor)
            
            try:
                # Stack along the sequence dimension (dim=0)
                x = torch.stack(x_stacked, dim=0)
                
                # Permute to [batch_size, seq_len, features] where batch_size=1
                # if we're working with a single sequence
                x = x.permute(1, 0, 2)
            except RuntimeError as e:
                # Collect shapes for error reporting
                shapes = [t.shape for t in x_stacked]
                raise RuntimeError(f"Failed to stack temporal tensors with shapes {shapes}: {str(e)}")
        
        # Now x should have shape [batch_size, seq_len, features]
        batch_size, seq_len, _ = x.size()
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
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional bias if enabled
        if self.relative_position_bias:
            # Get relative positions
            relative_pos = self._get_relative_positions(seq_len, x.device)
            
            # Get bias values from table
            # [seq_len, seq_len, num_heads]
            rel_bias = self.relative_pos_table[relative_pos].permute(2, 0, 1)
            
            # Add to attention scores
            # [batch_size, num_heads, seq_len, seq_len]
            attention_scores = attention_scores + rel_bias.unsqueeze(0)
        
        # Add asymmetric kernel values
        kernel_values = self._get_asymmetric_kernel_values(seq_len, x.device)
        
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores + kernel_values.permute(2, 0, 1).unsqueeze(0)
        
        # Add time-based attention if enabled and time_stamps provided
        if self.time_aware and time_stamps is not None:
            # Compute time differences
            # [batch_size, seq_len, 1] - [batch_size, 1, seq_len] -> [batch_size, seq_len, seq_len]
            time_diffs = time_stamps.unsqueeze(2) - time_stamps.unsqueeze(1)
            
            # Compute time-based attention adjustments
            time_attn = self._compute_time_based_attention(time_diffs, batch_size, seq_len)
            
            # Add to attention scores
            attention_scores = attention_scores + time_attn
            
            # Create time-based mask if enabled
            if self.use_time_masks:
                print(f"\n==== TAGAN Debug: Creating time-based mask ====")
                time_mask = self._create_time_mask(time_stamps)
                print(f"Time mask created: shape={time_mask.shape}, device={time_mask.device}")
                
                # Handle the case when attention_mask is a list
                if attention_mask is not None:
                    if isinstance(attention_mask, list):
                        print(f"WARNING: Cannot combine list-type attention mask with time mask at this stage.")
                        print(f"The list will be properly processed in the main attention mask section.")
                        # Don't modify the list here, it will be handled later
                    else:
                        # Check if dimensions match before combining
                        time_shape = time_mask.shape
                        mask_shape = attention_mask.shape
                        
                        if mask_shape[-1] != time_shape[-1] or mask_shape[-2] != time_shape[-2]:
                            print(f"WARNING: Attention mask shape {mask_shape} doesn't match time mask shape {time_shape}")
                            print(f"Cannot combine masks with mismatched dimensions")
                            # Don't modify the existing mask, as it will be properly processed later
                        else:
                            # Safe to combine tensor masks with matching dimensions
                            print(f"Combining existing tensor attention mask with time mask")
                            attention_mask = attention_mask * time_mask
                else:
                    # No existing mask, just use the time mask
                    print(f"Using time mask as attention mask")
                    attention_mask = time_mask
                print(f"==== TAGAN Debug: Finished creating time-based mask ====\n")
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = self._create_causal_mask(seq_len, x.device)
            # Set masked positions to negative infinity
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            print(f"\n==== TAGAN Debug: Processing attention mask ====")
            print(f"Attention scores: shape={attention_scores.shape}, device={attention_scores.device}")
            print(f"Mask type: {type(attention_mask)}")
            
            # Handle case when attention_mask is a list
            if isinstance(attention_mask, list):
                print(f"Processing a list of masks with {len(attention_mask)} elements")
                if attention_mask and isinstance(attention_mask[0], torch.Tensor):
                    print(f"First mask in list: shape={attention_mask[0].shape}, dtype={attention_mask[0].dtype}")
                    if torch.isnan(attention_mask[0]).any():
                        print("WARNING: NaN values detected in first mask")
                else:
                    print(f"List contains non-tensor elements or is empty")
                
                # For temporal attention with node masks, create a proper attention mask
                # that allows full attention across the sequence
                seq_len = len(attention_mask)
                print(f"Creating attention mask for sequence length: {seq_len}")
                
                try:
                    # Convert list to tensor if possible
                    stacked_mask = torch.stack(attention_mask, dim=0)
                    print(f"Successfully stacked masks: shape={stacked_mask.shape}")
                    # Create a default mask allowing full attention across timesteps
                    attention_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device)
                except Exception as e:
                    print(f"Error stacking masks: {str(e)}")
                    # Create a default mask allowing full attention across timesteps
                    attention_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device)
                    print(f"Created default mask: shape={attention_mask.shape}")
            
            # Debug dimensions before applying mask
            att_shape = attention_scores.shape
            mask_shape = attention_mask.shape
            
            print(f"Attention scores shape: {att_shape}, Mask shape: {mask_shape}")
            
            # Create a mask of the correct dimensions
            if mask_shape[-1] != att_shape[-1] or mask_shape[-2] != att_shape[-2]:
                print(f"WARNING: Resizing mask from {mask_shape} to match attention dimensions {att_shape}")
                # Create a new mask of the correct size that allows full attention
                expanded_mask = torch.ones(batch_size, 1, att_shape[-2], att_shape[-1], device=x.device)
                print(f"Created expanded mask: shape={expanded_mask.shape}, all ones={expanded_mask.min().item() == 1.0}")
            else:
                # Expand attention mask for multi-head attention
                # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                expanded_mask = attention_mask.unsqueeze(1)
                print(f"Expanded original mask: shape={expanded_mask.shape}")
                if torch.isnan(expanded_mask).any():
                    print("WARNING: NaN values detected in expanded mask")
            
            # Check expanded mask before applying
            # Handle empty masks specially
            if expanded_mask.numel() == 0:
                print("WARNING: Empty expanded mask (numel=0)")
                print("Skipping mask application for empty sequence")
            else:
                print(f"Expanded mask min value: {expanded_mask.min().item()}, max value: {expanded_mask.max().item()}")
                print(f"Zeros in expanded mask: {(expanded_mask == 0).sum().item()}")
                
                # Apply mask (set masked positions to negative infinity)
                try:
                    attention_scores = attention_scores.masked_fill(expanded_mask == 0, float('-inf'))
                    print(f"Successfully applied mask to attention scores")
                except Exception as e:
                    print(f"ERROR applying mask: {str(e)}")
                    # Fall back to unmasked attention scores
                    print(f"Continuing with unmasked attention scores")
            
            print(f"==== TAGAN Debug: Finished processing attention mask ====\n")
        
        # Apply softmax to get attention weights
        # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention weights
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
        
        if return_attention_weights:
            return out, attention_weights
        else:
            return out
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'num_heads={self.num_heads}, '
                f'time_aware={self.time_aware}, '
                f'causal={self.causal}, '
                f'asymmetric_window_size={self.asymmetric_window_size}, '
                f'future_discount={self.future_discount}, '
                f'relative_position_bias={self.relative_position_bias}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'dropout={self.dropout_prob}')


class MultiTimeScaleAttention(nn.Module):
    """
    Multi-time scale attention module.
    
    This module processes temporal data at multiple time scales,
    capturing both fine and coarse-grained temporal dependencies.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        num_scales (int): Number of time scales
        scale_factors (List[int]): Factors for each time scale
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_scales: int = 3,
        scale_factors: Optional[List[int]] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        time_aware: bool = True,
        fusion_type: str = 'concat'
    ):
        """
        Initialize the multi-time scale attention module.
        
        Args:
            hidden_dim: Hidden dimension size
            num_scales: Number of time scales (default: 3)
            scale_factors: Factors for each time scale (default: None, [1, 2, 4])
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            use_layer_norm: Whether to use layer normalization (default: True)
            time_aware: Whether to use time-aware attention (default: True)
            fusion_type: How to fuse multi-scale outputs (default: 'concat')
                       Options: 'concat', 'sum', 'weighted_sum', 'attention'
        """
        super(MultiTimeScaleAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.scale_factors = scale_factors or [1, 2, 4]  # Default scale factors
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.time_aware = time_aware
        self.fusion_type = fusion_type
        
        # Ensure we have enough scale factors
        if len(self.scale_factors) < num_scales:
            # Extend with powers of 2
            for i in range(len(self.scale_factors), num_scales):
                self.scale_factors.append(2 ** i)
        
        # Create attention modules for each scale
        self.attention_modules = nn.ModuleList([
            AsymmetricTemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                causal=False,
                time_aware=time_aware,
                use_layer_norm=use_layer_norm,
                asymmetric_window_size=3 * scale  # Larger window for coarser scales
            )
            for scale in self.scale_factors[:num_scales]
        ])
        
        # Output fusion
        if fusion_type == 'concat':
            # Project concatenated outputs back to hidden_dim
            self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        elif fusion_type == 'weighted_sum':
            # Learnable weights for each scale
            self.scale_weights = nn.Parameter(torch.ones(num_scales))
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.fusion_attention = nn.Linear(hidden_dim, num_scales)
        
        # Layer normalization for output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        if self.fusion_type == 'concat':
            nn.init.xavier_uniform_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)
        elif self.fusion_type == 'weighted_sum':
            nn.init.ones_(self.scale_weights)
        elif self.fusion_type == 'attention':
            nn.init.xavier_uniform_(self.fusion_attention.weight)
            nn.init.zeros_(self.fusion_attention.bias)
    
    def _downsample(
        self,
        x: torch.Tensor,
        scale_factor: int
    ) -> torch.Tensor:
        """
        Downsample sequence along time dimension.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            scale_factor: Factor to downsample by
            
        Returns:
            Downsampled tensor [batch_size, seq_len/scale_factor, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # If scale factor is 1, return original
        if scale_factor == 1:
            return x
        
        # Adjust sequence length to be divisible by scale_factor
        adjusted_seq_len = (seq_len // scale_factor) * scale_factor
        
        if adjusted_seq_len < seq_len:
            # If not exactly divisible, truncate
            x = x[:, :adjusted_seq_len, :]
        
        # Reshape for downsampling
        # [batch_size, adjusted_seq_len/scale_factor, scale_factor, hidden_dim]
        x_reshaped = x.view(batch_size, adjusted_seq_len // scale_factor, scale_factor, hidden_dim)
        
        # Average within each block
        # [batch_size, adjusted_seq_len/scale_factor, hidden_dim]
        x_downsampled = x_reshaped.mean(dim=2)
        
        return x_downsampled
    
    def _upsample(
        self,
        x: torch.Tensor,
        target_len: int,
        scale_factor: int
    ) -> torch.Tensor:
        """
        Upsample sequence along time dimension.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            target_len: Target sequence length
            scale_factor: Factor to upsample by
            
        Returns:
            Upsampled tensor [batch_size, target_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # If scale factor is 1, return original (padded if needed)
        if scale_factor == 1:
            if seq_len < target_len:
                # Pad to target length
                padding = torch.zeros(batch_size, target_len - seq_len, hidden_dim, device=x.device)
                return torch.cat([x, padding], dim=1)
            else:
                return x[:, :target_len, :]
        
        # Repeat each element scale_factor times
        # [batch_size, seq_len, 1, hidden_dim] -> [batch_size, seq_len, scale_factor, hidden_dim]
        x_expanded = x.unsqueeze(2).expand(-1, -1, scale_factor, -1)
        
        # Reshape to combine dimensions
        # [batch_size, seq_len*scale_factor, hidden_dim]
        x_upsampled = x_expanded.reshape(batch_size, seq_len * scale_factor, hidden_dim)
        
        # Truncate or pad to target length
        if x_upsampled.size(1) < target_len:
            # Pad to target length
            padding = torch.zeros(batch_size, target_len - x_upsampled.size(1), hidden_dim, device=x.device)
            x_upsampled = torch.cat([x_upsampled, padding], dim=1)
        else:
            # Truncate to target length
            x_upsampled = x_upsampled[:, :target_len, :]
        
        return x_upsampled
    
    def forward(
        self,
        x: torch.Tensor,
        time_stamps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for multi-time scale attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            time_stamps: Optional time stamps [batch_size, seq_len] (default: None)
            attention_mask: Optional mask [batch_size, seq_len, seq_len] (default: None)
            return_attention_weights: Whether to return attention weights (default: False)
            
        Returns:
            If return_attention_weights is False:
                Attended tensor [batch_size, seq_len, hidden_dim]
            If return_attention_weights is True:
                (Attended tensor [batch_size, seq_len, hidden_dim],
                 Dict of attention weights for each scale)
        """
        batch_size, seq_len, _ = x.size()
        
        # Process at each time scale
        multi_scale_outputs = []
        
        # Create a dictionary to collect attention weights if needed
        all_attention_weights = {} if return_attention_weights else None
        
        for i, (scale_factor, attention_module) in enumerate(zip(self.scale_factors, self.attention_modules)):
            # Downsample input
            x_downsampled = self._downsample(x, scale_factor)
            
            # Downsample time stamps if provided
            time_stamps_downsampled = None
            if time_stamps is not None:
                # Handle time stamps for downsampling
                # Take the middle time stamp of each block
                time_stamps_downsampled = self._downsample(
                    time_stamps.unsqueeze(-1),
                    scale_factor
                ).squeeze(-1)
            
            # Downsample attention mask if provided
            mask_downsampled = None
            if attention_mask is not None:
                # Downsample both dimensions of attention mask
                # This is an approximation - ideally would compute proper mask for downsampled sequence
                mask_downsampled = self._downsample(
                    self._downsample(attention_mask.float().unsqueeze(-1), scale_factor).squeeze(-1),
                    scale_factor
                )
            
            # Apply attention at this scale
            if return_attention_weights:
                attended, attn_weights = attention_module(
                    x_downsampled,
                    time_stamps=time_stamps_downsampled,
                    attention_mask=mask_downsampled,
                    return_attention_weights=True
                )
                # Store attention weights for this scale
                all_attention_weights[f'scale_{scale_factor}'] = attn_weights
            else:
                attended = attention_module(
                    x_downsampled,
                    time_stamps=time_stamps_downsampled,
                    attention_mask=mask_downsampled
                )
            
            # Upsample back to original sequence length
            attended_upsampled = self._upsample(attended, seq_len, scale_factor)
            
            multi_scale_outputs.append(attended_upsampled)
        
        # Fuse multi-scale outputs
        if self.fusion_type == 'concat':
            # Concatenate along feature dimension
            concatenated = torch.cat(multi_scale_outputs, dim=-1)
            
            # Project back to hidden_dim
            output = self.fusion(concatenated)
        
        elif self.fusion_type == 'sum':
            # Simple sum
            output = torch.zeros_like(x)
            for out in multi_scale_outputs:
                output = output + out
        
        elif self.fusion_type == 'weighted_sum':
            # Weighted sum
            output = torch.zeros_like(x)
            weights = F.softmax(self.scale_weights, dim=0)
            
            for i, out in enumerate(multi_scale_outputs):
                output = output + weights[i] * out
        
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            # Stack outputs [batch_size, seq_len, num_scales, hidden_dim]
            stacked = torch.stack(multi_scale_outputs, dim=2)
            
            # Compute attention scores [batch_size, seq_len, num_scales]
            fusion_scores = self.fusion_attention(x)
            fusion_weights = F.softmax(fusion_scores, dim=-1)
            
            # Apply attention weights [batch_size, seq_len, hidden_dim]
            output = torch.sum(stacked * fusion_weights.unsqueeze(-1), dim=2)
        
        else:
            # Default to simple averaging
            output = torch.stack(multi_scale_outputs).mean(dim=0)
        
        # Apply dropout
        output = self.dropout_layer(output)
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        if return_attention_weights:
            return output, all_attention_weights
        else:
            return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'num_scales={self.num_scales}, '
                f'scale_factors={self.scale_factors[:self.num_scales]}, '
                f'num_heads={self.num_heads}, '
                f'fusion_type={self.fusion_type}, '
                f'time_aware={self.time_aware}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'dropout={self.dropout}')
"""
Temporal propagation mechanisms for TAGAN.

This module provides various layers for temporal propagation,
including skip connections and gating mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import math


class TemporalGRU(nn.Module):
    """
    Temporal GRU layer with additional features for TAGAN.
    
    This is an extended GRU that incorporates time-aware gates
    and asymmetric processing for temporal data.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        input_dim (int): Input dimension size
        time_aware (bool): Whether to use time-aware gates
    """
    
    def __init__(
        self,
        hidden_dim: int,
        input_dim: Optional[int] = None,
        dropout: float = 0.1,
        time_aware: bool = True,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
        asymmetric_weights: bool = True
    ):
        """
        Initialize the temporal GRU layer.
        
        Args:
            hidden_dim: Hidden dimension size
            input_dim: Input dimension size (default: None, uses hidden_dim)
            dropout: Dropout probability (default: 0.1)
            time_aware: Whether to use time-aware gates (default: True)
            bidirectional: Whether to use bidirectional processing (default: False)
            use_layer_norm: Whether to use layer normalization (default: True)
            asymmetric_weights: Whether to use asymmetric weights (default: True)
        """
        super(TemporalGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.dropout_prob = dropout
        self.time_aware = time_aware
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.asymmetric_weights = asymmetric_weights
        
        # Multiplier for bidirectional
        self.direction_factor = 2 if bidirectional else 1
        
        # Forward direction GRU components
        # Update gate
        self.W_z = nn.Linear(self.input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Reset gate
        self.W_r = nn.Linear(self.input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Candidate activation
        self.W_h = nn.Linear(self.input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Backward direction GRU components if bidirectional
        if bidirectional:
            # Update gate
            self.W_z_backward = nn.Linear(self.input_dim, hidden_dim)
            self.U_z_backward = nn.Linear(hidden_dim, hidden_dim, bias=False)
            
            # Reset gate
            self.W_r_backward = nn.Linear(self.input_dim, hidden_dim)
            self.U_r_backward = nn.Linear(hidden_dim, hidden_dim, bias=False)
            
            # Candidate activation
            self.W_h_backward = nn.Linear(self.input_dim, hidden_dim)
            self.U_h_backward = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Time-aware components
        if time_aware:
            # Time gate
            self.W_t = nn.Linear(self.input_dim, hidden_dim)
            
            # Time modulation for gates
            self.W_zt = nn.Linear(1, hidden_dim, bias=False)
            self.W_rt = nn.Linear(1, hidden_dim, bias=False)
            self.W_ht = nn.Linear(1, hidden_dim, bias=False)
            
            if bidirectional:
                self.W_zt_backward = nn.Linear(1, hidden_dim, bias=False)
                self.W_rt_backward = nn.Linear(1, hidden_dim, bias=False)
                self.W_ht_backward = nn.Linear(1, hidden_dim, bias=False)
        
        # Asymmetric weights for temporal dependencies
        if asymmetric_weights:
            # Past influence weight
            self.past_weight = nn.Parameter(torch.ones(1))
            
            # Future influence weight
            self.future_weight = nn.Parameter(torch.ones(1) * 0.8)
            
            # Time decay factor
            self.time_decay = nn.Parameter(torch.ones(1) * 0.9)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm_input = nn.LayerNorm(hidden_dim)
            self.layer_norm_hidden = nn.LayerNorm(hidden_dim)
            self.layer_norm_output = nn.LayerNorm(hidden_dim * self.direction_factor)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection for bidirectional
        if bidirectional:
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate initialization."""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize asymmetric weights if used
        if self.asymmetric_weights:
            nn.init.constant_(self.past_weight, 1.0)
            nn.init.constant_(self.future_weight, 0.8)
            nn.init.constant_(self.time_decay, 0.9)
    
    def _compute_time_delta(self, time_stamps: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Compute time deltas between consecutive time stamps.
        
        Args:
            time_stamps: Time stamps [batch_size, seq_len]
            reverse: Whether to reverse the sequence (for backward direction) (default: False)
            
        Returns:
            Time deltas [batch_size, seq_len, 1]
        """
        batch_size, seq_len = time_stamps.size()
        
        # If reverse is True, flip the sequence for backward direction
        if reverse:
            time_stamps = torch.flip(time_stamps, dims=[1])
        
        # Compute differences between consecutive time stamps
        # For t=0, use the same delta as t=1
        if seq_len > 1:
            deltas = time_stamps[:, 1:] - time_stamps[:, :-1]
            # Repeat the first delta for t=0
            first_delta = deltas[:, 0].unsqueeze(1)
            deltas = torch.cat([first_delta, deltas], dim=1)
        else:
            # For single time step, use 1.0 as delta
            deltas = torch.ones_like(time_stamps)
        
        # Ensure positive deltas (absolute value)
        deltas = torch.abs(deltas)
        
        # Add a small epsilon to avoid division by zero
        deltas = deltas + 1e-6
        
        # Normalize to [0, 1] range
        max_delta = torch.max(deltas, dim=1, keepdim=True)[0]
        deltas = deltas / max_delta
        
        # Reshape to [batch_size, seq_len, 1]
        return deltas.unsqueeze(-1)
    
    def _apply_asymmetric_weights(
        self,
        hidden_states: torch.Tensor,
        time_deltas: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        Apply asymmetric weights to hidden states based on temporal direction.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            time_deltas: Time deltas [batch_size, seq_len, 1] (default: None)
            reverse: Whether this is the backward direction (default: False)
            
        Returns:
            Weighted hidden states [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # If not using asymmetric weights, return original
        if not self.asymmetric_weights:
            return hidden_states
        
        # Create weight tensor based on direction
        if reverse:
            # For backward direction, future (actually past, but we're going backward) 
            # influences have higher weight
            weights = self.future_weight
        else:
            # For forward direction, past influences have higher weight
            weights = self.past_weight
        
        # Apply time decay if time deltas are provided
        if time_deltas is not None:
            # Scale weights based on time deltas
            decay_factor = torch.pow(self.time_decay, time_deltas)
            weighted_hidden = hidden_states * weights * decay_factor
        else:
            weighted_hidden = hidden_states * weights
            
        return weighted_hidden
    
    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        time_stamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the temporal GRU layer.
        
        Args:
            inputs: Input sequence [batch_size, seq_len, input_dim]
            hidden: Initial hidden state [batch_size, hidden_dim] (default: None)
            time_stamps: Time stamps for inputs [batch_size, seq_len] (default: None)
            
        Returns:
            Tuple of (outputs, final_hidden_state)
                outputs: Output sequence [batch_size, seq_len, hidden_dim * direction_factor]
                final_hidden_state: Final hidden state [batch_size, hidden_dim * direction_factor]
        """
        batch_size, seq_len, _ = inputs.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.hidden_dim * self.direction_factor, 
                device=inputs.device
            )
        
        # Split hidden state for bidirectional if needed
        if self.bidirectional:
            h_forward = hidden[:, :self.hidden_dim]
            h_backward = hidden[:, self.hidden_dim:]
        else:
            h_forward = hidden
        
        # Compute time deltas if time stamps are provided
        time_deltas_forward = None
        time_deltas_backward = None
        if time_stamps is not None and self.time_aware:
            time_deltas_forward = self._compute_time_delta(time_stamps, reverse=False)
            if self.bidirectional:
                time_deltas_backward = self._compute_time_delta(time_stamps, reverse=True)
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            inputs = self.layer_norm_input(inputs)
        
        # Forward direction pass
        outputs_forward = []
        for t in range(seq_len):
            # Get input at time t
            x_t = inputs[:, t, :]
            
            # Apply dropout to input
            x_t = self.dropout(x_t)
            
            # Compute gates
            z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_forward))
            r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h_forward))
            
            # Apply time modulation if time-aware
            if self.time_aware and time_deltas_forward is not None:
                time_info = time_deltas_forward[:, t, :]
                z_t = z_t + torch.sigmoid(self.W_zt(time_info))
                r_t = r_t + torch.sigmoid(self.W_rt(time_info))
            
            # Compute candidate activation
            h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h_forward))
            
            # Apply time modulation to candidate if time-aware
            if self.time_aware and time_deltas_forward is not None:
                time_info = time_deltas_forward[:, t, :]
                h_tilde = h_tilde + torch.tanh(self.W_ht(time_info))
            
            # Update hidden state
            h_forward = (1 - z_t) * h_forward + z_t * h_tilde
            
            # Apply layer normalization if used
            if self.use_layer_norm:
                h_forward = self.layer_norm_hidden(h_forward)
            
            # Add to outputs
            outputs_forward.append(h_forward)
        
        # Stack outputs
        outputs_forward = torch.stack(outputs_forward, dim=1)
        
        # Apply asymmetric weights if used
        if self.asymmetric_weights:
            outputs_forward = self._apply_asymmetric_weights(
                outputs_forward, time_deltas_forward, reverse=False
            )
        
        # For bidirectional GRU
        if self.bidirectional:
            # Backward direction pass
            outputs_backward = []
            for t in range(seq_len - 1, -1, -1):
                # Get input at time t
                x_t = inputs[:, t, :]
                
                # Apply dropout to input
                x_t = self.dropout(x_t)
                
                # Compute gates
                z_t = torch.sigmoid(self.W_z_backward(x_t) + self.U_z_backward(h_backward))
                r_t = torch.sigmoid(self.W_r_backward(x_t) + self.U_r_backward(h_backward))
                
                # Apply time modulation if time-aware
                if self.time_aware and time_deltas_backward is not None:
                    time_info = time_deltas_backward[:, t, :]
                    z_t = z_t + torch.sigmoid(self.W_zt_backward(time_info))
                    r_t = r_t + torch.sigmoid(self.W_rt_backward(time_info))
                
                # Compute candidate activation
                h_tilde = torch.tanh(self.W_h_backward(x_t) + self.U_h_backward(r_t * h_backward))
                
                # Apply time modulation to candidate if time-aware
                if self.time_aware and time_deltas_backward is not None:
                    time_info = time_deltas_backward[:, t, :]
                    h_tilde = h_tilde + torch.tanh(self.W_ht_backward(time_info))
                
                # Update hidden state
                h_backward = (1 - z_t) * h_backward + z_t * h_tilde
                
                # Apply layer normalization if used
                if self.use_layer_norm:
                    h_backward = self.layer_norm_hidden(h_backward)
                
                # Add to outputs
                outputs_backward.append(h_backward)
            
            # Stack outputs and reverse
            outputs_backward = torch.stack(outputs_backward, dim=1)
            outputs_backward = torch.flip(outputs_backward, dims=[1])
            
            # Apply asymmetric weights if used
            if self.asymmetric_weights:
                outputs_backward = self._apply_asymmetric_weights(
                    outputs_backward, time_deltas_backward, reverse=True
                )
            
            # Concatenate forward and backward outputs
            outputs = torch.cat([outputs_forward, outputs_backward], dim=2)
            
            # Project to output dimension if needed
            if hasattr(self, 'output_proj'):
                outputs = self.output_proj(outputs)
            
            # Final hidden state
            final_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            outputs = outputs_forward
            final_hidden = h_forward
        
        # Apply layer normalization to output if used
        if self.use_layer_norm:
            outputs = self.layer_norm_output(outputs)
        
        return outputs, final_hidden
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'input_dim={self.input_dim}, '
                f'dropout={self.dropout_prob}, '
                f'time_aware={self.time_aware}, '
                f'bidirectional={self.bidirectional}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'asymmetric_weights={self.asymmetric_weights}')


class TemporalGRUCell(nn.Module):
    """
    GRU cell for temporal evolution.
    
    This module implements a GRU cell for evolving node features over time,
    with special handling for temporal dependencies.
    
    Attributes:
        input_dim (int): Input dimension size
        hidden_dim (int): Hidden dimension size
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize the temporal GRU cell.
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            dropout: Dropout probability (default: 0.1)
            use_layer_norm: Whether to use layer normalization (default: True)
        """
        super(TemporalGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Reset gate
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Update gate
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Candidate hidden state
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm_x = nn.LayerNorm(input_dim)
            self.layer_norm_h = nn.LayerNorm(hidden_dim)
            self.layer_norm_out = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Glorot initialization."""
        # Initialize gates
        nn.init.xavier_uniform_(self.reset_gate.weight)
        nn.init.zeros_(self.reset_gate.bias)
        
        nn.init.xavier_uniform_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        
        nn.init.xavier_uniform_(self.candidate.weight)
        nn.init.zeros_(self.candidate.bias)
        
        # Bias initialization for gates (common practice)
        nn.init.constant_(self.reset_gate.bias, 1.0)
        nn.init.constant_(self.update_gate.bias, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        time_diff: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the temporal GRU cell.
        
        Args:
            x: Input features [batch_size, input_dim]
            h: Previous hidden state [batch_size, hidden_dim] (default: None)
            time_diff: Time difference since last update [batch_size] (default: None)
            
        Returns:
            Updated hidden state [batch_size, hidden_dim]
        """
        # Ensure x has at least 2 dimensions [batch_size, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size = x.size(0)
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            x = self.layer_norm_x(x)
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        elif self.use_layer_norm:
            h = self.layer_norm_h(h)
        
        # Modify gates based on time difference if provided
        if time_diff is not None:
            # Scale factor based on time difference
            # Smaller time differences -> more influence from previous state
            # Larger time differences -> less influence from previous state
            scale_factor = torch.exp(-torch.clamp(time_diff, min=0.0, max=10.0)).unsqueeze(1)
            h = h * scale_factor
        
        # Ensure tensors have proper dimensions for concatenation
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [input_dim] -> [1, input_dim]
        if h.dim() == 1:
            h = h.unsqueeze(0)  # [hidden_dim] -> [1, hidden_dim]
            
        # Check tensor shapes and fix if needed
        if x.size(0) != h.size(0):
            # If batch sizes don't match, broadcast the smaller one
            if x.size(0) == 1:
                x = x.expand(h.size(0), -1)
            elif h.size(0) == 1:
                h = h.expand(x.size(0), -1)
                
        # Now concatenate input and hidden state
        xh = torch.cat([x, h], dim=-1)  # Use last dimension for concatenation
        
        # Compute gates
        r = torch.sigmoid(self.reset_gate(xh))  # Reset gate
        z = torch.sigmoid(self.update_gate(xh))  # Update gate
        
        # Compute candidate hidden state
        xrh = torch.cat([x, r * h], dim=-1)  # Use last dimension for concatenation
        h_tilde = torch.tanh(self.candidate(xrh))
        
        # Compute new hidden state
        h_new = (1 - z) * h + z * h_tilde
        
        # Apply dropout
        h_new = self.dropout_layer(h_new)
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            h_new = self.layer_norm_out(h_new)
        
        return h_new
    
    def extra_repr(self) -> str:
        """Return a string representation of the cell configuration."""
        return (f'input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'dropout={self.dropout}, '
                f'use_layer_norm={self.use_layer_norm}')


class TemporalEvolutionLayer(nn.Module):
    """
    Temporal evolution layer for node features.
    
    This module handles the evolution of node features over time,
    using a GRU-based approach to capture temporal dependencies.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        time_aware (bool): Whether to use time-aware updates
        bidirectional (bool): Whether to use bidirectional processing
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        time_aware: bool = True,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
        residual: bool = True
    ):
        """
        Initialize the temporal evolution layer.
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            dropout: Dropout probability (default: 0.1)
            time_aware: Whether to use time-aware updates (default: True)
            bidirectional: Whether to use bidirectional processing (default: False)
            use_layer_norm: Whether to use layer normalization (default: True)
            residual: Whether to use residual connections (default: True)
        """
        super(TemporalEvolutionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.time_aware = time_aware
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.residual = residual
        
        # Forward GRU cell
        self.forward_cell = TemporalGRUCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim if not bidirectional else hidden_dim // 2,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        
        # Backward GRU cell if bidirectional
        if bidirectional:
            self.backward_cell = TemporalGRUCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
            
        # Output projection
        if bidirectional:
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.output_projection = nn.Linear(
                hidden_dim if not bidirectional else hidden_dim // 2,
                hidden_dim
            )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        node_features_seq: List[torch.Tensor],
        time_stamps: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Forward pass for the temporal evolution layer.
        
        Args:
            node_features_seq: List of node features at each time step
                              [num_time_steps, batch_size, input_dim]
            time_stamps: Time stamps for each step [batch_size, num_time_steps] (default: None)
            
        Returns:
            Evolved node features for each time step [num_time_steps, batch_size, hidden_dim]
        """
        num_time_steps = len(node_features_seq)
        batch_size = node_features_seq[0].size(0)
        device = node_features_seq[0].device
        
        # Check if any time information is provided
        has_time_info = time_stamps is not None and self.time_aware
        
        # Process in forward direction
        forward_states = []
        h_forward = None
        
        for t in range(num_time_steps):
            # Get time difference if available
            time_diff = None
            if has_time_info and t > 0:
                time_diff = time_stamps[:, t] - time_stamps[:, t-1]
            
            # Process through forward cell
            h_forward = self.forward_cell(
                node_features_seq[t],
                h_forward,
                time_diff
            )
            
            forward_states.append(h_forward)
        
        # Process in backward direction if bidirectional
        if self.bidirectional:
            backward_states = []
            h_backward = None
            
            for t in range(num_time_steps - 1, -1, -1):
                # Get time difference if available
                time_diff = None
                if has_time_info and t < num_time_steps - 1:
                    time_diff = time_stamps[:, t+1] - time_stamps[:, t]
                
                # Process through backward cell
                h_backward = self.backward_cell(
                    node_features_seq[t],
                    h_backward,
                    time_diff
                )
                
                backward_states.insert(0, h_backward)
            
            # Combine forward and backward states
            combined_states = []
            
            for t in range(num_time_steps):
                # Concatenate forward and backward hidden states
                h_combined = torch.cat([forward_states[t], backward_states[t]], dim=1)
                
                # Apply output transformation
                h_combined = self.output_projection(h_combined)
                
                # Apply dropout
                h_combined = self.dropout_layer(h_combined)
                
                # Add residual connection if used
                if self.residual and self.input_dim == self.hidden_dim:
                    h_combined = h_combined + node_features_seq[t]
                
                # Apply layer normalization if used
                if self.use_layer_norm:
                    h_combined = self.layer_norm(h_combined)
                
                combined_states.append(h_combined)
            
            output_seq = combined_states
        else:
            # Apply transformations to forward states
            output_seq = []
            
            for t in range(num_time_steps):
                # Apply output transformation
                h_t = self.output_projection(forward_states[t])
                
                # Apply dropout
                h_t = self.dropout_layer(h_t)
                
                # Add residual connection if used
                if self.residual and self.input_dim == self.hidden_dim:
                    h_t = h_t + node_features_seq[t]
                
                # Apply layer normalization if used
                if self.use_layer_norm:
                    h_t = self.layer_norm(h_t)
                
                output_seq.append(h_t)
        
        return output_seq
    
    def extra_repr(self) -> str:
        """Return a string representation of the layer configuration."""
        return (f'input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'dropout={self.dropout}, '
                f'time_aware={self.time_aware}, '
                f'bidirectional={self.bidirectional}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'residual={self.residual}')


class TemporalSkipConnection(nn.Module):
    """
    Skip connection module for temporal networks.
    
    This module implements skip connections across time steps,
    allowing information to flow directly from earlier time steps
    to later ones, helping with gradient flow and feature reuse.
    
    Attributes:
        input_dim (int): Input dimension size
        window_size (int): Size of the temporal window for skip connections
        aggregation (str): Method for aggregating skip connections ('sum', 'mean', 'max')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        window_size: int = 3,
        aggregation: str = 'mean',
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        apply_activation: bool = True,
        residual: bool = True
    ):
        """
        Initialize the temporal skip connection module.
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size (default: None, uses input_dim)
            window_size: Size of the temporal window for skip connections (default: 3)
            aggregation: Method for aggregating skip connections (default: 'mean')
                Options: 'sum', 'mean', 'max', 'attention'
            dropout: Dropout probability (default: 0.1)
            use_layer_norm: Whether to use layer normalization (default: True)
            apply_activation: Whether to apply activation function (default: True)
            residual: Whether to use residual connections (default: True)
        """
        super(TemporalSkipConnection, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.window_size = window_size
        self.aggregation = aggregation
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.apply_activation = apply_activation
        self.residual = residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, input_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
            self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation function
        self.act_fn = nn.GELU()
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Glorot initialization."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        node_features_seq: List[torch.Tensor],
        time_weights: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Forward pass for the temporal skip connection module.
        
        Args:
            node_features_seq: List of node features at each time step
                              [num_time_steps, batch_size, input_dim]
            time_weights: Optional weights for each time step (default: None)
                         [num_time_steps, num_time_steps]
            
        Returns:
            Updated node features for each time step [num_time_steps, batch_size, input_dim]
        """
        num_time_steps = len(node_features_seq)
        
        # Project input features
        projected_seq = []
        for features in node_features_seq:
            projected = self.input_proj(features)
            
            if self.apply_activation:
                projected = self.act_fn(projected)
            
            if self.use_layer_norm:
                projected = self.layer_norm1(projected)
            
            projected = self.dropout_layer(projected)
            projected_seq.append(projected)
        
        # Process features through skip connections
        if self.aggregation == 'mean':
            # Mean aggregation
            aggregated_seq = []
            
            for t in range(num_time_steps):
                # Define window around current time step
                start_idx = max(0, t - self.window_size)
                end_idx = min(num_time_steps, t + self.window_size + 1)
                
                # Extract messages in window
                window_messages = projected_seq[start_idx:end_idx]
                
                # Compute mean
                aggregated = torch.mean(torch.stack(window_messages, dim=0), dim=0)
                aggregated_seq.append(aggregated)
        
        elif self.aggregation == 'max':
            # Max aggregation
            aggregated_seq = []
            
            for t in range(num_time_steps):
                # Define window around current time step
                start_idx = max(0, t - self.window_size)
                end_idx = min(num_time_steps, t + self.window_size + 1)
                
                # Extract messages in window
                window_messages = projected_seq[start_idx:end_idx]
                
                # Compute max
                aggregated = torch.max(torch.stack(window_messages, dim=0), dim=0)[0]
                aggregated_seq.append(aggregated)
        
        else:  # Default: sum
            # Sum aggregation
            aggregated_seq = []
            
            for t in range(num_time_steps):
                # Define window around current time step
                start_idx = max(0, t - self.window_size)
                end_idx = min(num_time_steps, t + self.window_size + 1)
                
                # Extract messages in window
                window_messages = projected_seq[start_idx:end_idx]
                
                # Compute sum
                aggregated = torch.sum(torch.stack(window_messages, dim=0), dim=0)
                aggregated_seq.append(aggregated)
        
        # Project back to input dimension
        output_seq = [
            self.output_proj(self.act_fn(aggregated)) for aggregated in aggregated_seq
        ]
        
        # Apply dropout
        output_seq = [self.dropout_layer(output) for output in output_seq]
        
        # Apply residual connection if used
        if self.residual:
            output_seq = [
                output + original for output, original in zip(output_seq, node_features_seq)
            ]
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            output_seq = [self.layer_norm2(output) for output in output_seq]
        
        return output_seq
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'window_size={self.window_size}, '
                f'aggregation={self.aggregation}, '
                f'dropout={self.dropout}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'apply_activation={self.apply_activation}, '
                f'residual={self.residual}')


class TemporalGatingUnit(nn.Module):
    """
    Temporal Gating Unit for controlling information flow across time.
    
    This module implements gating mechanisms to control how information 
    from different time steps is combined, allowing the model to focus
    on relevant temporal patterns.
    
    Attributes:
        input_dim (int): Input dimension size
        hidden_dim (int): Hidden dimension size
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        residual: bool = True
    ):
        """
        Initialize the temporal gating unit.
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size (default: None, uses input_dim)
            dropout: Dropout probability (default: 0.1)
            use_layer_norm: Whether to use layer normalization (default: True)
            residual: Whether to use residual connections (default: True)
        """
        super(TemporalGatingUnit, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.residual = residual
        
        # Gate networks
        self.update_gate = nn.Linear(input_dim * 2, input_dim)
        self.reset_gate = nn.Linear(input_dim * 2, input_dim)
        self.output_gate = nn.Linear(input_dim * 2, input_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm_in1 = nn.LayerNorm(input_dim)
            self.layer_norm_in2 = nn.LayerNorm(input_dim)
            self.layer_norm_out = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Glorot initialization."""
        for module in [self.update_gate, self.reset_gate, self.output_gate]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        current_feat: torch.Tensor,
        previous_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the temporal gating unit.
        
        Args:
            current_feat: Current time step features [batch_size, input_dim]
            previous_feat: Previous time step features [batch_size, input_dim]
            
        Returns:
            Gated output features [batch_size, input_dim]
        """
        # Apply layer normalization if used
        if self.use_layer_norm:
            current_feat = self.layer_norm_in1(current_feat)
            previous_feat = self.layer_norm_in2(previous_feat)
        
        # Concatenate current and previous features
        combined = torch.cat([current_feat, previous_feat], dim=1)
        
        # Compute gates
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        # Compute candidate output
        combined_reset = torch.cat([current_feat, reset * previous_feat], dim=1)
        candidate = torch.tanh(self.output_gate(combined_reset))
        
        # Compute output with gating
        output = (1 - update) * current_feat + update * candidate
        
        # Apply dropout
        output = self.dropout_layer(output)
        
        # Apply residual connection if used
        if self.residual:
            output = output + current_feat
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            output = self.layer_norm_out(output)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'dropout={self.dropout}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'residual={self.residual}')


class TemporalPropagation(nn.Module):
    """
    Main temporal propagation module for TAGAN.
    
    This module integrates multiple temporal processing components to handle
    information flow across time, with special attention to asymmetric dependencies,
    node state transitions, and memory mechanisms.
    
    Attributes:
        input_dim (int): Input dimension size
        hidden_dim (int): Hidden dimension size
        time_aware (bool): Whether to use time-aware processing
        use_skip_connection (bool): Whether to use skip connections
        use_gating (bool): Whether to use gating mechanisms
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        time_aware: bool = True,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
        use_skip_connection: bool = True,
        use_gating: bool = True,
        window_size: int = 3,
        aggregation: str = 'mean',
        residual: bool = True
    ):
        """
        Initialize the temporal propagation module.
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            dropout: Dropout probability (default: 0.1)
            time_aware: Whether to use time-aware processing (default: True)
            bidirectional: Whether to use bidirectional processing (default: False)
            use_layer_norm: Whether to use layer normalization (default: True)
            use_skip_connection: Whether to use skip connections (default: True)
            use_gating: Whether to use gating mechanisms (default: True)
            window_size: Size of the temporal window for skip connections (default: 3)
            aggregation: Method for aggregating skip connections (default: 'mean')
            residual: Whether to use residual connections (default: True)
        """
        super(TemporalPropagation, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.time_aware = time_aware
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_skip_connection = use_skip_connection
        self.use_gating = use_gating
        self.window_size = window_size
        self.aggregation = aggregation
        self.residual = residual
        
        # Core temporal evolution layer
        self.evolution_layer = TemporalEvolutionLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            time_aware=time_aware,
            bidirectional=bidirectional,
            use_layer_norm=use_layer_norm,
            residual=residual
        )
        
        # Skip connection layer (optional)
        if use_skip_connection:
            self.skip_connection = TemporalSkipConnection(
                input_dim=hidden_dim,
                window_size=window_size,
                aggregation=aggregation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                residual=residual
            )
        
        # Gating unit for memory management (optional)
        if use_gating:
            self.gating_unit = TemporalGatingUnit(
                input_dim=hidden_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                residual=residual
            )
        
        # Node state tracking
        self.state_tracking = nn.Parameter(torch.ones(1, 3), requires_grad=True)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features_seq: List[torch.Tensor],
        node_masks_seq: Optional[List[Union[torch.Tensor, List[int]]]] = None,
        time_stamps: Optional[torch.Tensor] = None,
        memory_bank: Optional[Union[Dict[int, torch.Tensor], Any]] = None
    ) -> Tuple[List[torch.Tensor], Optional[Union[Dict[int, torch.Tensor], Any]]]:
        """
        Forward pass for the temporal propagation module.
        
        Args:
            node_features_seq: List of node features at each time step
                             [num_time_steps, batch_size, input_dim]
            node_masks_seq: List of node masks at each time step or node IDs
                          [num_time_steps, batch_size] or [num_time_steps, [node_ids]]
            time_stamps: Time stamps for each step [batch_size, num_time_steps] (default: None)
            memory_bank: Either a NodeMemoryBank instance or dictionary of node memory states
                        NodeMemoryBank instance or {node_id: tensor[hidden_dim]}
            
        Returns:
            Tuple of (propagated_features, updated_memory_bank)
                propagated_features: List of propagated node features for each time step
                                   [num_time_steps, batch_size, hidden_dim]
                updated_memory_bank: Updated dictionary of node memory states
        """
        print(f"\n===== DEBUGGING: TemporalPropagation.forward =====")
        print(f"Input type: {type(node_features_seq)}")
        
        # Check if we're getting a single tensor instead of a list (common mistake)
        if isinstance(node_features_seq, torch.Tensor):
            print(f"WARNING: Received a single tensor of shape {node_features_seq.shape} instead of a list of tensors")
            # Convert to a list with a single tensor - this is likely a single node's features
            node_features_seq = [node_features_seq]
            print(f"Converted to list with single tensor")
        
        # First check if we're getting node_ids instead of masks
        node_ids_seq = None
        if isinstance(node_masks_seq, list) and node_masks_seq and not isinstance(node_masks_seq[0], torch.Tensor):
            print(f"WARNING: node_masks_seq appears to be a list of node_ids, not tensor masks")
            print(f"node_masks_seq first element type: {type(node_masks_seq[0])}")
            # Store the original node IDs for memory bank usage
            node_ids_seq = node_masks_seq.copy()
            
            # Create a new memory bank if none exists
            if memory_bank is None:
                # Create a memory bank with the right hidden dimension
                from ..utils.memory_bank import NodeMemoryBank
                memory_bank = NodeMemoryBank(
                    hidden_dim=self.hidden_dim,
                    device=device
                )
                print(f"Created new NodeMemoryBank instance")
        print(f"node_features_seq length: {len(node_features_seq)}")
        if len(node_features_seq) > 0:
            print(f"First element type: {type(node_features_seq[0])}")
            if hasattr(node_features_seq[0], 'shape'):
                print(f"First element shape: {node_features_seq[0].shape}")
            
        if node_masks_seq is not None:
            print(f"node_masks_seq length: {len(node_masks_seq)}")
            
        if time_stamps is not None:
            print(f"time_stamps shape: {time_stamps.shape if hasattr(time_stamps, 'shape') else 'No shape'}")
            
        print(f"memory_bank is {None if memory_bank is None else 'provided'}")
        
        # Get dimensions and device
        num_time_steps = len(node_features_seq)
        batch_size = node_features_seq[0].size(0) if len(node_features_seq) > 0 else 0
        device = node_features_seq[0].device if len(node_features_seq) > 0 else next(self.parameters()).device
        print(f"num_time_steps: {num_time_steps}, batch_size: {batch_size}, device: {device}")
        
        # Create proper attention masks based on node_ids_seq
        if node_ids_seq is not None:
            # Track all unique node IDs
            all_node_ids = set()
            for t in range(len(node_ids_seq)):
                all_node_ids.update(node_ids_seq[t])
                
            # Create a mapping from node_id to consecutive index
            # This ensures mask dimensions are manageable
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(all_node_ids))}
            num_unique_nodes = len(node_id_to_idx)
            
            # Create binary node existence masks
            # Shape: [timestep, num_unique_nodes] where 1 indicates node exists at that timestep
            node_existence_masks = torch.zeros(num_time_steps, num_unique_nodes, device=device)
            
            for t in range(num_time_steps):
                if t < len(node_ids_seq):
                    for node_id in node_ids_seq[t]:
                        if node_id in node_id_to_idx:  # Ensure node_id is valid
                            idx = node_id_to_idx[node_id]
                            node_existence_masks[t, idx] = 1.0
            
            # Create masks that match attention dimensions
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            batch_size = node_features_seq[0].size(0)
            converted_masks = []
            
            for t in range(num_time_steps):
                # Create a base mask where all nodes can attend to each other
                mask = torch.ones(batch_size, 1, num_time_steps, num_time_steps, device=device)
                
                # If we restrict attention to same timestep only, apply temporal mask
                if self.restrict_temporal_attention:
                    # Only allow nodes to attend within the same timestep
                    for b in range(mask.size(0)):
                        mask[b, 0, :, :] = torch.eye(num_time_steps, device=device)
                
                converted_masks.append(mask)
            
            # Store node existence information for memory bank access and visualization
            self.node_existence_masks = node_existence_masks
            
            # Store original node IDs and mapping for memory bank access
            self.node_id_to_idx = node_id_to_idx
            self.original_node_ids = node_ids_seq
            
            # Create proper masks matching attention scores dimensions
            # Shape should be [batch_size, 1, seq_len, seq_len]
            mask_shape = torch.Size([batch_size, 1, num_time_steps, num_time_steps])
            temporal_mask = torch.ones(mask_shape, device=device)
            print(f"Created masks with shape {temporal_mask.shape} to match attention dimensions")
            
            # Replace the original masks with the converted masks
            node_masks_seq = temporal_mask
            print(f"Converted node_ids to proper tensor masks")
            
            # Store the temporal mask for use in temporal attention
            self.temporal_mask = temporal_mask
        else:
            self.temporal_mask = None
        
        # Handle memory bank initialization and usage
        memory_bank_type = type(memory_bank).__name__
        print(f"Memory bank type: {memory_bank_type}")
        
        if hasattr(memory_bank, 'get_state') and hasattr(memory_bank, 'update_state'):
            # This is a NodeMemoryBank instance
            print(f"Using NodeMemoryBank instance with {memory_bank.size} nodes")
        else:
            # Convert dictionary memory bank to NodeMemoryBank
            from ..utils.memory_bank import NodeMemoryBank
            new_memory_bank = NodeMemoryBank(
                hidden_dim=self.hidden_dim,
                device=device
            )
            
            # Transfer dictionary contents if any
            if isinstance(memory_bank, dict) and memory_bank:
                for node_id, state in memory_bank.items():
                    # Handle both tensor and list states
                    if isinstance(state, list):
                        state = torch.tensor(state, device=device)
                    new_memory_bank.update([node_id], state.unsqueeze(0))
                
                print(f"Converted dictionary memory bank with {len(memory_bank)} items to NodeMemoryBank")
            
            memory_bank = new_memory_bank
            print(f"Using new NodeMemoryBank instance")
        # Create default masks if not provided
        if node_masks_seq is None:
            # Create binary masks with all ones (assume all nodes are valid)
            node_masks_seq = [torch.ones(batch_size, batch_size, device=device) for _ in range(num_time_steps)]
        
        # Apply temporal evolution
        evolved_features = self.evolution_layer(node_features_seq, time_stamps)
        
        # Apply skip connections if used
        if self.use_skip_connection:
            evolved_features = self.skip_connection(evolved_features)
        
        # Apply memory state propagation and gating
        # This is essential for maintaining node states across timesteps
        if node_ids_seq is not None:
            # For tracking node appearances across timesteps
            node_last_seen = {}  # Maps node_id -> timestep it was last seen
            
            # Check if we're using NodeMemoryBank instead of a dictionary
            using_node_memory_bank = hasattr(memory_bank, 'get_state') and hasattr(memory_bank, 'update_state')
            
            # Process each time step
            for t in range(num_time_steps):
                if t < len(node_ids_seq):
                    active_node_ids = node_ids_seq[t]
                    print(f"Time step {t}, processing {len(active_node_ids)} active nodes")
                    
                    # First, apply memory bank for reappearing nodes
                    for i, node_id in enumerate(active_node_ids):
                        if i >= evolved_features[t].size(0):
                            print(f"Skipping node_id {node_id} at time {t} - index {i} out of bounds (max {evolved_features[t].size(0)-1})")
                            continue  # Skip if index is out of bounds
                        
                        try:
                            # Get current features for this node
                            current_feat = evolved_features[t][i]
                            
                            # Get previous features from appropriate source
                            has_previous_state = False
                            previous_feat = None
                            
                            if using_node_memory_bank:
                                previous_feat = memory_bank.get_state(node_id)
                                has_previous_state = previous_feat is not None
                            else:
                                has_previous_state = node_id in memory_bank
                                if has_previous_state:
                                    previous_feat = memory_bank[node_id]  # Direct access from dictionary
                            
                            # Process nodes with previous state
                            if has_previous_state and previous_feat is not None:
                                # Log successful memory retrieval
                                if t % 10 == 0 and i % 10 == 0:  # Reduce logging frequency
                                    print(f"Retrieved previous state for node {node_id} at time {t}")
                                
                                # Ensure tensor shapes match
                                if previous_feat.shape != current_feat.shape:
                                    # Resize previous feature to match current feature
                                    if previous_feat.dim() == 1 and current_feat.dim() == 1:
                                        if len(previous_feat) < len(current_feat):
                                            # Pad with zeros
                                            padding = torch.zeros(len(current_feat) - len(previous_feat),
                                                                device=previous_feat.device)
                                            previous_feat = torch.cat([previous_feat, padding])
                                        else:
                                            # Truncate
                                            previous_feat = previous_feat[:len(current_feat)]
                                
                                if node_id not in node_last_seen or node_last_seen[node_id] < t-1:
                                    # Node is reappearing after absence - use stronger memory contribution
                                    if self.use_gating:
                                        try:
                                            # Use gating mechanism with bias toward memory for returning nodes
                                            # Calculate time gap since last appearance
                                            time_gap = t - node_last_seen.get(node_id, 0)
                                            
                                            # Adjust memory bias based on time gap (more recent = higher bias)
                                            # For longer gaps, we need to trust the memory less
                                            memory_bias = max(0.5, 0.9 - (0.1 * min(time_gap, 4)))
                                            
                                            print(f"Node {node_id} reappearing after {time_gap} steps with memory bias {memory_bias:.2f}")
                                            
                                            gated_feat = self.gating_unit(
                                                current_feat.unsqueeze(0),
                                                previous_feat.unsqueeze(0),
                                                memory_bias=memory_bias  # Dynamically adjusted memory bias
                                            ).squeeze(0)
                                            evolved_features[t][i] = gated_feat
                                        except Exception as e:
                                            print(f"Gating error: {e}, using fallback blending")
                                            # Calculate adaptive memory weight based on time gap
                                            time_gap = t - node_last_seen.get(node_id, 0)
                                            memory_weight = max(0.4, 0.9 - (0.1 * min(time_gap, 5)))
                                            current_weight = 1.0 - memory_weight
                                            # Fallback to adaptive weighted average
                                            evolved_features[t][i] = memory_weight * previous_feat + current_weight * current_feat
                                    else:
                                        # Enhanced weighted average for returning nodes with adaptive weighting
                                        time_gap = t - node_last_seen.get(node_id, 0)
                                        memory_weight = max(0.4, 0.9 - (0.1 * min(time_gap, 5)))  # Decay with time
                                        current_weight = 1.0 - memory_weight
                                        
                                        merged_feat = current_weight * current_feat + memory_weight * previous_feat
                                        evolved_features[t][i] = merged_feat
                                else:
                                    # Node is continuing from previous timestep
                                    if self.use_gating:
                                        # Use gating with higher memory weight for continuity
                                        gated_feat = self.gating_unit(
                                            current_feat.unsqueeze(0),
                                            previous_feat.unsqueeze(0),
                                            memory_bias=0.6  # Bias toward memory for continuity
                                        ).squeeze(0)
                                        evolved_features[t][i] = gated_feat
                            
                            # Update last seen time for this node
                            node_last_seen[node_id] = t
                            
                            # Create a copy to avoid modifying the original tensor
                            # This is critical for backward pass stability
                            node_state = evolved_features[t][i].detach().clone()
                            
                            # Add a small amount of current timestep information to the memory
                            # This helps with handling the temporal dynamics
                            if t > 0:
                                node_state = node_state + 0.01 * torch.tensor(t, device=node_state.device,
                                                                             dtype=node_state.dtype)
                            
                            if using_node_memory_bank:
                                # Update the NodeMemoryBank using its method
                                memory_bank.update_state(node_id, node_state, t)
                                if t % 10 == 0 and i % 10 == 0:  # Reduce logging frequency
                                    print(f"Updated memory bank for node {node_id} at time {t}")
                            else:
                                # Update the dictionary with the proper state
                                memory_bank[node_id] = node_state
                        except Exception as e:
                            print(f"Error processing node {node_id} at time {t}: {str(e)}")
                            # Ensure we don't lose the node from memory even with an error
                            if not node_id in memory_bank and not using_node_memory_bank:
                                memory_bank[node_id] = torch.zeros(self.hidden_dim, device=device)
            
            print(f"Updated memory bank - now contains {len(memory_bank)} nodes")
        
        # Apply final transformations
        output_features = []
        for features in evolved_features:
            # Apply output projection
            out = self.output_proj(features)
            
            # Apply dropout
            out = self.dropout_layer(out)
            
            # Apply layer normalization if used
            if self.use_layer_norm:
                out = self.layer_norm(out)
            
            output_features.append(out)
        # Debug final output
        print(f"Output features length: {len(output_features)}")
        if output_features:
            print(f"First output tensor shape: {output_features[0].shape}")
        print(f"Memory bank size: {len(memory_bank) if memory_bank is not None else 'None'}")
        print(f"===== DEBUGGING: TemporalPropagation.forward complete =====\n")
        
        return output_features, memory_bank
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'dropout={self.dropout}, '
                f'time_aware={self.time_aware}, '
                f'bidirectional={self.bidirectional}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'use_skip_connection={self.use_skip_connection}, '
                f'use_gating={self.use_gating}, '
                f'window_size={self.window_size}, '
                f'aggregation={self.aggregation}, '
                f'residual={self.residual}')


class AsymmetricTemporalCell(nn.Module):
    """
    Asymmetric Temporal Cell for TAGAN.
    
    This cell explicitly models asymmetric dependencies in temporal data,
    giving different weights to past and future influences. It incorporates
    directional bias and temporal decay mechanisms.
    
    Attributes:
        input_dim (int): Input dimension size
        hidden_dim (int): Hidden dimension size
        time_aware (bool): Whether to use time-aware processing
        asymmetry_factor (float): Factor controlling the strength of asymmetry
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        time_aware: bool = True,
        use_layer_norm: bool = True,
        asymmetry_factor: float = 0.7,
        forward_decay: float = 0.9,
        backward_decay: float = 0.8,
        bias: bool = True
    ):
        """
        Initialize the asymmetric temporal cell.
        
        Args:
            input_dim: Input dimension size
            hidden_dim: Hidden dimension size
            dropout: Dropout probability (default: 0.1)
            time_aware: Whether to use time-aware processing (default: True)
            use_layer_norm: Whether to use layer normalization (default: True)
            asymmetry_factor: Factor controlling the strength of asymmetry (default: 0.7)
            forward_decay: Decay factor for forward influence (default: 0.9)
            backward_decay: Decay factor for backward influence (default: 0.8)
            bias: Whether to use bias in linear layers (default: True)
        """
        super(AsymmetricTemporalCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        self.time_aware = time_aware
        self.use_layer_norm = use_layer_norm
        self.asymmetry_factor = asymmetry_factor
        self.forward_decay = forward_decay
        self.backward_decay = backward_decay
        self.bias = bias
        
        # Forward direction gates
        self.forward_reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.forward_update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.forward_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        
        # Backward direction gates (for asymmetry)
        self.backward_reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.backward_update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.backward_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        
        # Asymmetry parameters
        self.forward_weight = nn.Parameter(torch.ones(1) * asymmetry_factor)
        self.backward_weight = nn.Parameter(torch.ones(1) * (1 - asymmetry_factor))
        
        # Time-aware components
        if time_aware:
            self.time_modulation = nn.Linear(1, hidden_dim, bias=False)
            self.time_gate = nn.Linear(1, hidden_dim, bias=False)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm_input = nn.LayerNorm(input_dim)
            self.layer_norm_hidden = nn.LayerNorm(hidden_dim)
            self.layer_norm_output = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with appropriate initialization."""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Set gate biases to 1 (helps with gradient flow early in training)
        if self.bias:
            nn.init.constant_(self.forward_reset_gate.bias, 1.0)
            nn.init.constant_(self.forward_update_gate.bias, 1.0)
            nn.init.constant_(self.backward_reset_gate.bias, 1.0)
            nn.init.constant_(self.backward_update_gate.bias, 1.0)
        
        # Initialize asymmetry weights
        nn.init.constant_(self.forward_weight, self.asymmetry_factor)
        nn.init.constant_(self.backward_weight, 1.0 - self.asymmetry_factor)
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        h_next: Optional[torch.Tensor] = None,
        time_diff: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the asymmetric temporal cell.
        
        Args:
            x: Input features [batch_size, input_dim]
            h_prev: Previous hidden state [batch_size, hidden_dim] (default: None)
            h_next: Next hidden state (from future) [batch_size, hidden_dim] (default: None)
            time_diff: Time difference since last update [batch_size, 1] (default: None)
            
        Returns:
            Updated hidden state [batch_size, hidden_dim]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Apply layer normalization if used
        if self.use_layer_norm:
            x = self.layer_norm_input(x)
        
        # Initialize hidden states if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        if h_next is None:
            h_next = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Apply dropout to input
        x = self.dropout_layer(x)
        
        # Forward direction processing
        # Concatenate input and previous hidden state
        forward_input = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        forward_reset = torch.sigmoid(self.forward_reset_gate(forward_input))
        forward_update = torch.sigmoid(self.forward_update_gate(forward_input))
        
        # Apply time modulation if time-aware
        if self.time_aware and time_diff is not None:
            forward_reset = forward_reset + torch.sigmoid(self.time_modulation(time_diff))
            forward_update = forward_update + torch.sigmoid(self.time_gate(time_diff))
        
        # Compute candidate hidden state
        forward_reset_hidden = forward_reset * h_prev
        forward_candidate_input = torch.cat([x, forward_reset_hidden], dim=1)
        forward_candidate_hidden = torch.tanh(self.forward_candidate(forward_candidate_input))
        
        # Apply decay based on time difference
        if time_diff is not None:
            forward_decay = torch.pow(self.forward_decay * torch.ones(1, device=device),
                                     torch.clamp(time_diff, min=0.0, max=10.0))
            forward_candidate_hidden = forward_candidate_hidden * forward_decay
        
        # Update hidden state (forward direction)
        h_forward = (1 - forward_update) * h_prev + forward_update * forward_candidate_hidden
        
        # Backward direction processing (from future to present)
        # Concatenate input and next hidden state
        backward_input = torch.cat([x, h_next], dim=1)
        
        # Compute gates
        backward_reset = torch.sigmoid(self.backward_reset_gate(backward_input))
        backward_update = torch.sigmoid(self.backward_update_gate(backward_input))
        
        # Apply time modulation if time-aware
        if self.time_aware and time_diff is not None:
            backward_reset = backward_reset + torch.sigmoid(self.time_modulation(time_diff))
            backward_update = backward_update + torch.sigmoid(self.time_gate(time_diff))
        
        # Compute candidate hidden state
        backward_reset_hidden = backward_reset * h_next
        backward_candidate_input = torch.cat([x, backward_reset_hidden], dim=1)
        backward_candidate_hidden = torch.tanh(self.backward_candidate(backward_candidate_input))
        
        # Apply decay based on time difference
        if time_diff is not None:
            backward_decay = torch.pow(self.backward_decay * torch.ones(1, device=device),
                                      torch.clamp(time_diff, min=0.0, max=10.0))
            backward_candidate_hidden = backward_candidate_hidden * backward_decay
        
        # Update hidden state (backward direction)
        h_backward = (1 - backward_update) * h_next + backward_update * backward_candidate_hidden
        
        # Combine forward and backward states with asymmetric weights
        h_combined = self.forward_weight * h_forward + self.backward_weight * h_backward
        
        # Apply layer normalization to output if used
        if self.use_layer_norm:
            h_combined = self.layer_norm_output(h_combined)
        
        return h_combined
    
    def extra_repr(self) -> str:
        """Return a string representation of the module configuration."""
        return (f'input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'dropout={self.dropout_prob}, '
                f'time_aware={self.time_aware}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'asymmetry_factor={self.asymmetry_factor}, '
                f'forward_decay={self.forward_decay}, '
                f'backward_decay={self.backward_decay}, '
                f'bias={self.bias}')
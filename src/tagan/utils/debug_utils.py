"""
TAGAN Debugging Utilities.

This module contains utilities for debugging and diagnosing issues in TAGAN models.
It provides tools for:
1. Memory tracking and optimization
2. Shape checking and validation
3. Attention pattern visualization
4. Performance monitoring
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
from datetime import datetime
import gc


class TAGANDebugger:
    """
    Debugger for TAGAN models to help identify and fix issues.
    
    This class provides methods for:
    - Tracking tensor shapes and types
    - Memory usage monitoring
    - Attention pattern visualization
    - Execution time profiling
    - Data flow validation
    """
    
    def __init__(self, enabled=True, log_file='./debug_output/tagan_debug.log'):
        """
        Initialize the TAGAN debugger.
        
        Args:
            enabled: Whether debugging is enabled
            log_file: Path to debug log file
        """
        self.enabled = enabled
        self.log_file = log_file
        self.step_timers = {}
        self.step_memory = {}
        
        # Ensure debug directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Clear log file
        if enabled:
            with open(log_file, 'w') as f:
                f.write(f"TAGAN Debug Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n\n")
    
    def log(self, message, level="INFO", print_to_console=True):
        """Log a debug message."""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        if print_to_console:
            print(log_message)
            
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def start_timer(self, step_name):
        """Start timing a step."""
        if not self.enabled:
            return
            
        self.step_timers[step_name] = time.time()
        self.log(f"Starting step: {step_name}", level="TIMER", print_to_console=False)
        
        # Track memory before step
        self.step_memory[f"{step_name}_start"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    def end_timer(self, step_name):
        """End timing a step and report duration."""
        if not self.enabled or step_name not in self.step_timers:
            return
            
        duration = time.time() - self.step_timers[step_name]
        
        # Track memory after step
        current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_diff = current_memory - self.step_memory.get(f"{step_name}_start", 0)
        
        memory_str = ""
        if torch.cuda.is_available():
            memory_str = f" | Memory change: {memory_diff/1024**2:.2f} MB"
        
        self.log(f"Completed step: {step_name} in {duration:.4f} seconds{memory_str}", level="TIMER")
        return duration
    
    def check_tensor(self, tensor, name="tensor", check_nan=True, check_inf=True):
        """Check tensor for NaN or Inf values and report shape/type."""
        if not self.enabled:
            return True
            
        if not isinstance(tensor, torch.Tensor):
            self.log(f"{name} is not a tensor, but {type(tensor)}", level="WARNING")
            return False
            
        self.log(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}", level="TENSOR")
        
        issues = []
        
        if check_nan and torch.isnan(tensor).any():
            issues.append("contains NaN values")
            
        if check_inf and torch.isinf(tensor).any():
            issues.append("contains Inf values")
            
        if issues:
            self.log(f"WARNING: {name} {', '.join(issues)}", level="ERROR")
            # Get statistics on the tensor
            tensor_stats = {
                'min': tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].min().item() if tensor.numel() > 0 else None,
                'max': tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].max().item() if tensor.numel() > 0 else None,
                'mean': tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].mean().item() if tensor.numel() > 0 else None,
                'std': tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].std().item() if tensor.numel() > 0 else None
            }
            self.log(f"Tensor stats: {tensor_stats}", level="DEBUG")
            return False
            
        return True
    
    def visualize_attention(self, attention_weights, title="Attention Weights", save_path=None):
        """Visualize attention weights for debugging."""
        if not self.enabled:
            return
            
        # Handle different attention formats
        if isinstance(attention_weights, list):
            if len(attention_weights) == 0:
                self.log("Empty attention weights list", level="WARNING")
                return
                
            # Try to concatenate multiple attention matrices
            if all(isinstance(w, torch.Tensor) for w in attention_weights):
                try:
                    attention_weights = torch.cat(attention_weights, dim=0)
                except:
                    attention_weights = attention_weights[0]
            else:
                attention_weights = attention_weights[0]
                
        # Convert to numpy for visualization
        if isinstance(attention_weights, torch.Tensor):
            # Average over batch and heads if needed
            if attention_weights.dim() == 4:
                attention_weights = attention_weights.mean(dim=(0, 1)).cpu().detach().numpy()
            elif attention_weights.dim() == 3:
                attention_weights = attention_weights.mean(dim=0).cpu().detach().numpy()
            else:
                attention_weights = attention_weights.cpu().detach().numpy()
        else:
            self.log(f"Unsupported attention weights type: {type(attention_weights)}", level="ERROR")
            return
            
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.log(f"Saved attention visualization to {save_path}", level="INFO")
        else:
            plt.show()
            
        plt.close()
    
    def check_memory_usage(self):
        """Check current memory usage and report statistics."""
        if not self.enabled:
            return
            
        gc.collect()
        
        # Report memory usage statistics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            
            self.log(f"CUDA Memory: allocated={allocated:.2f} MB, reserved={reserved:.2f} MB, peak={max_allocated:.2f} MB", level="MEMORY")
            
        # Report Python memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.log(f"Process Memory: RSS={memory_info.rss/1024**2:.2f} MB, VMS={memory_info.vms/1024**2:.2f} MB", level="MEMORY")
        except:
            self.log("Could not get process memory info. Install psutil for full memory tracking.", level="WARNING")
    
    def inspect_sequence(self, graph_sequence, max_items=3):
        """Inspect a graph sequence and verify its format."""
        if not self.enabled:
            return
            
        self.log(f"Graph sequence: length={len(graph_sequence)}", level="SEQUENCE")
        
        # Inspect the first few items
        for i, snapshot in enumerate(graph_sequence[:max_items]):
            if isinstance(snapshot, dict):
                keys = list(snapshot.keys())
                self.log(f"  Snapshot {i}: dict with keys {keys}", level="SEQUENCE")
                
                # Check core components
                for key in ['x', 'edge_index', 'node_ids']:
                    if key in snapshot:
                        value = snapshot[key]
                        if isinstance(value, torch.Tensor):
                            self.log(f"    {key}: tensor of shape {value.shape}", level="SEQUENCE")
                        else:
                            self.log(f"    {key}: {type(value)} of length {len(value) if hasattr(value, '__len__') else 'N/A'}", level="SEQUENCE")
                    else:
                        self.log(f"    Missing key: {key}", level="WARNING")
                        
            elif isinstance(snapshot, tuple):
                self.log(f"  Snapshot {i}: tuple of length {len(snapshot)}", level="SEQUENCE")
                
                # Check tuple components assuming standard format
                if len(snapshot) >= 4:
                    # Typically (node_features, edge_index, edge_attr, node_ids)
                    components = ["node_features", "edge_index", "edge_attr", "node_ids"]
                    for j, (name, value) in enumerate(zip(components, snapshot[:4])):
                        if isinstance(value, torch.Tensor):
                            self.log(f"    {name}: tensor of shape {value.shape}", level="SEQUENCE")
                        elif value is None:
                            self.log(f"    {name}: None", level="SEQUENCE")
                        else:
                            self.log(f"    {name}: {type(value)} of length {len(value) if hasattr(value, '__len__') else 'N/A'}", level="SEQUENCE")
            else:
                self.log(f"  Snapshot {i}: unsupported type {type(snapshot)}", level="WARNING")
                
        if len(graph_sequence) > max_items:
            self.log(f"  ... and {len(graph_sequence) - max_items} more snapshots", level="SEQUENCE")
    
    def profile_forward_pass(self, model, sample_input, labels=None):
        """Profile a forward pass through the model."""
        if not self.enabled:
            return
            
        self.log("Profiling model forward pass...", level="PROFILE")
        
        # Check input format
        self.inspect_sequence(sample_input)
        
        # Track memory before forward pass
        self.check_memory_usage()
        
        # Time the forward pass
        self.start_timer("forward_pass")
        try:
            with torch.no_grad():
                outputs = model(sample_input, labels=labels)
            
            # Check outputs
            self.log("Forward pass successful", level="PROFILE")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    self.check_tensor(value, name=f"outputs['{key}']")
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    self.log(f"outputs['{key}']: list of {len(value)} tensors", level="TENSOR")
                    self.check_tensor(value[0], name=f"outputs['{key}'][0]")
                else:
                    self.log(f"outputs['{key}']: {type(value)}", level="TENSOR")
        except Exception as e:
            self.log(f"Forward pass failed: {str(e)}", level="ERROR")
            self.log(traceback.format_exc(), level="ERROR")
            
        duration = self.end_timer("forward_pass")
        
        # Check memory after forward pass
        self.check_memory_usage()
        
        return duration


# Singleton debugger instance
_debugger = TAGANDebugger(enabled=False)

def get_debugger(enabled=None, log_file=None):
    """Get or configure the global debugger instance."""
    global _debugger
    
    if enabled is not None:
        _debugger.enabled = enabled
        
    if log_file is not None:
        _debugger.log_file = log_file
        
    return _debugger

def enable_debugging(log_file='./debug_output/tagan_debug.log'):
    """Enable debugging with specified log file."""
    return get_debugger(enabled=True, log_file=log_file)

def disable_debugging():
    """Disable debugging."""
    return get_debugger(enabled=False)


# Convenient tensor checking function
def check_tensor(tensor, name="tensor", check_nan=True, check_inf=True):
    """Check a tensor for issues and log the results."""
    return get_debugger().check_tensor(tensor, name, check_nan, check_inf)


# Visualization utility function
def plot_attention_patterns(geometric_attention, title="Geometric Attention", figsize=(12, 8), 
                           save_path=None, show_plot=True):
    """
    Plot attention patterns from the geometric attention module.
    
    Args:
        geometric_attention: List of attention weights or tensors
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
    """
    # Create figure and subplots based on number of attention heads/layers
    if isinstance(geometric_attention, list):
        num_plots = len(geometric_attention)
    else:
        num_plots = 1
        geometric_attention = [geometric_attention]
        
    # Determine grid layout
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each attention pattern
    for i, attn in enumerate(geometric_attention[:num_plots]):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Handle different attention formats
        if isinstance(attn, dict) and 'attention' in attn:
            attn_weights = attn['attention']
        elif isinstance(attn, torch.Tensor):
            attn_weights = attn
        else:
            # Skip if format is not recognized
            ax.text(0.5, 0.5, f"Unsupported format: {type(attn)}", 
                    ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Convert to numpy array for plotting
        if isinstance(attn_weights, torch.Tensor):
            # Handle different tensor dimensions
            if attn_weights.dim() == 3:  # [batch, seq_len, seq_len]
                # Use first batch
                attn_weights = attn_weights[0].detach().cpu().numpy()
            elif attn_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
                # Average over heads
                attn_weights = attn_weights[0].mean(dim=0).detach().cpu().numpy()
            else:
                attn_weights = attn_weights.detach().cpu().numpy()
        
        # Create heatmap
        im = ax.imshow(attn_weights, aspect='auto', cmap='viridis')
        ax.set_title(f"Attention Head {i+1}")
        ax.set_xlabel("Node index")
        ax.set_ylabel("Node index")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_temporal_graph_attention(geometric_attention, graph_data, title="Temporal Graph Attention",
                                 figsize=(16, 10), save_path=None, show_plot=True):
    """
    Create a more sophisticated visualization of temporal graph attention.
    
    Args:
        geometric_attention: List of attention weights
        graph_data: List of dictionaries with 'edge_index' and 'node_ids'
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    num_timesteps = len(graph_data)
    
    # Create a grid of plots
    rows = (num_timesteps + 1) // 2
    cols = min(2, num_timesteps)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each timestep
    for t, (attn, graph) in enumerate(zip(geometric_attention[:num_timesteps], graph_data[:num_timesteps])):
        if t >= len(axes):
            break
            
        ax = axes[t]
        edge_index = graph['edge_index']
        
        # Handle different attention formats
        if isinstance(attn, dict) and 'attention' in attn:
            attn_weights = attn['attention']
        elif isinstance(attn, torch.Tensor):
            attn_weights = attn
        else:
            attn_weights = None
            
        # Convert edge_index to CPU numpy
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
            
        # Get number of nodes
        num_nodes = edge_index.max() + 1 if edge_index.size > 0 else 0
        
        # Create a simple network visualization
        pos = {}
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            pos[i] = (np.cos(angle), np.sin(angle))
            
        # Draw the graph structure
        for e in range(edge_index.shape[1]):
            src, dst = edge_index[0, e], edge_index[1, e]
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            
            # Determine edge color based on attention
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                if attn_weights.dim() >= 2 and attn_weights.shape[0] > src and attn_weights.shape[1] > dst:
                    # Get attention weight for this edge
                    weight = attn_weights[src, dst].item() if attn_weights.dim() == 2 else attn_weights[0, src, dst].item()
                    color = plt.cm.plasma(weight)
                    width = 1 + weight * 3
                else:
                    color = 'gray'
                    width = 1
            else:
                color = 'gray'
                width = 1
                
            # Draw the edge with appropriate width
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.6)
            
        # Draw the nodes
        node_sizes = []
        node_colors = []
        
        for i in range(num_nodes):
            # Determine node size and color based on attention
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                if attn_weights.dim() >= 2 and i < attn_weights.shape[0]:
                    # Use row-wise mean as node importance
                    importance = attn_weights[i].mean().item() if attn_weights.dim() == 2 else attn_weights[0, i].mean().item()
                    size = 100 + importance * 500
                    color = plt.cm.plasma(importance)
                else:
                    size = 100
                    color = 'skyblue'
            else:
                size = 100
                color = 'skyblue'
                
            node_sizes.append(size)
            node_colors.append(color)
            
        # Plot all nodes at once for efficiency
        x_values = [pos[i][0] for i in range(num_nodes)]
        y_values = [pos[i][1] for i in range(num_nodes)]
        
        ax.scatter(x_values, y_values, s=node_sizes, c=node_colors, zorder=10)
        
        # Add node labels
        for i in range(num_nodes):
            ax.text(pos[i][0], pos[i][1], str(i), fontsize=9, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                   zorder=11)
        
        # Set plot appearance
        ax.set_title(f"Timestep {t}")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
    # Hide unused subplots
    for i in range(num_timesteps, len(axes)):
        axes[i].axis('off')
        
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Attention Weight')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close()
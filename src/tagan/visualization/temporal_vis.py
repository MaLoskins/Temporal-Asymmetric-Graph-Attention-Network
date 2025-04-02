"""
Temporal visualization tools for TAGAN.

This module provides functions for visualizing temporal aspects of graph data,
including temporal graph evolution and time-series patterns.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def visualize_temporal_graph(
    node_coords: List[torch.Tensor],
    edge_indices: List[torch.Tensor],
    node_features: Optional[List[torch.Tensor]] = None,
    edge_features: Optional[List[torch.Tensor]] = None,
    node_labels: Optional[List[List[int]]] = None,
    timestamps: Optional[List[float]] = None,
    title: str = "Temporal Graph Visualization",
    figsize: Tuple[int, int] = (15, 10),
    node_size: int = 100,
    edge_width: float = 1.0,
    colormap: str = "viridis",
    save_path: Optional[str] = None,
    show_animation: bool = True,
    fps: int = 2
) -> plt.Figure:
    """
    Visualize a temporal graph as an animation or sequence of plots.
    
    Args:
        node_coords: List of node coordinate tensors [num_timesteps, [num_nodes, 2]]
        edge_indices: List of edge index tensors [num_timesteps, [2, num_edges]]
        node_features: Optional list of node feature tensors [num_timesteps, [num_nodes, feature_dim]]
        edge_features: Optional list of edge feature tensors [num_timesteps, [num_edges, feature_dim]]
        node_labels: Optional list of node labels [num_timesteps, [num_nodes]]
        timestamps: Optional list of timestamps for each frame
        title: Title for the visualization
        figsize: Figure size (width, height)
        node_size: Size of nodes in the plot
        edge_width: Width of edges in the plot
        colormap: Colormap for node colors
        save_path: Optional path to save the animation (requires ffmpeg)
        show_animation: Whether to show the animation
        fps: Frames per second for the animation
        
    Returns:
        Matplotlib figure object
    """
    if not node_coords or not edge_indices:
        raise ValueError("Node coordinates and edge indices must be provided")
    
    if len(node_coords) != len(edge_indices):
        raise ValueError("Number of timesteps must match between node_coords and edge_indices")
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    cmap = plt.cm.get_cmap(colormap)
    
    # Number of timesteps
    num_timesteps = len(node_coords)
    
    # Set up timestamps or default time values
    if timestamps is None:
        timestamps = list(range(num_timesteps))
    
    # Function to create graph for a specific timestep
    def create_graph(t):
        # Extract data for current timestep
        coords = node_coords[t]
        edges = edge_indices[t]
        
        # Convert tensors to numpy arrays
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        if isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        for i in range(coords.shape[0]):
            G.add_node(i, pos=(coords[i, 0], coords[i, 1]))
            
        # Add edges
        for j in range(edges.shape[1]):
            src, dst = edges[0, j], edges[1, j]
            if src < coords.shape[0] and dst < coords.shape[0]:  # Ensure valid indices
                G.add_edge(int(src), int(dst))
        
        return G
    
    # Function to draw graph for a specific timestep
    def draw_graph(t):
        ax.clear()
        
        # Create graph
        G = create_graph(t)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Determine node colors based on features or use default
        node_colors = None
        if node_features is not None and t < len(node_features):
            features = node_features[t]
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            
            # Use first feature dimension for color if multi-dimensional
            if features.ndim > 1 and features.shape[1] > 0:
                node_colors = features[:, 0]
            else:
                node_colors = features
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_size,
            node_color=node_colors,
            cmap=cmap,
            ax=ax
        )
        
        # Draw edges with width based on features if provided
        edge_weights = None
        if edge_features is not None and t < len(edge_features):
            e_features = edge_features[t]
            if isinstance(e_features, torch.Tensor):
                e_features = e_features.detach().cpu().numpy()
            
            # Use first feature dimension for edge width if multi-dimensional
            if e_features.ndim > 1 and e_features.shape[1] > 0:
                edge_weights = e_features[:, 0] * edge_width
            else:
                edge_weights = e_features * edge_width
        
        # Draw edges
        edges = nx.draw_networkx_edges(
            G, pos,
            width=edge_weights if edge_weights is not None else edge_width,
            arrowsize=10,
            arrowstyle='-|>',
            ax=ax
        )
        
        # Draw labels if provided
        if node_labels is not None and t < len(node_labels):
            labels = {i: str(l) for i, l in enumerate(node_labels[t])}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        
        # Set title with timestamp
        if timestamps:
            ax.set_title(f"{title} (Time: {timestamps[t]:.2f})")
        else:
            ax.set_title(f"{title} (Step: {t})")
            
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.axis('off')
        
        return nodes, edges
    
    # Create animation
    if show_animation or save_path:
        # Initialize with first frame
        nodes, edges = draw_graph(0)
        
        # Define update function for animation
        def update(frame):
            nodes, edges = draw_graph(frame)
            return nodes, edges
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=num_timesteps, 
            interval=1000//fps, blit=False
        )
        
        # Save animation if path provided
        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        
        # Show animation if requested
        if show_animation:
            plt.tight_layout()
            plt.show()
    else:
        # Create static plot for each timestep
        plt.figure(figsize=figsize)
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_timesteps)))
        
        for t in range(num_timesteps):
            # Create subplot
            plt.subplot(grid_size, grid_size, t + 1)
            
            # Create graph for this timestep
            G = create_graph(t)
            pos = nx.get_node_attributes(G, 'pos')
            
            # Determine node colors
            node_colors = None
            if node_features is not None and t < len(node_features):
                features = node_features[t]
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                
                if features.ndim > 1 and features.shape[1] > 0:
                    node_colors = features[:, 0]
                else:
                    node_colors = features
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_size//2,  # Smaller nodes for grid view
                node_color=node_colors,
                cmap=cmap
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                width=edge_width/2,  # Thinner edges for grid view
                arrowsize=5,
                arrowstyle='-|>'
            )
            
            # Set title for each subplot
            if timestamps:
                plt.title(f"t={timestamps[t]:.2f}")
            else:
                plt.title(f"Step {t}")
                
            plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save static plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show plot
        plt.show()
    
    return fig


def plot_node_feature_evolution(
    node_features: List[torch.Tensor],
    node_ids: Optional[List[int]] = None,
    feature_idx: Optional[Union[int, List[int]]] = None,
    timestamps: Optional[List[float]] = None,
    title: str = "Node Feature Evolution",
    figsize: Tuple[int, int] = (12, 8),
    colormap: str = "tab10",
    save_path: Optional[str] = None,
    include_legend: bool = True
) -> plt.Figure:
    """
    Plot the evolution of node features over time.
    
    Args:
        node_features: List of node feature tensors [num_timesteps, [num_nodes, feature_dim]]
        node_ids: Optional list of specific node IDs to plot (default: first 10 nodes)
        feature_idx: Index or list of indices of features to plot (default: first feature)
        timestamps: Optional list of timestamps
        title: Plot title
        figsize: Figure size (width, height)
        colormap: Colormap for node colors
        save_path: Optional path to save the figure
        include_legend: Whether to include a legend
        
    Returns:
        Matplotlib figure object
    """
    if not node_features:
        raise ValueError("Node features must be provided")
    
    # Convert tensors to numpy arrays
    features_np = []
    for features in node_features:
        if isinstance(features, torch.Tensor):
            features_np.append(features.detach().cpu().numpy())
        else:
            features_np.append(features)
    
    # Get number of nodes and timesteps
    num_timesteps = len(features_np)
    num_nodes = features_np[0].shape[0]
    num_features = features_np[0].shape[1] if features_np[0].ndim > 1 else 1
    
    # Set default node IDs if not provided
    if node_ids is None:
        node_ids = list(range(min(10, num_nodes)))
    
    # Set default feature index if not provided
    if feature_idx is None:
        feature_idx = 0
    
    # Convert feature_idx to list if it's a single value
    if isinstance(feature_idx, int):
        feature_idx = [feature_idx]
    
    # Set up timestamps or default time values
    if timestamps is None:
        timestamps = list(range(num_timesteps))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    cmap = plt.cm.get_cmap(colormap, len(node_ids))
    
    # Plot feature evolution for each selected node
    for i, node_id in enumerate(node_ids):
        if node_id >= num_nodes:
            continue
            
        # Extract feature values over time
        for f_idx in feature_idx:
            if f_idx >= num_features:
                continue
                
            values = []
            for t in range(num_timesteps):
                if node_id < features_np[t].shape[0]:
                    if features_np[t].ndim > 1:
                        values.append(features_np[t][node_id, f_idx])
                    else:
                        values.append(features_np[t][node_id])
                else:
                    # Handle missing nodes at some timesteps
                    values.append(np.nan)
            
            # Plot feature evolution
            feature_name = f"Feature {f_idx}" if len(feature_idx) > 1 else "Feature"
            ax.plot(
                timestamps[:len(values)], values, 
                marker='o',
                color=cmap(i) if len(feature_idx) == 1 else None,
                label=f"Node {node_id}, {feature_name}" if len(feature_idx) > 1 else f"Node {node_id}"
            )
    
    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Feature Value")
    ax.set_title(title)
    
    # Add legend if requested
    if include_legend and len(node_ids) <= 20:  # Only show legend if not too many nodes
        ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def animate_feature_timeseries(
    node_sequences: Union[List[torch.Tensor], torch.Tensor],
    target_sequences: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    pred_sequences: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    node_indices: Optional[List[int]] = None,
    feature_indices: Optional[List[int]] = None,
    time_steps: Optional[List[float]] = None,
    title: str = "Temporal Feature Animation",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    fps: int = 4
) -> plt.Figure:
    """
    Create an animation of temporal feature evolution, optionally with predictions.
    
    Args:
        node_sequences: List of node feature tensors or batched tensor [batch_size, seq_len, num_nodes, feature_dim]
        target_sequences: Optional ground truth sequences
        pred_sequences: Optional prediction sequences
        node_indices: Indices of nodes to visualize (default: first 5 nodes)
        feature_indices: Indices of features to visualize (default: first feature)
        time_steps: Optional time step values
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the animation (requires ffmpeg)
        fps: Frames per second for the animation
        
    Returns:
        Matplotlib figure object
    """
    # Process input sequences to standard format
    if isinstance(node_sequences, torch.Tensor):
        # Convert [batch_size, seq_len, nodes, features] to list of tensors
        node_sequences = [node_sequences[i] for i in range(node_sequences.size(0))]
    
    if target_sequences is not None and isinstance(target_sequences, torch.Tensor):
        target_sequences = [target_sequences[i] for i in range(target_sequences.size(0))]
    
    if pred_sequences is not None and isinstance(pred_sequences, torch.Tensor):
        pred_sequences = [pred_sequences[i] for i in range(pred_sequences.size(0))]
    
    # Convert tensors to numpy
    node_sequences_np = []
    for seq in node_sequences:
        if isinstance(seq, torch.Tensor):
            node_sequences_np.append(seq.detach().cpu().numpy())
        else:
            node_sequences_np.append(seq)
    
    target_sequences_np = None
    if target_sequences:
        target_sequences_np = []
        for seq in target_sequences:
            if isinstance(seq, torch.Tensor):
                target_sequences_np.append(seq.detach().cpu().numpy())
            else:
                target_sequences_np.append(seq)
    
    pred_sequences_np = None
    if pred_sequences:
        pred_sequences_np = []
        for seq in pred_sequences:
            if isinstance(seq, torch.Tensor):
                pred_sequences_np.append(seq.detach().cpu().numpy())
            else:
                pred_sequences_np.append(seq)
    
    # Get sequence dimensions
    num_sequences = len(node_sequences_np)
    seq_len = node_sequences_np[0].shape[0]
    num_nodes = node_sequences_np[0].shape[1]
    num_features = node_sequences_np[0].shape[2] if node_sequences_np[0].ndim > 2 else 1
    
    # Set default indices if not provided
    if node_indices is None:
        node_indices = list(range(min(5, num_nodes)))
    
    if feature_indices is None:
        feature_indices = [0] if num_features > 0 else []
    
    # Set up time steps or default values
    if time_steps is None:
        time_steps = list(range(seq_len))
    
    # Create figure and axes
    fig, axes = plt.subplots(
        len(node_indices), len(feature_indices), 
        figsize=figsize, sharex=True
    )
    
    # Ensure axes is always a 2D array
    if len(node_indices) == 1 and len(feature_indices) == 1:
        axes = np.array([[axes]])
    elif len(node_indices) == 1:
        axes = np.array([axes])
    elif len(feature_indices) == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Initialize line objects for animations
    lines = []
    
    # Add lines for each node and feature
    for n_idx, node_id in enumerate(node_indices):
        for f_idx, feat_id in enumerate(feature_indices):
            ax = axes[n_idx, f_idx]
            
            # Set up subplot title
            ax.set_title(f"Node {node_id}, Feature {feat_id}")
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Create input line
            input_line, = ax.plot([], [], 'b-', label='Input')
            lines.append(input_line)
            
            # Create target line if available
            if target_sequences_np is not None:
                target_line, = ax.plot([], [], 'g-', label='Target')
                lines.append(target_line)
            
            # Create prediction line if available
            if pred_sequences_np is not None:
                pred_line, = ax.plot([], [], 'r--', label='Prediction')
                lines.append(pred_line)
            
            # Add legend
            ax.legend(loc='upper right')
    
    # Set common labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Time Step')
    
    for ax in axes[:, 0]:
        ax.set_ylabel('Value')
    
    # Add title
    fig.suptitle(title)
    plt.tight_layout()
    
    # Find overall min and max values for consistent y-axis scaling
    all_values = []
    for seq in node_sequences_np:
        for node_id in node_indices:
            for feat_id in feature_indices:
                if node_id < seq.shape[1] and feat_id < seq.shape[2]:
                    all_values.extend(seq[:, node_id, feat_id].flatten())
    
    if target_sequences_np:
        for seq in target_sequences_np:
            for node_id in node_indices:
                for feat_id in feature_indices:
                    if node_id < seq.shape[1] and feat_id < seq.shape[2]:
                        all_values.extend(seq[:, node_id, feat_id].flatten())
    
    if pred_sequences_np:
        for seq in pred_sequences_np:
            for node_id in node_indices:
                for feat_id in feature_indices:
                    if node_id < seq.shape[1] and feat_id < seq.shape[2]:
                        all_values.extend(seq[:, node_id, feat_id].flatten())
    
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_margin = (y_max - y_min) * 0.1
    y_min -= y_margin
    y_max += y_margin
    
    # Initialize function
    def init():
        for n_idx, node_id in enumerate(node_indices):
            for f_idx, feat_id in enumerate(feature_indices):
                ax = axes[n_idx, f_idx]
                ax.set_xlim(min(time_steps), max(time_steps))
                ax.set_ylim(y_min, y_max)
        
        # Initialize all lines empty
        for line in lines:
            line.set_data([], [])
        
        return lines
    
    # Animation update function
    def update(frame):
        line_idx = 0
        
        for seq_idx in range(num_sequences):
            for n_idx, node_id in enumerate(node_indices):
                for f_idx, feat_id in enumerate(feature_indices):
                    # Skip if node or feature index is out of bounds
                    if node_id >= node_sequences_np[seq_idx].shape[1] or feat_id >= node_sequences_np[seq_idx].shape[2]:
                        continue
                    
                    # Get data for this node and feature
                    t = time_steps[:frame+1]
                    
                    # Input sequence
                    input_data = node_sequences_np[seq_idx][:frame+1, node_id, feat_id]
                    lines[line_idx].set_data(t, input_data)
                    line_idx += 1
                    
                    # Target sequence if available
                    if target_sequences_np is not None:
                        if seq_idx < len(target_sequences_np) and node_id < target_sequences_np[seq_idx].shape[1]:
                            target_data = target_sequences_np[seq_idx][:frame+1, node_id, feat_id]
                            lines[line_idx].set_data(t, target_data)
                        line_idx += 1
                    
                    # Prediction sequence if available
                    if pred_sequences_np is not None:
                        if seq_idx < len(pred_sequences_np) and node_id < pred_sequences_np[seq_idx].shape[1]:
                            pred_data = pred_sequences_np[seq_idx][:frame+1, node_id, feat_id]
                            lines[line_idx].set_data(t, pred_data)
                        line_idx += 1
        
        return lines
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=seq_len, 
        init_func=init, blit=True,
        interval=1000//fps
    )
    
    # Save animation if path provided
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=fps)
    
    # Show animation
    plt.show()
    
    return fig


def interactive_temporal_graph(
    node_coords: List[torch.Tensor],
    edge_indices: List[torch.Tensor],
    node_features: Optional[List[torch.Tensor]] = None,
    edge_features: Optional[List[torch.Tensor]] = None,
    node_labels: Optional[List[List[int]]] = None,
    timestamps: Optional[List[float]] = None,
    title: str = "Interactive Temporal Graph",
    width: int = 800,
    height: int = 600,
    node_size: int = 10,
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Create an interactive temporal graph visualization using Plotly.
    
    Args:
        node_coords: List of node coordinate tensors [num_timesteps, [num_nodes, 2]]
        edge_indices: List of edge index tensors [num_timesteps, [2, num_edges]]
        node_features: Optional list of node feature tensors [num_timesteps, [num_nodes, feature_dim]]
        edge_features: Optional list of edge feature tensors [num_timesteps, [num_edges, feature_dim]]
        node_labels: Optional list of node labels [num_timesteps, [num_nodes]]
        timestamps: Optional list of timestamps for each frame
        title: Title for the visualization
        width: Figure width in pixels
        height: Figure height in pixels
        node_size: Base size of nodes
        colorscale: Plotly colorscale for node colors
        
    Returns:
        Plotly figure object
    """
    if not node_coords or not edge_indices:
        raise ValueError("Node coordinates and edge indices must be provided")
    
    if len(node_coords) != len(edge_indices):
        raise ValueError("Number of timesteps must match between node_coords and edge_indices")
    
    # Number of timesteps
    num_timesteps = len(node_coords)
    
    # Set up timestamps or default time values
    if timestamps is None:
        timestamps = list(range(num_timesteps))
    
    # Create frames for animation
    frames = []
    
    # Process each timestep
    for t in range(num_timesteps):
        # Convert tensors to numpy
        coords = node_coords[t].detach().cpu().numpy() if isinstance(node_coords[t], torch.Tensor) else node_coords[t]
        edges = edge_indices[t].detach().cpu().numpy() if isinstance(edge_indices[t], torch.Tensor) else edge_indices[t]
        
        # Prepare node trace
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Node colors based on features if provided
        node_color = None
        if node_features is not None and t < len(node_features):
            features = node_features[t]
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            
            # Use first feature dimension for color if multi-dimensional
            if features.ndim > 1 and features.shape[1] > 0:
                node_color = features[:, 0]
            else:
                node_color = features
        
        # Node text labels
        text_labels = None
        if node_labels is not None and t < len(node_labels):
            text_labels = [f"Node {i}: {label}" for i, label in enumerate(node_labels[t])]
        else:
            text_labels = [f"Node {i}" for i in range(len(x_coords))]
        
        # Create node trace
        node_trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale=colorscale,
                showscale=True if node_color is not None else False
            ),
            text=text_labels,
            hoverinfo='text'
        )
        
        # Create edge traces
        edge_traces = []
        
        # Edge colors based on features if provided
        edge_color = None
        if edge_features is not None and t < len(edge_features):
            e_features = edge_features[t]
            if isinstance(e_features, torch.Tensor):
                e_features = e_features.detach().cpu().numpy()
            
            # Use first feature dimension for color if multi-dimensional
            if e_features.ndim > 1 and e_features.shape[1] > 0:
                edge_color = e_features[:, 0]
            else:
                edge_color = e_features
        
        # Create traces for edges
        for i in range(edges.shape[1]):
            src_idx, dst_idx = int(edges[0, i]), int(edges[1, i])
            
            # Skip if indices are out of bounds
            if src_idx >= len(x_coords) or dst_idx >= len(x_coords):
                continue
            
            # Extract coordinates
            x0, y0 = x_coords[src_idx], y_coords[src_idx]
            x1, y1 = x_coords[dst_idx], y_coords[dst_idx]
            
            # Edge color
            color = edge_color[i] if edge_color is not None else 'rgba(150,150,150,0.7)'
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],  # None creates a break between edges
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color=color),
                hoverinfo='none'
            )
            
            edge_traces.append(edge_trace)
        
        # Combine all traces for this frame
        frame_data = [node_trace] + edge_traces
        
        # Create frame
        frame = go.Frame(
            data=frame_data,
            name=f"frame_{t}"
        )
        
        frames.append(frame)
    
    # Create figure with initial frame data
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=title,
            width=width,
            height=height,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame_{i}"],
                            {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}
                        ],
                        "label": f"{timestamps[i]:.2f}" if timestamps else str(i),
                        "method": "animate"
                    }
                    for i in range(num_timesteps)
                ]
            }]
        )
    )
    
    return fig


def plot_temporal_patterns(
    feature_sequences: Union[List[torch.Tensor], torch.Tensor],
    target_sequences: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    prediction_sequences: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    timestamps: Optional[List[float]] = None,
    sample_indices: Optional[List[int]] = None,
    feature_indices: Optional[List[int]] = None,
    title: str = "Temporal Feature Patterns",
    figsize: Tuple[int, int] = (15, 10),
    colormap: str = "tab10",
    include_legend: bool = True,
    plot_type: str = "line",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal patterns from sequences with optional predictions.
    
    Args:
        feature_sequences: Input feature sequences [batch_size, seq_len, feature_dim]
                          or list of tensors
        target_sequences: Optional target sequences
        prediction_sequences: Optional prediction sequences
        timestamps: Optional time values for x-axis
        sample_indices: Optional indices of samples to plot (default: first 3)
        feature_indices: Optional indices of features to plot (default: first 2)
        title: Plot title
        figsize: Figure size (width, height)
        colormap: Colormap for lines
        include_legend: Whether to include a legend
        plot_type: Type of plot ('line', 'heatmap', 'scatter')
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy arrays
    if isinstance(feature_sequences, torch.Tensor):
        feature_sequences = feature_sequences.detach().cpu().numpy()
    elif isinstance(feature_sequences, list) and isinstance(feature_sequences[0], torch.Tensor):
        feature_sequences = [seq.detach().cpu().numpy() for seq in feature_sequences]
    
    if target_sequences is not None:
        if isinstance(target_sequences, torch.Tensor):
            target_sequences = target_sequences.detach().cpu().numpy()
        elif isinstance(target_sequences, list) and isinstance(target_sequences[0], torch.Tensor):
            target_sequences = [seq.detach().cpu().numpy() for seq in target_sequences]
    
    if prediction_sequences is not None:
        if isinstance(prediction_sequences, torch.Tensor):
            prediction_sequences = prediction_sequences.detach().cpu().numpy()
        elif isinstance(prediction_sequences, list) and isinstance(prediction_sequences[0], torch.Tensor):
            prediction_sequences = [seq.detach().cpu().numpy() for seq in prediction_sequences]
    
    # Handle list of sequences
    if isinstance(feature_sequences, list):
        # Convert to batch format: [batch_size, seq_len, feature_dim]
        feature_sequences = np.array(feature_sequences)
    
    if target_sequences is not None and isinstance(target_sequences, list):
        target_sequences = np.array(target_sequences)
    
    if prediction_sequences is not None and isinstance(prediction_sequences, list):
        prediction_sequences = np.array(prediction_sequences)
    
    # Get dimensions
    batch_size, seq_len, feature_dim = feature_sequences.shape
    
    # Set default indices if not provided
    if sample_indices is None:
        sample_indices = list(range(min(3, batch_size)))
    
    if feature_indices is None:
        feature_indices = list(range(min(2, feature_dim)))
    
    # Set up timestamps
    if timestamps is None:
        timestamps = np.arange(seq_len)
    
    # Create figure
    if plot_type == 'heatmap':
        # Create heatmap visualization
        nrows = len(sample_indices)
        fig, axes = plt.subplots(nrows, 3, figsize=figsize)
        
        # Ensure axes is a 2D array
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot heatmaps for each sample
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= batch_size:
                continue
            
            # Plot input sequence
            im1 = axes[i, 0].imshow(
                feature_sequences[sample_idx].T, 
                aspect='auto', 
                interpolation='nearest',
                cmap='viridis'
            )
            axes[i, 0].set_title(f"Input (Sample {sample_idx})")
            axes[i, 0].set_xlabel("Time Step")
            axes[i, 0].set_ylabel("Feature")
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot target sequence if available
            if target_sequences is not None:
                im2 = axes[i, 1].imshow(
                    target_sequences[sample_idx].T,
                    aspect='auto',
                    interpolation='nearest',
                    cmap='viridis'
                )
                axes[i, 1].set_title(f"Target (Sample {sample_idx})")
                axes[i, 1].set_xlabel("Time Step")
                plt.colorbar(im2, ax=axes[i, 1])
            else:
                axes[i, 1].set_visible(False)
            
            # Plot prediction sequence if available
            if prediction_sequences is not None:
                im3 = axes[i, 2].imshow(
                    prediction_sequences[sample_idx].T,
                    aspect='auto',
                    interpolation='nearest',
                    cmap='viridis'
                )
                axes[i, 2].set_title(f"Prediction (Sample {sample_idx})")
                axes[i, 2].set_xlabel("Time Step")
                plt.colorbar(im3, ax=axes[i, 2])
            else:
                axes[i, 2].set_visible(False)
    
    elif plot_type == 'scatter':
        # Create scatter plot visualization
        fig, axes = plt.subplots(len(sample_indices), len(feature_indices), figsize=figsize)
        
        # Ensure axes is a 2D array
        if len(sample_indices) == 1 and len(feature_indices) == 1:
            axes = np.array([[axes]])
        elif len(sample_indices) == 1:
            axes = np.array([axes])
        elif len(feature_indices) == 1:
            axes = np.array([[ax] for ax in axes])
        
        # Define colors for different series
        input_color = 'blue'
        target_color = 'green'
        pred_color = 'red'
        
        # Plot scatter plots for each sample and feature
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= batch_size:
                continue
                
            for j, feature_idx in enumerate(feature_indices):
                if feature_idx >= feature_dim:
                    continue
                    
                ax = axes[i, j]
                
                # Plot input sequence
                ax.scatter(
                    timestamps, 
                    feature_sequences[sample_idx, :, feature_idx],
                    color=input_color,
                    label='Input',
                    alpha=0.7,
                    marker='o'
                )
                
                # Plot target sequence if available
                if target_sequences is not None:
                    ax.scatter(
                        timestamps,
                        target_sequences[sample_idx, :, feature_idx],
                        color=target_color,
                        label='Target',
                        alpha=0.7,
                        marker='s'
                    )
                
                # Plot prediction sequence if available
                if prediction_sequences is not None:
                    ax.scatter(
                        timestamps,
                        prediction_sequences[sample_idx, :, feature_idx],
                        color=pred_color,
                        label='Prediction',
                        alpha=0.7,
                        marker='x'
                    )
                
                # Set labels and title
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
                ax.set_title(f"Sample {sample_idx}, Feature {feature_idx}")
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend for first subplot only
                if i == 0 and j == 0 and include_legend:
                    ax.legend()
    
    else:  # Default: line plot
        # Create line plot visualization
        fig, axes = plt.subplots(len(sample_indices), len(feature_indices), figsize=figsize)
        
        # Ensure axes is a 2D array
        if len(sample_indices) == 1 and len(feature_indices) == 1:
            axes = np.array([[axes]])
        elif len(sample_indices) == 1:
            axes = np.array([axes])
        elif len(feature_indices) == 1:
            axes = np.array([[ax] for ax in axes])
        
        # Define colors for different series
        input_color = 'blue'
        target_color = 'green'
        pred_color = 'red'
        
        # Plot line plots for each sample and feature
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= batch_size:
                continue
                
            for j, feature_idx in enumerate(feature_indices):
                if feature_idx >= feature_dim:
                    continue
                    
                ax = axes[i, j]
                
                # Plot input sequence
                ax.plot(
                    timestamps, 
                    feature_sequences[sample_idx, :, feature_idx],
                    color=input_color,
                    label='Input',
                    marker='o'
                )
                
                # Plot target sequence if available
                if target_sequences is not None:
                    ax.plot(
                        timestamps,
                        target_sequences[sample_idx, :, feature_idx],
                        color=target_color,
                        label='Target',
                        linestyle=':'
                    )
                
                # Plot prediction sequence if available
                if prediction_sequences is not None:
                    ax.plot(
                        timestamps,
                        prediction_sequences[sample_idx, :, feature_idx],
                        color=pred_color,
                        label='Prediction',
                        linestyle='--'
                    )
                
                # Set labels and title
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
                ax.set_title(f"Sample {sample_idx}, Feature {feature_idx}")
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend for first subplot only
                if i == 0 and j == 0 and include_legend:
                    ax.legend()
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
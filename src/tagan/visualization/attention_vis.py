"""
Attention pattern visualization for TAGAN models.

This module provides functions for visualizing attention patterns 
in geometric and temporal attention layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_attention_patterns(
    geometric_attention: List[torch.Tensor],
    temporal_attention: Optional[torch.Tensor] = None,
    node_ids: Optional[List[Any]] = None,
    snapshot_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (15, 10),
    colormap: str = 'plasma',
    title: str = 'Attention Patterns',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot geometric and/or temporal attention patterns.
    
    Args:
        geometric_attention: List of geometric attention weight tensors for each snapshot
        temporal_attention: Optional temporal attention weight tensor
        node_ids: Optional list of node IDs for labeling
        snapshot_indices: Optional list of snapshot indices to plot
        figsize: Figure size (width, height) in inches
        colormap: Matplotlib colormap for attention weights
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    if snapshot_indices is None and geometric_attention:
        # Default to all snapshots
        snapshot_indices = list(range(len(geometric_attention)))
    
    # Determine the number of plots needed
    n_geo_plots = len(snapshot_indices) if snapshot_indices else 0
    has_temp_plot = temporal_attention is not None
    n_plots = n_geo_plots + (1 if has_temp_plot else 0)
    
    if n_plots == 0:
        raise ValueError("No attention weights provided for visualization")
    
    # Create figure
    fig, axes = plt.subplots(
        1, n_plots, figsize=figsize, 
        gridspec_kw={'width_ratios': [1] * n_plots}
    )
    
    # Ensure axes is always a list
    if n_plots == 1:
        axes = [axes]
    
    # Plot geometric attention for selected snapshots
    for i, snapshot_idx in enumerate(snapshot_indices):
        if snapshot_idx >= len(geometric_attention):
            continue
            
        # Get attention weights for this snapshot
        attn = geometric_attention[snapshot_idx]
        
        # Handle different attention weight formats
        if isinstance(attn, torch.Tensor):
            # Convert to numpy for plotting
            attn_np = attn.detach().cpu().numpy()
            
            # Handle multi-head attention by averaging heads
            if len(attn_np.shape) > 2:
                # Shape: [batch_size, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
                if len(attn_np.shape) == 4:
                    # Average over batch dimension if present
                    attn_np = attn_np.mean(axis=0)
                # Average over head dimension
                attn_np = attn_np.mean(axis=0)
        else:
            # Handle dictionary or other formats
            if isinstance(attn, dict) and 'attention' in attn:
                attn_np = attn['attention']
                if isinstance(attn_np, torch.Tensor):
                    attn_np = attn_np.detach().cpu().numpy()
            else:
                continue  # Skip if format is not supported
        
        # Plot heatmap
        im = axes[i].imshow(attn_np, cmap=colormap, aspect='auto')
        axes[i].set_title(f'Geometric Attention (Snapshot {snapshot_idx})')
        
        # Add node IDs as labels if provided
        if node_ids is not None:
            # Get the number of nodes in this snapshot
            n_nodes = attn_np.shape[0]
            
            # Create labels based on available node IDs
            if len(node_ids) >= n_nodes:
                # Select node IDs for this snapshot
                snapshot_node_ids = node_ids[:n_nodes]
                
                # Set tick positions and labels
                tick_positions = np.arange(n_nodes)
                axes[i].set_xticks(tick_positions)
                axes[i].set_yticks(tick_positions)
                axes[i].set_xticklabels(snapshot_node_ids, rotation=45, ha='right')
                axes[i].set_yticklabels(snapshot_node_ids)
            else:
                # Use indices if not enough node IDs
                axes[i].set_xticks(np.arange(n_nodes))
                axes[i].set_yticks(np.arange(n_nodes))
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    # Plot temporal attention if provided
    if has_temp_plot:
        if isinstance(temporal_attention, torch.Tensor):
            # Convert to numpy for plotting
            temp_attn_np = temporal_attention.detach().cpu().numpy()
            
            # Handle multi-head attention by averaging heads
            if len(temp_attn_np.shape) > 2:
                # Shape: [batch_size, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
                if len(temp_attn_np.shape) == 4:
                    # Average over batch dimension if present
                    temp_attn_np = temp_attn_np.mean(axis=0)
                # Average over head dimension
                temp_attn_np = temp_attn_np.mean(axis=0)
            
            # Plot temporal attention heatmap
            temp_im = axes[-1].imshow(temp_attn_np, cmap=colormap, aspect='auto')
            axes[-1].set_title('Temporal Attention')
            
            # Add time step labels
            n_timesteps = temp_attn_np.shape[0]
            axes[-1].set_xticks(np.arange(n_timesteps))
            axes[-1].set_yticks(np.arange(n_timesteps))
            axes[-1].set_xticklabels([f't{i}' for i in range(n_timesteps)])
            axes[-1].set_yticklabels([f't{i}' for i in range(n_timesteps)])
            
            # Add colorbar
            plt.colorbar(temp_im, ax=axes[-1])
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_interactive_attention(
    geometric_attention: List[torch.Tensor],
    temporal_attention: Optional[torch.Tensor] = None,
    node_ids: Optional[List[Any]] = None,
    snapshot_indices: Optional[List[int]] = None,
    title: str = 'Interactive Attention Patterns',
    width: int = 1200,
    height: int = 800,
    colorscale: str = 'Plasma'
) -> go.Figure:
    """
    Create an interactive visualization of attention patterns using Plotly.
    
    Args:
        geometric_attention: List of geometric attention weight tensors for each snapshot
        temporal_attention: Optional temporal attention weight tensor
        node_ids: Optional list of node IDs for labeling
        snapshot_indices: Optional list of snapshot indices to plot
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        colorscale: Plotly colorscale for attention weights
        
    Returns:
        Plotly figure object
    """
    if snapshot_indices is None and geometric_attention:
        # Default to all snapshots
        snapshot_indices = list(range(len(geometric_attention)))
    
    # Determine layout based on number of plots
    n_geo_plots = len(snapshot_indices) if snapshot_indices else 0
    has_temp_plot = temporal_attention is not None
    n_plots = n_geo_plots + (1 if has_temp_plot else 0)
    
    if n_plots == 0:
        raise ValueError("No attention weights provided for visualization")
    
    # Create subplot layout
    row_cols = 1
    if n_plots > 2:
        row_cols = int(np.ceil(np.sqrt(n_plots)))
    
    subplot_titles = []
    for idx in snapshot_indices:
        subplot_titles.append(f'Geometric Attention (Snapshot {idx})')
    
    if has_temp_plot:
        subplot_titles.append('Temporal Attention')
    
    fig = make_subplots(
        rows=row_cols, 
        cols=row_cols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "heatmap"} for _ in range(row_cols)] for _ in range(row_cols)]
    )
    
    # Plot geometric attention for selected snapshots
    plot_index = 0
    for snapshot_idx in snapshot_indices:
        if snapshot_idx >= len(geometric_attention):
            continue
            
        # Get attention weights for this snapshot
        attn = geometric_attention[snapshot_idx]
        
        # Handle different attention weight formats
        if isinstance(attn, torch.Tensor):
            # Convert to numpy for plotting
            attn_np = attn.detach().cpu().numpy()
            
            # Handle multi-head attention by averaging heads
            if len(attn_np.shape) > 2:
                # Shape: [batch_size, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
                if len(attn_np.shape) == 4:
                    # Average over batch dimension if present
                    attn_np = attn_np.mean(axis=0)
                # Average over head dimension
                attn_np = attn_np.mean(axis=0)
        else:
            # Handle dictionary or other formats
            if isinstance(attn, dict) and 'attention' in attn:
                attn_np = attn['attention']
                if isinstance(attn_np, torch.Tensor):
                    attn_np = attn_np.detach().cpu().numpy()
            else:
                continue  # Skip if format is not supported
        
        # Set node labels
        n_nodes = attn_np.shape[0]
        x_labels = list(range(n_nodes))
        y_labels = list(range(n_nodes))
        
        if node_ids is not None and len(node_ids) >= n_nodes:
            # Use provided node IDs as labels
            node_labels = node_ids[:n_nodes]
            x_labels = node_labels
            y_labels = node_labels
        
        # Calculate row and column for subplot
        row = (plot_index // row_cols) + 1
        col = (plot_index % row_cols) + 1
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=attn_np,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                showscale=True,
                name=f'Snapshot {snapshot_idx}'
            ),
            row=row, col=col
        )
        
        plot_index += 1
    
    # Plot temporal attention if provided
    if has_temp_plot:
        if isinstance(temporal_attention, torch.Tensor):
            # Convert to numpy for plotting
            temp_attn_np = temporal_attention.detach().cpu().numpy()
            
            # Handle multi-head attention by averaging heads
            if len(temp_attn_np.shape) > 2:
                # Shape: [batch_size, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
                if len(temp_attn_np.shape) == 4:
                    # Average over batch dimension if present
                    temp_attn_np = temp_attn_np.mean(axis=0)
                # Average over head dimension
                temp_attn_np = temp_attn_np.mean(axis=0)
            
            # Set time step labels
            n_timesteps = temp_attn_np.shape[0]
            time_labels = [f't{i}' for i in range(n_timesteps)]
            
            # Calculate row and column for subplot
            row = (plot_index // row_cols) + 1
            col = (plot_index % row_cols) + 1
            
            # Add temporal attention heatmap
            fig.add_trace(
                go.Heatmap(
                    z=temp_attn_np,
                    x=time_labels,
                    y=time_labels,
                    colorscale=colorscale,
                    showscale=True,
                    name='Temporal Attention'
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        width=width,
        height=height,
        showlegend=False
    )
    
    return fig


def plot_temporal_graph_attention(
    geometric_attention: List[torch.Tensor],
    graph_data: List[Dict[str, Any]],
    node_positions: Optional[Dict[Any, Tuple[float, float]]] = None,
    snapshot_indices: Optional[List[int]] = None,
    node_size: int = 30,
    edge_width_scale: float = 3.0,
    colormap: str = 'plasma',
    figsize: Tuple[int, int] = (18, 12),
    title: str = 'Temporal Graph Attention',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize attention patterns over a temporal graph structure.
    
    Args:
        geometric_attention: List of geometric attention weight tensors for each snapshot
        graph_data: List of graph data dictionaries with 'edge_index' and optionally 'node_ids'
        node_positions: Optional dictionary mapping node IDs to (x,y) positions
        snapshot_indices: Optional list of snapshot indices to plot
        node_size: Size of nodes in the visualization
        edge_width_scale: Scale factor for edge widths based on attention
        colormap: Matplotlib colormap for attention weights
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    if snapshot_indices is None and geometric_attention:
        # Default to first 4 snapshots or all if fewer
        snapshot_indices = list(range(min(4, len(geometric_attention))))
    
    n_plots = len(snapshot_indices)
    if n_plots == 0:
        raise ValueError("No snapshots selected for visualization")
    
    # Determine grid layout
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes array for easy indexing
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create colormap function
    cmap = plt.cm.get_cmap(colormap)
    
    # Plot each selected snapshot
    for i, snapshot_idx in enumerate(snapshot_indices):
        if snapshot_idx >= len(geometric_attention) or snapshot_idx >= len(graph_data):
            continue
        
        # Get graph data for this snapshot
        graph_snapshot = graph_data[snapshot_idx]
        edge_index = graph_snapshot['edge_index']
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.detach().cpu().numpy()
        
        # Get node IDs if available
        node_ids = graph_snapshot.get('node_ids', None)
        
        # Get attention weights for this snapshot
        attn = geometric_attention[snapshot_idx]
        
        # Convert to numpy and average over heads if needed
        if isinstance(attn, torch.Tensor):
            attn_np = attn.detach().cpu().numpy()
            if len(attn_np.shape) > 2:
                # Average over batch and/or head dimensions
                if len(attn_np.shape) == 4:  # [batch, heads, nodes, nodes]
                    attn_np = attn_np.mean(axis=(0, 1))
                else:  # [heads, nodes, nodes]
                    attn_np = attn_np.mean(axis=0)
        else:
            # Handle dictionary format
            if isinstance(attn, dict) and 'attention' in attn:
                attn_np = attn['attention']
                if isinstance(attn_np, torch.Tensor):
                    attn_np = attn_np.detach().cpu().numpy()
            else:
                continue
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        n_nodes = attn_np.shape[0]
        for n in range(n_nodes):
            node_id = node_ids[n] if node_ids and n < len(node_ids) else n
            G.add_node(node_id)
        
        # Add edges with attention weights
        for src, tgt in zip(edge_index[0], edge_index[1]):
            if src < n_nodes and tgt < n_nodes:
                weight = attn_np[src, tgt]
                G.add_edge(
                    node_ids[src] if node_ids and src < len(node_ids) else src,
                    node_ids[tgt] if node_ids and tgt < len(node_ids) else tgt,
                    weight=weight
                )
        
        # Get node positions
        if node_positions:
            pos = {node: node_positions[node] for node in G.nodes if node in node_positions}
            # For any nodes without positions, assign random positions
            for node in G.nodes:
                if node not in pos:
                    pos[node] = np.random.rand(2)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        ax = axes[i]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_size, 
            node_color='lightblue', edgecolors='black'
        )
        
        # Draw labels if not too many nodes
        if n_nodes <= 25:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        # Draw edges with colors based on attention weights
        for (u, v, data) in G.edges(data=True):
            weight = data['weight']
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v)],
                width=weight * edge_width_scale,
                edge_color=[cmap(weight)],
                alpha=0.7,
                arrows=True,
                arrowsize=10
            )
        
        # Set title for this subplot
        ax.set_title(f'Snapshot {snapshot_idx}')
        ax.axis('off')
    
    # Remove empty subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label('Attention Weight')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def create_animated_attention(
    geometric_attention: List[torch.Tensor],
    graph_data: List[Dict[str, Any]],
    node_ids: Optional[List[Any]] = None,
    width: int = 800,
    height: int = 600,
    colorscale: str = 'Plasma',
    title: str = 'Animated Attention Patterns'
) -> go.Figure:
    """
    Create an animated visualization of attention patterns over time using Plotly.
    
    Args:
        geometric_attention: List of geometric attention weight tensors for each snapshot
        graph_data: List of graph data dictionaries with 'edge_index'
        node_ids: Optional list of node IDs for labeling
        width: Figure width in pixels
        height: Figure height in pixels
        colorscale: Plotly colorscale for attention weights
        title: Plot title
        
    Returns:
        Plotly figure object with animation frames
    """
    # Create frame data for each time step
    frames = []
    
    # Process each snapshot
    for t, (attn, graph) in enumerate(zip(geometric_attention, graph_data)):
        # Convert attention to numpy and average over heads if needed
        if isinstance(attn, torch.Tensor):
            attn_np = attn.detach().cpu().numpy()
            if len(attn_np.shape) > 2:
                # Average over batch and/or head dimensions
                if len(attn_np.shape) == 4:  # [batch, heads, nodes, nodes]
                    attn_np = attn_np.mean(axis=(0, 1))
                else:  # [heads, nodes, nodes]
                    attn_np = attn_np.mean(axis=0)
        else:
            # Handle dictionary format
            if isinstance(attn, dict) and 'attention' in attn:
                attn_np = attn['attention']
                if isinstance(attn_np, torch.Tensor):
                    attn_np = attn_np.detach().cpu().numpy()
            else:
                continue
        
        # Create frame for this snapshot
        frame = {
            "name": f"t{t}",
            "data": [go.Heatmap(
                z=attn_np,
                colorscale=colorscale,
                showscale=True if t == 0 else False  # Only show colorbar for first frame
            )]
        }
        frames.append(frame)
    
    # Get data for first frame
    initial_attn = None
    if geometric_attention:
        initial_attn = geometric_attention[0]
        if isinstance(initial_attn, torch.Tensor):
            initial_attn = initial_attn.detach().cpu().numpy()
            if len(initial_attn.shape) > 2:
                if len(initial_attn.shape) == 4:
                    initial_attn = initial_attn.mean(axis=(0, 1))
                else:
                    initial_attn = initial_attn.mean(axis=0)
    
    if initial_attn is None:
        raise ValueError("No valid attention data found for visualization")
    
    # Create figure with initial state
    fig = go.Figure(
        data=[go.Heatmap(
            z=initial_attn,
            colorscale=colorscale,
            showscale=True
        )],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                }
            ],
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }]
    )
    
    # Add slider for manual navigation
    sliders = [{
        "active": 0,
        "steps": [
            {
                "label": f"t{t}",
                "method": "animate",
                "args": [
                    [f"t{t}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
                ]
            }
            for t in range(len(frames))
        ],
        "x": 0.1,
        "y": 0,
        "xanchor": "left",
        "yanchor": "top"
    }]
    
    fig.update_layout(sliders=sliders)
    
    return fig


def plot_graph_with_attention(
    graph_data: Dict[str, Any],
    attention_weights: torch.Tensor,
    node_ids: Optional[List[Any]] = None,
    node_positions: Optional[Dict[Any, Tuple[float, float]]] = None,
    title: str = "Graph with Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 300,
    edge_width_scale: float = 5.0,
    colormap: str = "plasma",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize a graph with attention weights determining edge thickness.
    
    Args:
        graph_data: Dictionary containing 'edge_index' and optionally 'node_features'
        attention_weights: Tensor of attention weights between nodes
        node_ids: Optional list of node IDs for labeling
        node_positions: Optional dictionary mapping node IDs to (x,y) positions
        title: Plot title
        figsize: Figure size (width, height) in inches
        node_size: Size of nodes in the visualization
        edge_width_scale: Scale factor for edge widths based on attention
        colormap: Matplotlib colormap for attention weights
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract edge indices
    edge_index = graph_data['edge_index']
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    
    # Convert attention weights to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attn_np = attention_weights.detach().cpu().numpy()
        
        # Handle multi-head attention by averaging heads if necessary
        if len(attn_np.shape) > 2:
            # Average over batch dimension if present
            if len(attn_np.shape) == 4:  # [batch, heads, nodes, nodes]
                attn_np = attn_np.mean(axis=(0, 1))
            elif len(attn_np.shape) == 3:  # [heads, nodes, nodes]
                attn_np = attn_np.mean(axis=0)
    else:
        attn_np = attention_weights
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Determine number of nodes from attention matrix
    n_nodes = attn_np.shape[0]
    
    # Add nodes to the graph
    for n in range(n_nodes):
        node_id = node_ids[n] if node_ids and n < len(node_ids) else n
        G.add_node(node_id)
    
    # Add edges with attention weights
    for src, tgt in zip(edge_index[0], edge_index[1]):
        if src < n_nodes and tgt < n_nodes:
            src_id = node_ids[src] if node_ids and src < len(node_ids) else src
            tgt_id = node_ids[tgt] if node_ids and tgt < len(node_ids) else tgt
            
            # Use attention weight for this edge
            weight = attn_np[src, tgt]
            G.add_edge(src_id, tgt_id, weight=weight)
    
    # Get node positions - either from parameter, or compute layout
    if node_positions:
        pos = {node: node_positions[node] for node in G.nodes() if node in node_positions}
        # For any nodes without positions, compute with layout algorithm
        missing_nodes = [node for node in G.nodes() if node not in pos]
        if missing_nodes:
            missing_pos = nx.spring_layout(G.subgraph(missing_nodes), seed=42)
            pos.update(missing_pos)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Create colormap function
    cmap = plt.cm.get_cmap(colormap)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, 
        node_size=node_size,
        node_color="lightblue",
        edgecolors="black"
    )
    
    # Draw edges with varying widths and colors based on attention
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=weight * edge_width_scale,
            edge_color=[cmap(weight)],
            alpha=0.7,
            arrows=True,
            arrowsize=15
        )
    
    # Draw node labels if there are not too many nodes
    if n_nodes <= 30:
        nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set title
    ax.set_title(title)
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig
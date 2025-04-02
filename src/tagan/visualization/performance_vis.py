"""
Performance visualization tools for TAGAN models.

This module provides functions for visualizing model performance metrics,
including confusion matrices, ROC curves, and training history.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    auc, average_precision_score
)
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from sklearn.inspection import permutation_importance


def plot_performance_metrics(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Performance Metrics",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    log_scale: bool = False,
    include_legend: bool = True,
    smoothing: float = 0.0
) -> plt.Figure:
    """
    Plot multiple performance metrics over time.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        log_scale: Whether to use log scale for y-axis
        include_legend: Whether to include a legend
        smoothing: Exponential moving average smoothing factor (0 = no smoothing)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color cycle for lines
    colors = plt.cm.tab10.colors
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        if not values:
            continue
            
        epochs = list(range(1, len(values) + 1))
        
        # Apply smoothing if requested
        if smoothing > 0 and len(values) > 1:
            smoothed_values = [values[0]]
            for i in range(1, len(values)):
                smoothed_values.append(smoothing * values[i] + (1 - smoothing) * smoothed_values[-1])
            values = smoothed_values
        
        ax.plot(epochs, values, 'o-', color=colors[i % len(colors)], label=metric_name)
    
    # Configure plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    
    # Use log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if requested
    if include_legend:
        ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    normalize: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    include_colorbar: bool = True,
    include_values: bool = True
) -> plt.Figure:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        figsize: Figure size (width, height) in inches
        title: Plot title
        cmap: Colormap name
        normalize: Whether to normalize confusion matrix values
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        include_colorbar: Whether to include a colorbar
        include_values: Whether to display values in cells
        
    Returns:
        Matplotlib figure object
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Determine number of classes
    n_classes = cm.shape[0]
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    
    # Add colorbar if requested
    if include_colorbar:
        plt.colorbar(im, ax=ax)
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add class names
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add values to cells
    if include_values:
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_roc_curve(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_scores: Union[List[float], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    include_micro_avg: bool = True,
    include_macro_avg: bool = True
) -> plt.Figure:
    """
    Plot ROC curve for binary or multi-class classification.
    
    Args:
        y_true: Ground truth labels (one-hot encoded for multi-class)
        y_scores: Predicted probabilities
        class_names: Optional list of class names
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        include_micro_avg: Whether to include micro-average ROC curve
        include_macro_avg: Whether to include macro-average ROC curve
        
    Returns:
        Matplotlib figure object
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if it's binary or multi-class
    if len(y_scores.shape) == 1 or y_scores.shape[1] == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Set class names
        if class_names is None:
            class_names = ['Negative', 'Positive']
    else:
        # Multi-class classification
        n_classes = y_scores.shape[1]
        
        # Set default class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        # Convert to one-hot encoded if not already
        if len(y_true.shape) == 1:
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i in range(len(y_true)):
                y_true_onehot[i, y_true[i]] = 1
            y_true = y_true_onehot
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curve for this class
            ax.plot(fpr[i], tpr[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Compute micro-average ROC curve and ROC area
        if include_micro_avg:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            ax.plot(fpr["micro"], tpr["micro"], lw=2, linestyle='--',
                    label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        # Compute macro-average ROC curve and ROC area
        if include_macro_avg:
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            
            roc_auc["macro"] = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, lw=2, linestyle=':',
                    label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # Add reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Configure plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_precision_recall_curve(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_scores: Union[List[float], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    include_micro_avg: bool = True,
    include_macro_avg: bool = True
) -> plt.Figure:
    """
    Plot precision-recall curve for binary or multi-class classification.
    
    Args:
        y_true: Ground truth labels (one-hot encoded for multi-class)
        y_scores: Predicted probabilities
        class_names: Optional list of class names
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        include_micro_avg: Whether to include micro-average P-R curve
        include_macro_avg: Whether to include macro-average P-R curve
        
    Returns:
        Matplotlib figure object
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if it's binary or multi-class
    if len(y_scores.shape) == 1 or y_scores.shape[1] == 1:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Plot precision-recall curve
        ax.plot(recall, precision, lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        
        # Set class names
        if class_names is None:
            class_names = ['Negative', 'Positive']
    else:
        # Multi-class classification
        n_classes = y_scores.shape[1]
        
        # Set default class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Compute precision-recall curve and average precision for each class
        precision = {}
        recall = {}
        avg_precision = {}
        
        # Convert to one-hot encoded if not already
        if len(y_true.shape) == 1:
            y_true_onehot = np.zeros((len(y_true), n_classes))
            for i in range(len(y_true)):
                y_true_onehot[i, y_true[i]] = 1
            y_true = y_true_onehot
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
            avg_precision[i] = average_precision_score(y_true[:, i], y_scores[:, i])
            
            # Plot precision-recall curve for this class
            ax.plot(recall[i], precision[i], lw=2,
                    label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')
        
        # Compute micro-average precision-recall curve
        if include_micro_avg:
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_true.ravel(), y_scores.ravel())
            avg_precision["micro"] = average_precision_score(
                y_true.ravel(), y_scores.ravel(), average="micro")
            ax.plot(recall["micro"], precision["micro"], lw=2, linestyle='--',
                    label=f'Micro-average (AP = {avg_precision["micro"]:.3f})')
        
        # Compute macro-average precision-recall curve
        if include_macro_avg and n_classes > 1:
            # Use interpolation to compute a macro-average curve
            all_recall = np.linspace(0, 1, 100)
            mean_precision = np.zeros_like(all_recall)
            
            for i in range(n_classes):
                mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
            
            mean_precision /= n_classes
            
            # Compute average precision score
            avg_precision["macro"] = np.mean([avg_precision[i] for i in range(n_classes)])
            
            ax.plot(all_recall, mean_precision, lw=2, linestyle=':',
                    label=f'Macro-average (AP = {avg_precision["macro"]:.3f})')
    
    # Configure plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    val_history: Optional[Dict[str, List[float]]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Training History",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    metrics_to_plot: Optional[List[str]] = None,
    subplot_layout: Optional[Tuple[int, int]] = None,
    smoothing: float = 0.0
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary mapping metric names to lists of training values
        val_history: Optional dictionary mapping metric names to lists of validation values
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        metrics_to_plot: Optional list of metrics to plot (default: all)
        subplot_layout: Optional layout for subplots (rows, cols)
        smoothing: Exponential moving average smoothing factor (0 = no smoothing)
        
    Returns:
        Matplotlib figure object
    """
    # Determine which metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = list(history.keys())
    
    # Filter metrics
    metrics = [m for m in metrics_to_plot if m in history]
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        raise ValueError("No valid metrics to plot")
    
    # Determine subplot layout
    if subplot_layout is None:
        if n_metrics <= 2:
            rows, cols = 1, n_metrics
        else:
            cols = 2
            rows = (n_metrics + cols - 1) // cols
    else:
        rows, cols = subplot_layout
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = np.array([axes])
    
    # Ensure axes is a 1D array
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Get training values
            values = history[metric]
            epochs = list(range(1, len(values) + 1))
            
            # Apply smoothing if requested
            if smoothing > 0 and len(values) > 1:
                smoothed_values = [values[0]]
                for i in range(1, len(values)):
                    smoothed_values.append(smoothing * values[i] + (1 - smoothing) * smoothed_values[-1])
                values = smoothed_values
            
            # Plot training values
            ax.plot(epochs, values, 'b-', label='Training')
            
            # Plot validation values if available
            if val_history is not None and metric in val_history:
                val_values = val_history[metric]
                
                # Ensure same length
                val_epochs = list(range(1, len(val_values) + 1))
                
                # Apply smoothing if requested
                if smoothing > 0 and len(val_values) > 1:
                    smoothed_val_values = [val_values[0]]
                    for i in range(1, len(val_values)):
                        smoothed_val_values.append(
                            smoothing * val_values[i] + (1 - smoothing) * smoothed_val_values[-1]
                        )
                    val_values = smoothed_val_values
                
                ax.plot(val_epochs, val_values, 'r-', label='Validation')
            
            # Configure subplot
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
    
    # Remove unused subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_metric_comparison(
    metrics: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Performance Comparison",
    plot_type: str = "bar",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    colormap: str = "tab10"
) -> plt.Figure:
    """
    Compare multiple metrics across different models or configurations.
    
    Args:
        metrics: Dictionary mapping model names to dictionaries of metrics
        figsize: Figure size (width, height) in inches
        title: Plot title
        plot_type: Type of plot ('bar', 'radar', or 'heatmap')
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        sort_by: Optional metric name to sort by
        ascending: Whether to sort in ascending order
        colormap: Colormap for the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert metrics to DataFrame
    df = pd.DataFrame(metrics).T
    
    # Sort if requested
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Create figure based on plot type
    if plot_type == 'bar':
        # Bar chart
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, colormap=colormap)
        
        # Configure plot
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
    elif plot_type == 'radar':
        # Radar chart (spider plot)
        fig = plt.figure(figsize=figsize)
        
        # Get number of metrics and models
        n_metrics = len(df.columns)
        n_models = len(df.index)
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create axis
        ax = fig.add_subplot(111, polar=True)
        
        # Get colormap
        cmap = plt.cm.get_cmap(colormap, n_models)
        
        # Plot each model
        for i, (model, row) in enumerate(df.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, color=cmap(i), label=model)
            ax.fill(angles, values, alpha=0.1, color=cmap(i))
        
        # Set metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df.columns)
        
        # Configure plot
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
    elif plot_type == 'heatmap':
        # Heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap=colormap, ax=ax)
        
        # Configure plot
        ax.set_title(title)
        plt.tight_layout()
        
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def interactive_performance_plot(
    metrics: Dict[str, List[float]],
    val_metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Interactive Performance Plot",
    width: int = 900,
    height: int = 600,
    smoothing: float = 0.0
) -> go.Figure:
    """
    Create an interactive performance plot using Plotly.
    
    Args:
        metrics: Dictionary mapping metric names to lists of training values
        val_metrics: Optional dictionary mapping metric names to lists of validation values
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        smoothing: Exponential moving average smoothing factor (0 = no smoothing)
        
    Returns:
        Plotly figure object
    """
    # Create subplots - one for each metric
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        raise ValueError("No metrics provided for plotting")
    
    # Determine subplot layout
    if n_metrics <= 2:
        rows, cols = 1, n_metrics
    else:
        cols = 2
        rows = (n_metrics + cols - 1) // cols
    
    # Create subplot titles
    subplot_titles = [key.replace('_', ' ').title() for key in metrics.keys()]
    
    # Create figure
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    
    # Add traces for each metric
    i = 0
    for metric_name, values in metrics.items():
        row = i // cols + 1
        col = i % cols + 1
        
        epochs = list(range(1, len(values) + 1))
        
        # Apply smoothing if requested
        if smoothing > 0 and len(values) > 1:
            smoothed_values = [values[0]]
            for j in range(1, len(values)):
                smoothed_values.append(smoothing * values[j] + (1 - smoothing) * smoothed_values[-1])
            plot_values = smoothed_values
        else:
            plot_values = values
        
        # Add training trace
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=plot_values,
                mode='lines+markers',
                name=f'Training {metric_name}',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
        # Add validation trace if available
        if val_metrics is not None and metric_name in val_metrics:
            val_values = val_metrics[metric_name]
            val_epochs = list(range(1, len(val_values) + 1))
            
            # Apply smoothing if requested
            if smoothing > 0 and len(val_values) > 1:
                smoothed_val_values = [val_values[0]]
                for j in range(1, len(val_values)):
                    smoothed_val_values.append(
                        smoothing * val_values[j] + (1 - smoothing) * smoothed_val_values[-1]
                    )
                plot_val_values = smoothed_val_values
            else:
                plot_val_values = val_values
            
            fig.add_trace(
                go.Scatter(
                    x=val_epochs,
                    y=plot_val_values,
                    mode='lines+markers',
                    name=f'Validation {metric_name}',
                    line=dict(color='red')
                ),
                row=row, col=col
            )
        
        # Update axis labels
        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
        
        i += 1
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=True
    )
    
    return fig


def interactive_confusion_matrix(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    title: str = "Interactive Confusion Matrix",
    width: int = 700,
    height: int = 600,
    colorscale: str = "Blues",
    normalize: bool = False
) -> go.Figure:
    """
    Create an interactive confusion matrix visualization using Plotly.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels
        colorscale: Plotly colorscale name
        normalize: Whether to normalize confusion matrix values
        
    Returns:
        Plotly figure object
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_template = ".2%"
    else:
        text_template = "d"
    
    # Determine number of classes
    n_classes = cm.shape[0]
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=colorscale,
        showscale=True
    ))
    
    # Add text annotations
    annotations = []
    for i in range(n_classes):
        for j in range(n_classes):
            if normalize:
                text = f"{cm[i, j]:.2f}"
            else:
                text = f"{cm[i, j]}"
                
            annotations.append(dict(
                x=class_names[j],
                y=class_names[i],
                text=text,
                showarrow=False,
                font=dict(
                    color="white" if cm[i, j] > cm.max() / 2 else "black"
                )
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label", autorange="reversed"),
        annotations=annotations
    )
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: Union[List[float], np.ndarray, torch.Tensor],
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    plot_type: str = "bar",
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    colormap: str = "viridis",
    error_bars: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: Names of features
        importances: Importance scores for each feature
        title: Plot title
        figsize: Figure size (width, height) in inches
        plot_type: Type of plot ('bar' or 'horizontal_bar')
        top_n: Optional number of top features to show
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        colormap: Colormap for the plot
        error_bars: Optional standard deviations for error bars
        
    Returns:
        Matplotlib figure object
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(importances, torch.Tensor):
        importances = importances.detach().cpu().numpy()
    
    if isinstance(error_bars, torch.Tensor):
        error_bars = error_bars.detach().cpu().numpy()
    
    # Sort features by importance
    indices = np.argsort(importances)
    
    if top_n is not None and top_n < len(indices):
        # Get top N most important features
        indices = indices[-top_n:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get sorted data
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    sorted_errors = None if error_bars is None else error_bars[indices]
    
    # Create colormap
    cmap = plt.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(sorted_importances)))
    
    # Plot based on type
    if plot_type == 'bar':
        bars = ax.bar(
            range(len(sorted_importances)),
            sorted_importances,
            yerr=sorted_errors,
            color=colors
        )
        ax.set_xticks(range(len(sorted_importances)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        
    elif plot_type == 'horizontal_bar':
        bars = ax.barh(
            range(len(sorted_importances)),
            sorted_importances,
            xerr=sorted_errors,
            color=colors
        )
        ax.set_yticks(range(len(sorted_importances)))
        ax.set_yticklabels(sorted_names)
        ax.set_ylabel('Feature')
        ax.set_xlabel('Importance')
        ax.invert_yaxis()  # Highest values at the top
        
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    # Configure plot
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7, axis='both')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig

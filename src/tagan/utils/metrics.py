"""
Metrics utilities for TAGAN.

This module provides functions for calculating various metrics
for model evaluation, as well as a metrics tracker for monitoring
performance over time.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt
import os


def calculate_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    threshold: float = 0.5  # Standard threshold for binary classification
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        labels: Ground truth labels
        predictions: Model predictions (logits or probabilities)
        threshold: Threshold for converting probabilities to class labels (default: 0.5)
                  Using standard threshold based on testing results
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert tensors to numpy arrays
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Handle different prediction formats
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Multi-class: Convert probabilities to class labels
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions  # Keep probabilities for ROC AUC
    else:
        # Binary: Apply threshold to convert to class labels
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        
        # Handle empty predictions
        if predictions.size == 0:
            print("Warning: Empty predictions array")
            # Return default metrics
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
        
        pred_classes = (predictions > threshold).astype(int)
        pred_probs = predictions  # Keep probabilities for ROC AUC
    
    # Ensure labels are in the right format
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # Convert one-hot encoded labels to class indices
        true_classes = np.argmax(labels, axis=1)
    else:
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        true_classes = labels.astype(int)
    
    # Make sure we have data to calculate metrics
    # Ensure we're working with arrays, not scalar values
    if np.isscalar(true_classes):
        true_classes = np.array([true_classes])
    if np.isscalar(pred_classes):
        pred_classes = np.array([pred_classes])
        
    if len(true_classes) == 0 or len(pred_classes) == 0:
        print(f"Warning: Empty arrays for metrics calculation. true_classes: {true_classes}, pred_classes: {pred_classes}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0
        }
    
    # Ensure pred_classes is not a scalar
    if np.isscalar(pred_classes):
        pred_classes = np.array([pred_classes])
    
    # Ensure true_classes is not a scalar
    if np.isscalar(true_classes):
        true_classes = np.array([true_classes])
    
    # Calculate metrics
    metrics = {}
    
    # Accuracy
    try:
        metrics['accuracy'] = accuracy_score(true_classes, pred_classes)
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        print(f"true_classes shape: {true_classes.shape if hasattr(true_classes, 'shape') else 'scalar'}")
        print(f"pred_classes shape: {pred_classes.shape if hasattr(pred_classes, 'shape') else 'scalar'}")
        metrics['accuracy'] = 0.0
    
    # For binary classification, calculate additional metrics
    num_classes = max(np.max(true_classes) + 1, 2)
    if num_classes <= 2:
        # Precision
        metrics['precision'] = precision_score(
            true_classes, pred_classes, average='binary', zero_division=0
        )
        
        # Recall
        metrics['recall'] = recall_score(
            true_classes, pred_classes, average='binary', zero_division=0
        )
        
        # F1 score
        metrics['f1'] = f1_score(
            true_classes, pred_classes, average='binary', zero_division=0
        )
        
        # ROC AUC
        try:
            # Handle scalar pred_probs
            if np.isscalar(pred_probs):
                print(f"Converting scalar pred_probs ({pred_probs}) to array")
                pred_probs = np.array([pred_probs])
                
            # For a single sample, we can't calculate ROC AUC
            if len(true_classes) <= 1 or len(np.unique(true_classes)) <= 1:
                print("Skipping ROC AUC calculation - insufficient samples or classes")
                metrics['roc_auc'] = 0.5
            else:
                metrics['roc_auc'] = roc_auc_score(true_classes, pred_probs)
        except Exception as e:
            # ROC AUC can't be calculated if there's only one class or other error
            print(f"ROC AUC calculation failed: {e}")
            metrics['roc_auc'] = 0.5
        
        # Average precision
        try:
            # For a single sample, we can't calculate average precision
            if len(true_classes) <= 1 or len(np.unique(true_classes)) <= 1:
                print("Skipping average precision calculation - insufficient samples or classes")
                metrics['avg_precision'] = 0.5
            else:
                metrics['avg_precision'] = average_precision_score(true_classes, pred_probs)
        except Exception as e:
            print(f"Average precision calculation failed: {e}")
            metrics['avg_precision'] = 0.5
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            true_classes, pred_classes, labels=[0, 1]
        ).ravel()
        
        metrics['true_positives'] = float(tp)
        metrics['false_positives'] = float(fp)
        metrics['true_negatives'] = float(tn)
        metrics['false_negatives'] = float(fn)
        
        # Specificity (true negative rate)
        metrics['specificity'] = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class metrics
        # Precision
        metrics['precision'] = precision_score(
            true_classes, pred_classes, average='macro', zero_division=0
        )
        
        # Recall
        metrics['recall'] = recall_score(
            true_classes, pred_classes, average='macro', zero_division=0
        )
        
        # F1 score
        metrics['f1'] = f1_score(
            true_classes, pred_classes, average='macro', zero_division=0
        )
        
        # Weighted F1 score
        metrics['weighted_f1'] = f1_score(
            true_classes, pred_classes, average='weighted', zero_division=0
        )
        
        # Multi-class ROC AUC
        try:
            # One-vs-rest ROC AUC for each class
            roc_auc = roc_auc_score(
                np.eye(num_classes)[true_classes], 
                pred_probs, 
                multi_class='ovr',
                average='macro'
            )
            metrics['roc_auc'] = roc_auc
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def calculate_regression_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        labels: Ground truth values
        predictions: Model predictions
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert tensors to numpy arrays
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Ensure arrays have the right shape
    if len(labels.shape) > 1:
        labels = labels.squeeze()
    if len(predictions.shape) > 1:
        predictions = predictions.squeeze()
    
    # Calculate metrics
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = float(np.mean((predictions - labels) ** 2))
    
    # Root Mean Squared Error
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    # Mean Absolute Error
    metrics['mae'] = float(np.mean(np.abs(predictions - labels)))
    
    # Mean Absolute Percentage Error
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((labels - predictions) / np.maximum(np.abs(labels), 1e-8))) * 100
        metrics['mape'] = float(mape) if not np.isinf(mape) and not np.isnan(mape) else 0.0
    
    # R-squared
    if np.var(labels) > 0:
        metrics['r2'] = float(1 - np.sum((labels - predictions) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
    else:
        metrics['r2'] = 0.0
    
    return metrics


def calculate_graph_metrics(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate graph-level metrics.
    
    Args:
        edge_index: Graph connectivity in COO format [2, num_edges]
        edge_weights: Optional edge weights [num_edges]
        
    Returns:
        Dictionary of metric names and values
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    if edge_weights is not None and isinstance(edge_weights, torch.Tensor):
        edge_weights = edge_weights.detach().cpu().numpy()
    
    # Calculate metrics
    metrics = {}
    
    # Number of nodes and edges
    num_nodes = max(np.max(edge_index[0]), np.max(edge_index[1])) + 1
    num_edges = edge_index.shape[1]
    
    metrics['num_nodes'] = float(num_nodes)
    metrics['num_edges'] = float(num_edges)
    
    # Graph density
    max_edges = num_nodes * (num_nodes - 1) / 2  # Undirected graph
    metrics['density'] = float(num_edges / max_edges) if max_edges > 0 else 0.0
    
    # Average degree
    metrics['avg_degree'] = float(num_edges / num_nodes) if num_nodes > 0 else 0.0
    
    # Average weighted degree (if weights provided)
    if edge_weights is not None:
        metrics['avg_weighted_degree'] = float(np.sum(edge_weights) / num_nodes) if num_nodes > 0 else 0.0
    
    return metrics


class MetricsTracker:
    """
    Tracker for monitoring metrics over time.
    
    This class keeps track of various metrics during training and evaluation,
    provides methods for updating metrics, and generates visualizations.
    
    Attributes:
        metrics (dict): Dictionary of metrics
        best_metrics (dict): Dictionary of best metrics
        epochs (list): List of epochs
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        # Dictionary to store metrics
        self.metrics = {
            'train': {},  # Training metrics
            'val': {},    # Validation metrics
            'test': {}    # Test metrics
        }
        
        # Store best metrics
        self.best_metrics = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        # Store epochs for each metric
        self.epochs = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def update_metrics(
        self,
        metrics_dict: Dict[str, float],
        split: str = 'train',
        epoch: Optional[int] = None
    ):
        """
        Update metrics for a specific split.
        
        Args:
            metrics_dict: Dictionary of metrics to update
            split: Data split ('train', 'val', or 'test')
            epoch: Current epoch (default: None)
        """
        if split not in self.metrics:
            raise ValueError(f"Invalid split: {split}")
        
        # Update epochs
        if epoch is not None:
            self.epochs[split].append(epoch)
        
        # Update metrics
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.metrics[split]:
                self.metrics[split][metric_name] = []
            
            self.metrics[split][metric_name].append(value)
            
            # Update best metrics
            if metric_name not in self.best_metrics[split] or value > self.best_metrics[split][metric_name]:
                self.best_metrics[split][metric_name] = value
    
    def update_epoch_metrics(
        self, 
        metrics_dict: Dict[str, float],
        epoch: int
    ):
        """
        Update validation metrics for a specific epoch.
        
        Args:
            metrics_dict: Dictionary of metrics to update
            epoch: Current epoch
        """
        self.update_metrics(metrics_dict, split='val', epoch=epoch)
    
    def get_latest_metrics(self, split: str = 'val') -> Dict[str, float]:
        """
        Get the latest metrics for a specific split.
        
        Args:
            split: Data split (default: 'val')
            
        Returns:
            Dictionary of latest metrics
        """
        if split not in self.metrics:
            raise ValueError(f"Invalid split: {split}")
        
        latest_metrics = {}
        
        for metric_name, values in self.metrics[split].items():
            if values:
                latest_metrics[metric_name] = values[-1]
        
        return latest_metrics
    
    def get_best_metrics(self, split: str = 'val') -> Dict[str, float]:
        """
        Get the best metrics for a specific split.
        
        Args:
            split: Data split (default: 'val')
            
        Returns:
            Dictionary of best metrics
        """
        if split not in self.best_metrics:
            raise ValueError(f"Invalid split: {split}")
        
        return self.best_metrics[split].copy()
    
    def plot_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot metrics over time.
        
        Args:
            metric_names: List of metrics to plot (default: None, plots all)
            save_dir: Directory to save plots (default: None, doesn't save)
            show: Whether to show plots (default: False)
        """
        # Create save directory if provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # Get all available metrics
        all_metrics = set()
        for split in self.metrics:
            all_metrics.update(self.metrics[split].keys())
        
        # Filter metrics if specified
        if metric_names is not None:
            metrics_to_plot = [m for m in metric_names if m in all_metrics]
        else:
            metrics_to_plot = list(all_metrics)
        
        # Plot each metric
        for metric_name in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            for split in self.metrics:
                if metric_name in self.metrics[split] and len(self.metrics[split][metric_name]) > 0:
                    # Get epochs for this split
                    epochs = self.epochs[split]
                    if not epochs:
                        # If no epochs provided, use range
                        epochs = list(range(len(self.metrics[split][metric_name])))
                    
                    # Plot metric values
                    plt.plot(
                        epochs, 
                        self.metrics[split][metric_name],
                        label=f"{split} {metric_name}"
                    )
            
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} over time')
            plt.legend()
            plt.grid(True)
            
            # Save plot if directory provided
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f'{metric_name}_plot.png'))
            
            # Show plot if requested
            if show:
                plt.show()
            else:
                plt.close()
    
    def get_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with splits, metrics, and values
        """
        summary = {
            'latest': {},
            'best': {}
        }
        
        for split in self.metrics:
            # Latest metrics
            latest = self.get_latest_metrics(split)
            if latest:
                summary['latest'][split] = latest
            
            # Best metrics
            best = self.get_best_metrics(split)
            if best:
                summary['best'][split] = best
        
        return summary
    
    def reset(self):
        """Reset the metrics tracker."""
        self.metrics = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        self.best_metrics = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        self.epochs = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def __repr__(self) -> str:
        """String representation of the metrics tracker."""
        return f"MetricsTracker(metrics={self.metrics})"
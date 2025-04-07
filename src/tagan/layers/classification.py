"""
Classification and prediction layers for TAGAN.

This module provides classification and prediction heads for various tasks,
along with specialized loss functions for temporal predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import math


class TemporalPredictionHead(nn.Module):
    """
    Prediction head for temporal data.
    
    This module takes temporal features and produces predictions,
    supporting various prediction tasks such as classification and regression.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        output_dim (int): Output dimension size
        task_type (str): Type of prediction task
        num_layers (int): Number of layers in the prediction head
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        task_type: str = 'classification',
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        use_time_embedding: bool = False,
        time_dim: int = 0
    ):
        """
        Initialize the temporal prediction head.
        
        Args:
            hidden_dim: Hidden dimension size
            output_dim: Output dimension size
            task_type: Type of prediction task (default: 'classification')
                     Options: 'classification', 'regression', 'multi_class',
                              'multi_label', 'sequence'
            num_layers: Number of layers in the prediction head (default: 2)
            dropout: Dropout probability (default: 0.1)
            activation: Activation function (default: 'relu')
            use_layer_norm: Whether to use layer normalization (default: True)
            use_time_embedding: Whether to use time embedding (default: False)
            time_dim: Dimension of time embedding (default: 0)
        """
        super(TemporalPredictionHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.use_time_embedding = use_time_embedding
        self.time_dim = time_dim
        
        # Define input dimension with optional time embedding
        input_dim = hidden_dim + time_dim if use_time_embedding else hidden_dim
        
        # Build layer architecture
        layers = []
        
        for i in range(num_layers):
            # First layer takes input_dim, others take hidden_dim
            in_features = input_dim if i == 0 else hidden_dim
            # Last layer outputs output_dim, others output hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            
            # Add linear layer
            layers.append(nn.Linear(in_features, out_features))
            
            # Add layer normalization if not the last layer
            if use_layer_norm and i < num_layers - 1:
                layers.append(nn.LayerNorm(out_features))
            
            # Add activation if not the last layer
            if i < num_layers - 1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'elu':
                    layers.append(nn.ELU())
                else:
                    layers.append(nn.ReLU())  # Default
                
                # Add dropout
                layers.append(nn.Dropout(dropout))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Output activation based on task type
        if task_type == 'classification' or task_type == 'multi_label':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'multi_class':
            self.output_activation = None  # Use softmax in loss function for numerical stability
        elif task_type == 'regression' or task_type == 'sequence':
            self.output_activation = None  # No activation for regression
        else:
            self.output_activation = None
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Glorot/Xavier initialization."""
        # Extract only the linear layers for more accurate identification
        linear_layers = [m for m in self.model if isinstance(m, nn.Linear)]
        num_linear_layers = len(linear_layers)
        
        # Apply to each linear layer in the model
        for i, m in enumerate(self.model):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Check if this is the final linear layer
                    is_final_layer = (i == linear_layers.index(linear_layers[-1]))
                    
                    # Initialize bias for classification layers
                    if is_final_layer and self.task_type == 'classification' and self.output_dim == 1:
                        # For binary classification, initialize with a positive bias
                        # to counter the observed negative bias in the output predictions
                        nn.init.constant_(m.bias, 0.5)  # Positive bias to counter the 0 F1 score issue
                    else:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the prediction head.
        
        Args:
            x: Input features [batch_size, ..., hidden_dim]
            time_embedding: Optional time embedding [batch_size, ..., time_dim]
            
        Returns:
            Predictions [batch_size, ..., output_dim]
        """
        # Concatenate time embedding if used
        if self.use_time_embedding and time_embedding is not None:
            x = torch.cat([x, time_embedding], dim=-1)
        
        # Apply model
        x = self.model(x)
        
        # Apply output activation if defined
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x
    
    def extra_repr(self) -> str:
        """Return a string representation of the head configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, '
                f'task_type={self.task_type}, '
                f'num_layers={self.num_layers}, '
                f'use_layer_norm={self.use_layer_norm}, '
                f'use_time_embedding={self.use_time_embedding}, '
                f'time_dim={self.time_dim}, '
                f'dropout={self.dropout}')


class MultiTaskPredictionHead(nn.Module):
    """
    Multi-task prediction head for temporal data.
    
    This module handles predictions for multiple tasks simultaneously,
    supporting different output dimensions and task types.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        task_heads (nn.ModuleDict): Dictionary of task-specific prediction heads
    """
    
    def __init__(
        self,
        hidden_dim: int,
        task_configs: Dict[str, Dict[str, Any]],
        shared_layers: int = 1,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        use_time_embedding: bool = False,
        time_dim: int = 0
    ):
        """
        Initialize the multi-task prediction head.
        
        Args:
            hidden_dim: Hidden dimension size
            task_configs: Dictionary mapping task names to configurations
                        Each configuration should include 'output_dim' and 'task_type'
            shared_layers: Number of shared layers before task-specific heads (default: 1)
            dropout: Dropout probability (default: 0.1)
            activation: Activation function (default: 'relu')
            use_layer_norm: Whether to use layer normalization (default: True)
            use_time_embedding: Whether to use time embedding (default: False)
            time_dim: Dimension of time embedding (default: 0)
        """
        super(MultiTaskPredictionHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.task_configs = task_configs
        self.shared_layers = shared_layers
        self.dropout = dropout
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.use_time_embedding = use_time_embedding
        self.time_dim = time_dim
        
        # Define input dimension with optional time embedding
        input_dim = hidden_dim + time_dim if use_time_embedding else hidden_dim
        
        # Build shared layers
        shared_network = []
        
        for i in range(shared_layers):
            # First layer takes input_dim, others take hidden_dim
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim
            
            # Add linear layer
            shared_network.append(nn.Linear(in_features, out_features))
            
            # Add layer normalization
            if use_layer_norm:
                shared_network.append(nn.LayerNorm(out_features))
            
            # Add activation
            if activation == 'relu':
                shared_network.append(nn.ReLU())
            elif activation == 'leaky_relu':
                shared_network.append(nn.LeakyReLU())
            elif activation == 'gelu':
                shared_network.append(nn.GELU())
            elif activation == 'elu':
                shared_network.append(nn.ELU())
            else:
                shared_network.append(nn.ReLU())  # Default
            
            # Add dropout
            shared_network.append(nn.Dropout(dropout))
        
        # Create shared model
        self.shared_network = nn.Sequential(*shared_network) if shared_network else nn.Identity()
        
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            output_dim = config.get('output_dim', 1)
            task_type = config.get('task_type', 'classification')
            num_layers = config.get('num_layers', 1)  # Task-specific layers
            
            self.task_heads[task_name] = TemporalPredictionHead(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                task_type=task_type,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm,
                use_time_embedding=False  # Already handled in shared network
            )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Glorot/Xavier initialization."""
        # Apply to each linear layer in the shared network
        for m in self.shared_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multi-task prediction head.
        
        Args:
            x: Input features [batch_size, ..., hidden_dim]
            time_embedding: Optional time embedding [batch_size, ..., time_dim]
            tasks: Optional list of tasks to compute predictions for (default: None, all tasks)
            
        Returns:
            Dictionary mapping task names to predictions
        """
        # Concatenate time embedding if used
        if self.use_time_embedding and time_embedding is not None:
            x = torch.cat([x, time_embedding], dim=-1)
        
        # Apply shared network
        shared_features = self.shared_network(x)
        
        # Determine which tasks to compute
        if tasks is None:
            tasks = list(self.task_heads.keys())
        
        # Compute predictions for each task
        predictions = {}
        for task_name in tasks:
            if task_name in self.task_heads:
                predictions[task_name] = self.task_heads[task_name](shared_features)
        
        return predictions
    
    def extra_repr(self) -> str:
        """Return a string representation of the head configuration."""
        task_info = ', '.join([
            f"{name}({config.get('task_type', 'unknown')})"
            for name, config in self.task_configs.items()
        ])
        
        return (f'hidden_dim={self.hidden_dim}, '
                f'shared_layers={self.shared_layers}, '
                f'tasks=[{task_info}], '
                f'use_layer_norm={self.use_layer_norm}, '
                f'use_time_embedding={self.use_time_embedding}, '
                f'time_dim={self.time_dim}, '
                f'dropout={self.dropout}')


class TemporalLossFunction:
    """
    Loss functions for temporal predictions.
    
    This class provides various loss functions tailored for temporal data,
    with special handling for temporal dependencies.
    
    Attributes:
        task_type (str): Type of prediction task
        reduction (str): Loss reduction method
        pos_weight (torch.Tensor): Weight for positive samples in binary classification
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        temporal_discount: float = 1.0,
        huber_delta: float = 1.0,
        quantile_tau: float = 0.5
    ):
        """
        Initialize the temporal loss function.
        
        Args:
            task_type: Type of prediction task (default: 'classification')
                     Options: 'classification', 'regression', 'multi_class',
                              'multi_label', 'sequence', 'focal', 'huber', 'quantile'
            reduction: Loss reduction method (default: 'mean')
                     Options: 'none', 'mean', 'sum'
            pos_weight: Weight for positive samples in binary classification (default: None)
            class_weights: Weights for classes in multi-class classification (default: None)
            focal_gamma: Gamma parameter for focal loss (default: 2.0)
            focal_alpha: Alpha parameter for focal loss balancing positive/negative classes (default: None)
            temporal_discount: Discount factor for temporal dependencies (default: 1.0)
            huber_delta: Delta parameter for Huber loss (default: 1.0)
            quantile_tau: Tau parameter for quantile loss (default: 0.5)
        """
        self.task_type = task_type
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.temporal_discount = temporal_discount
        self.huber_delta = huber_delta
        self.quantile_tau = quantile_tau
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        time_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Predicted values [batch_size, ..., output_dim]
            targets: Target values [batch_size, ..., output_dim]
            time_weights: Optional weights for time steps [batch_size, seq_len] (default: None)
            mask: Optional mask for valid entries [batch_size, ..., output_dim] (default: None)
            
        Returns:
            Loss value
        """
        # Handle shape mismatches between predictions and targets
        # For binary classification with output_dim=1, reshape as needed
        if self.task_type in ['classification', 'focal'] and predictions.size(-1) == 1:
            # Case: predictions is [batch_size, 1] and targets is [batch_size]
            if predictions.dim() == 2 and predictions.size(1) == 1 and targets.dim() == 1 and predictions.size(0) == targets.size(0):
                # Squeeze the last dimension of predictions to match targets
                predictions = predictions.squeeze(-1)
            # Case: predictions is [1, 1] and targets is [batch_size]
            elif predictions.size(0) == 1 and targets.dim() == 1:
                # First squeeze to [1] then expand to [batch_size]
                predictions = predictions.squeeze(-1).expand(targets.size(0))
            # Case: predictions is [batch_size] and targets is [batch_size, 1]
            elif predictions.dim() == 1 and targets.dim() == 2 and predictions.size(0) == targets.size(0):
                targets = targets.squeeze(-1)
            # Case: predictions is [1, 1] and targets is [batch_size, 1]
            elif predictions.size(0) == 1 and targets.dim() == 2:
                # Expand to [batch_size, 1]
                predictions = predictions.expand(targets.size(0), -1)
            # Case: predictions is [batch_size, num_classes] and targets is [batch_size]
            # This is normal for multi-class classification - no reshape needed
            elif predictions.dim() == 2 and targets.dim() == 1 and predictions.size(0) == targets.size(0) and predictions.size(1) > 1:
                # This is the expected format for cross-entropy loss - no reshaping needed
                pass
        
        # Special handling for multi-class classification
        if self.task_type in ['multi_class', 'classification'] and predictions.dim() == 2 and predictions.size(1) > 1 and targets.dim() == 1:
            # For cross-entropy, this is the expected format - don't raise shape mismatch
            pass
        # Final shape check after adjustments for other cases
        elif predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} does not match targets shape {targets.shape} after attempted reshaping")
        
        # Binary classification (BCE loss)
        if self.task_type == 'classification':
            loss = F.binary_cross_entropy_with_logits(
                predictions, targets, pos_weight=self.pos_weight, reduction='none'
            )
        
        # Multi-class classification (Cross Entropy loss)
        elif self.task_type == 'multi_class':
            # For multi-class, targets should be class indices [batch_size, ...]
            # and predictions should be logits [batch_size, ..., num_classes]
            
            # Check if the last dimensions match
            if predictions.size(-1) == targets.size(-1):
                # Targets are one-hot encoded, convert to class indices
                targets = targets.argmax(dim=-1)
            
            # Reshape for CE loss
            pred_shape = predictions.shape
            preds_reshaped = predictions.view(-1, pred_shape[-1])
            targets_reshaped = targets.view(-1)
            
            loss = F.cross_entropy(
                preds_reshaped, targets_reshaped,
                weight=self.class_weights, reduction='none'
            )
            
            # Reshape loss back to original shape (excluding last dimension of predictions)
            loss = loss.view(*pred_shape[:-1])
        
        # Multi-label classification (BCE loss for each label)
        elif self.task_type == 'multi_label':
            loss = F.binary_cross_entropy_with_logits(
                predictions, targets, reduction='none'
            )
        
        # Regression (MSE loss)
        elif self.task_type == 'regression':
            loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Sequence prediction (combination of losses across sequence)
        elif self.task_type == 'sequence':
            # Default to MSE for sequence prediction
            loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Focal loss for imbalanced classification
        elif self.task_type == 'focal':
            # Compute probabilities
            if predictions.size(-1) == 1:
                # Binary case
                probs = torch.sigmoid(predictions)
                p_t = torch.where(targets == 1, probs, 1 - probs)
                
                # Alpha term adjusts for class imbalance
                if self.focal_alpha is not None:
                    alpha_t = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
                else:
                    alpha_t = torch.ones_like(p_t)
            else:
                # Multi-class case
                probs = F.softmax(predictions, dim=-1)
                
                # Convert targets to one-hot if they are class indices
                if targets.dim() == predictions.dim() - 1:
                    targets_one_hot = F.one_hot(targets, num_classes=predictions.size(-1))
                    targets_one_hot = targets_one_hot.float()
                else:
                    targets_one_hot = targets
                
                # Get probability of the target class
                p_t = torch.sum(probs * targets_one_hot, dim=-1)
                
                # Apply class-specific alpha if provided
                if self.focal_alpha is not None:
                    # If focal_alpha is a single value (non-Tensor), apply uniformly
                    if not isinstance(self.focal_alpha, torch.Tensor):
                        alpha_t = torch.ones_like(p_t) * self.focal_alpha
                    else:
                        # If focal_alpha is a tensor of weights per class, select the right alpha for each target
                        alpha_t = torch.sum(self.focal_alpha.unsqueeze(0) * targets_one_hot, dim=-1)
                else:
                    alpha_t = torch.ones_like(p_t)
            
            # Compute focal factor (focusing on hard examples)
            focal_weight = (1 - p_t) ** self.focal_gamma
            
            # Compute binary or cross-entropy loss
            if predictions.size(-1) == 1:
                loss = F.binary_cross_entropy_with_logits(
                    predictions, targets, reduction='none'
                )
            else:
                # Use class_weights alongside focal_alpha (they have different purposes)
                loss = F.cross_entropy(
                    predictions, targets, weight=self.class_weights, reduction='none'
                )
            
            # Apply both focal weight and alpha balancing
            loss = alpha_t * focal_weight * loss
        
        # Huber loss for regression with outlier robustness
        elif self.task_type == 'huber':
            loss = F.smooth_l1_loss(
                predictions, targets, beta=self.huber_delta, reduction='none'
            )
        
        # Quantile loss for quantile regression
        elif self.task_type == 'quantile':
            # Compute quantile loss: tau * (y - pred) if y > pred else (1 - tau) * (pred - y)
            diff = targets - predictions
            loss = torch.max(self.quantile_tau * diff, (self.quantile_tau - 1) * diff)
        
        # Default to MSE
        else:
            loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Apply time weights if provided
        if time_weights is not None:
            # Check dimensions
            if time_weights.dim() < loss.dim():
                # Reshape time_weights to match loss dimensions
                for _ in range(loss.dim() - time_weights.dim()):
                    time_weights = time_weights.unsqueeze(-1)
            
            # Apply weights
            loss = loss * time_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            # If mask is provided, compute mean over masked elements
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class TemporalLossModule(nn.Module):
    """
    Multi-task temporal loss module.
    
    This module computes combined losses for multiple prediction tasks,
    with support for customized loss functions and task weights.
    
    Attributes:
        loss_functions (Dict[str, TemporalLossFunction]): Loss functions for each task
        task_weights (Dict[str, float]): Weight for each task in the combined loss
        default_loss_fn (TemporalLossFunction): Default loss function
    """
    
    def __init__(
        self,
        task_configs: Dict[str, Dict[str, Any]],
        loss_config: Optional[Dict[str, Any]] = None,
        default_task_type: str = 'classification',
        default_reduction: str = 'mean'
    ):
        """
        Initialize the multi-task temporal loss module.
        
        Args:
            task_configs: Dictionary mapping task names to configurations
                        Each configuration should include 'task_type' and optionally
                        'loss_weight', 'pos_weight', 'class_weights', etc.
            loss_config: Optional global loss configuration that applies to all tasks
                       Can include 'reduction', 'focal_alpha', 'focal_gamma', etc.
            default_task_type: Default task type for tasks without 'task_type' (default: 'classification')
            default_reduction: Default reduction method (default: 'mean')
        """
        super(TemporalLossModule, self).__init__()
        
        self.task_configs = task_configs
        self.loss_config = loss_config or {}
        self.default_task_type = default_task_type
        self.default_reduction = self.loss_config.get('reduction', default_reduction)
        
        # Create loss functions for each task
        self.loss_functions = {}
        self.task_weights = {}
        
        for task_name, config in task_configs.items():
            # Combine task-specific config with global loss_config
            # Task-specific settings take precedence over global settings
            task_type = config.get('task_type', default_task_type)
            reduction = config.get('reduction', self.default_reduction)
            
            # Get loss parameters, prioritizing task-specific configs over global loss_config
            pos_weight = config.get('pos_weight', self.loss_config.get('pos_weight'))
            class_weights = config.get('class_weights', self.loss_config.get('class_weights'))
            focal_gamma = config.get('focal_gamma', self.loss_config.get('focal_gamma', 2.0))
            focal_alpha = config.get('focal_alpha', self.loss_config.get('focal_alpha'))
            temporal_discount = config.get('temporal_discount', self.loss_config.get('temporal_discount', 1.0))
            huber_delta = config.get('huber_delta', self.loss_config.get('huber_delta', 1.0))
            quantile_tau = config.get('quantile_tau', self.loss_config.get('quantile_tau', 0.5))
            
            # Convert pos_weight and class_weights to tensors if provided
            if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight)
            
            if class_weights is not None and not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights)
            
            # Create loss function
            self.loss_functions[task_name] = TemporalLossFunction(
                task_type=task_type,
                reduction=reduction,
                pos_weight=pos_weight,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                temporal_discount=temporal_discount,
                huber_delta=huber_delta,
                quantile_tau=quantile_tau
            )
            
            # Set task weight
            self.task_weights[task_name] = config.get('loss_weight', 1.0)
        
        # Create default loss function
        self.default_loss_fn = TemporalLossFunction(
            task_type=default_task_type,
            reduction=default_reduction
        )
    
    def forward(
        self,
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        time_weights: Optional[Dict[str, torch.Tensor]] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        return_task_losses: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for the loss module.
        
        Args:
            predictions: Predicted values, either a tensor or dictionary mapping tasks to tensors
            targets: Target values, either a tensor or dictionary mapping tasks to tensors
            time_weights: Optional dictionary mapping tasks to time weights (default: None)
            masks: Optional dictionary mapping tasks to masks (default: None)
            return_task_losses: Whether to return individual task losses (default: False)
            
        Returns:
            - If return_task_losses is False: Combined loss
            - If return_task_losses is True: Tuple of (combined loss, task losses)
        """
        # Check if predictions and targets are dictionaries or direct tensors
        if isinstance(predictions, dict) and isinstance(targets, dict):
            # Dictionary case: compute loss for each task
            task_losses = {}
            
            for task_name, task_pred in predictions.items():
                if task_name in targets:
                    # Get loss function for the task
                    loss_fn = self.loss_functions.get(task_name)
                    
                    if loss_fn is not None:
                        # Get time weights and mask for the task if provided
                        task_time_weights = time_weights.get(task_name) if time_weights else None
                        task_mask = masks.get(task_name) if masks else None
                        
                        # Compute task loss
                        task_loss = loss_fn(
                            task_pred, targets[task_name],
                            time_weights=task_time_weights,
                            mask=task_mask
                        )
                        
                        # Apply task weight
                        task_weight = self.task_weights.get(task_name, 1.0)
                        weighted_loss = task_weight * task_loss
                        task_losses[task_name] = weighted_loss
        else:
            # Direct tensor case: use default loss function
            task_losses = {"default": self.default_loss_fn(predictions, targets)}
            
        # Compute combined loss
        combined_loss = sum(task_losses.values())
        
        if return_task_losses:
            return combined_loss, task_losses
        else:
            return combined_loss


class TemporalClassificationHead(nn.Module):
    """
    Classification head for temporal data.
    
    This module takes temporal embeddings and produces classification outputs,
    with special handling for temporal dynamics.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        num_classes (int): Number of classes
        pooling_type (str): How to pool temporal information
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        pooling_type: str = 'attention',
        dropout: float = 0.1,
        activation: str = 'relu',
        num_layers: int = 2,
        use_layer_norm: bool = True,
        multi_label: bool = False,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize the temporal classification head.
        
        Args:
            hidden_dim: Hidden dimension size
            num_classes: Number of classes
            pooling_type: How to pool temporal information (default: 'attention')
                        Options: 'attention', 'mean', 'max', 'last', 'first'
            dropout: Dropout probability (default: 0.1)
            activation: Activation function (default: 'relu')
            num_layers: Number of layers in the classifier (default: 2)
            use_layer_norm: Whether to use layer normalization (default: True)
            multi_label: Whether to perform multi-label classification (default: False)
            class_weights: Optional class weights for loss computation (default: None)
        """
        super(TemporalClassificationHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.multi_label = multi_label
        
        # Temporal pooling
        if pooling_type == 'attention':
            # Attention-based pooling
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        
        # Classifier
        classifier_layers = []
        
        for i in range(num_layers):
            # First layer takes hidden_dim, last layer outputs num_classes
            in_features = hidden_dim if i == 0 else hidden_dim
            out_features = num_classes if i == num_layers - 1 else hidden_dim
            
            # Add linear layer
            classifier_layers.append(nn.Linear(in_features, out_features))
            
            # Add normalization, activation, and dropout except for the last layer
            if i < num_layers - 1:
                # Add layer normalization
                if use_layer_norm:
                    classifier_layers.append(nn.LayerNorm(out_features))
                
                # Add activation
                if activation == 'relu':
                    classifier_layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    classifier_layers.append(nn.LeakyReLU())
                elif activation == 'gelu':
                    classifier_layers.append(nn.GELU())
                elif activation == 'elu':
                    classifier_layers.append(nn.ELU())
                else:
                    classifier_layers.append(nn.ReLU())  # Default
                
                # Add dropout
                classifier_layers.append(nn.Dropout(dropout))
        
        # Create classifier
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Loss function
        if multi_label:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier/Glorot initialization."""
        # Apply to linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _pool_temporal(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool temporal information.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] (default: None)
            
        Returns:
            Pooled tensor [batch_size, hidden_dim]
        """
        if self.pooling_type == 'mean':
            # Mean pooling
            if mask is not None:
                # Apply mask and compute mean
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                output = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-10)
            else:
                output = x.mean(dim=1)
        
        elif self.pooling_type == 'max':
            # Max pooling
            if mask is not None:
                # Apply mask (set padded positions to large negative value)
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded + (1 - mask_expanded) * -1e9
                output, _ = x_masked.max(dim=1)
            else:
                output, _ = x.max(dim=1)
        
        elif self.pooling_type == 'last':
            # Last state
            if mask is not None:
                # Get the last valid position for each sequence
                seq_lengths = mask.sum(dim=1, keepdim=True).long() - 1
                seq_lengths = torch.clamp(seq_lengths, min=0)
                batch_size = x.size(0)
                
                # Gather the last valid state for each sequence
                indices = seq_lengths.view(-1, 1).expand(-1, x.size(-1))
                output = x[torch.arange(batch_size).unsqueeze(1), indices].squeeze(1)
            else:
                output = x[:, -1]
        
        elif self.pooling_type == 'first':
            # First state
            output = x[:, 0]
        
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            # [batch_size, seq_len, 1]
            attention_scores = self.attention(x)
            
            if mask is not None:
                # Apply mask (set padded positions to large negative value)
                mask_expanded = mask.unsqueeze(-1).float()
                attention_scores = attention_scores * mask_expanded + (1 - mask_expanded) * -1e9
            
            # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # [batch_size, hidden_dim]
            output = (x * attention_weights).sum(dim=1)
        
        else:
            # Default to mean pooling
            output = x.mean(dim=1)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the classification head.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] (default: None)
            labels: Optional labels for loss computation [batch_size, (num_classes)] (default: None)
            
        Returns:
            If labels is None:
                Logits [batch_size, num_classes]
            If labels is provided:
                Tuple of (loss, logits)
        """
        # Pool temporal information
        pooled = self._pool_temporal(x, mask)
        
        # Apply classifier
        logits = self.classifier(pooled)
        
        # Compute loss if labels provided
        if labels is not None:
            # For multi-label, ensure labels have same shape as logits
            if self.multi_label and labels.dim() == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes).float()
            
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits
    
    def extra_repr(self) -> str:
        """Return a string representation of the head configuration."""
        return (f'hidden_dim={self.hidden_dim}, '
                f'num_classes={self.num_classes}, '
                f'pooling_type={self.pooling_type}, '
                f'multi_label={self.multi_label}, '
                f'num_layers={self.num_layers}, '
                f'dropout={self.dropout}')


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for handling imbalanced classification.
    
    This loss is designed for multi-label classification with class imbalance,
    applying different focus to positive and negative samples.
    
    Attributes:
        gamma_pos (float): Focusing parameter for positive samples
        gamma_neg (float): Focusing parameter for negative samples
        clip (float): Clipping value for numerical stability
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        """
        Initialize the asymmetric focal loss.
        
        Args:
            gamma_pos: Focusing parameter for positive samples (default: 0.0)
            gamma_neg: Focusing parameter for negative samples (default: 4.0)
            clip: Clipping value for probabilities (default: 0.05)
            reduction: Loss reduction method (default: 'mean')
            eps: Small constant for numerical stability (default: 1e-8)
        """
        super(AsymmetricFocalLoss, self).__init__()
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute asymmetric focal loss.
        
        Args:
            predictions: Predicted logits [batch_size, ..., num_classes]
            targets: Binary targets [batch_size, ..., num_classes]
            weights: Optional sample weights [batch_size, ...] (default: None)
            
        Returns:
            Loss value
        """
        # Ensure predictions and targets have same shape
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}")
        
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Clipping probabilities
        if self.clip > 0:
            probs = torch.clamp(probs, min=self.clip, max=1.0 - self.clip)
        
        # Separate positive and negative targets
        pos_mask = targets.eq(1).float()
        neg_mask = targets.eq(0).float()
        
        # Compute positive and negative losses
        pos_loss = pos_mask * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs + self.eps)
        neg_loss = neg_mask * torch.pow(probs, self.gamma_neg) * torch.log(1 - probs + self.eps)
        
        # Combine losses
        loss = -(pos_loss + neg_loss)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ClassificationModule(nn.Module):
    """
    Classification module for TAGAN.
    
    This module provides a unified interface for classification tasks,
    supporting both single-task and multi-task scenarios.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        multi_task (bool): Whether to use multi-task classification
    """
    
    def __init__(
        self,
        hidden_dim: int,
        task_configs: Union[Dict[str, Any], int],
        pooling_type: str = 'attention',
        dropout: float = 0.1,
        activation: str = 'relu',
        num_layers: int = 2,
        use_layer_norm: bool = True,
        multi_task: bool = False,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize the classification module.
        
        Args:
            hidden_dim: Hidden dimension size
            task_configs: Either number of classes (int) for single-task or
                         dictionary of task configurations for multi-task
            pooling_type: How to pool temporal information (default: 'attention')
                        Options: 'attention', 'mean', 'max', 'last', 'first'
            dropout: Dropout probability (default: 0.1)
            activation: Activation function (default: 'relu')
            num_layers: Number of layers in the classifier (default: 2)
            use_layer_norm: Whether to use layer normalization (default: True)
            multi_task: Whether to use multi-task classification (default: False)
            class_weights: Optional class weights for loss computation (default: None)
        """
        super(ClassificationModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.multi_task = multi_task
        
        # Handle task_configs parameter
        if isinstance(task_configs, int):
            # Single-task case with num_classes as input
            num_classes = task_configs
            self.task_configs = {'default': {'output_dim': num_classes, 'task_type': 'classification'}}
        elif isinstance(task_configs, dict):
            # Either single task with output_dim and task_type specified
            # or multi-task with multiple task configurations
            if 'output_dim' in task_configs:
                # Single task specified as dict
                self.task_configs = {'default': task_configs}
            else:
                # Multi-task configuration
                self.task_configs = task_configs
        else:
            raise ValueError("task_configs must be either int (num_classes) or dict")
        
        # Create appropriate head based on multi_task flag
        if multi_task:
            # Create multi-task prediction head
            self.classification_head = MultiTaskPredictionHead(
                hidden_dim=hidden_dim,
                task_configs=self.task_configs,
                shared_layers=1,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm
            )
        else:
            # Get output_dim for single task
            if 'default' in self.task_configs:
                task_config = self.task_configs['default']
            else:
                # Just take the first task if 'default' not available
                task_name = next(iter(self.task_configs))
                task_config = self.task_configs[task_name]
            
            output_dim = task_config.get('output_dim', 1)
            task_type = task_config.get('task_type', 'classification')
            
            # Determine if multi-label based on task_type
            multi_label = task_type == 'multi_label'
            
            self.classification_head = TemporalClassificationHead(
                hidden_dim=hidden_dim,
                num_classes=output_dim,
                pooling_type=pooling_type,
                dropout=dropout,
                activation=activation,
                num_layers=num_layers,
                use_layer_norm=use_layer_norm,
                multi_label=multi_label,
                class_weights=class_weights
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        tasks: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple, Dict[str, Tuple]]:
        """
        Forward pass for the classification module.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] (default: None)
            labels: Optional labels for loss computation (default: None)
                   For single-task: [batch_size, (num_classes)]
                   For multi-task: Dict mapping task names to labels
            tasks: Optional list of tasks to compute predictions for (multi-task only)
            
        Returns:
            For single-task:
                If labels is None: Logits [batch_size, num_classes]
                If labels is provided: Tuple of (loss, logits)
            For multi-task:
                If labels is None: Dict mapping task names to logits
                If labels is provided: Dict mapping task names to tuples of (loss, logits)
        """
        if self.multi_task:
            # Multi-task forward pass
            predictions = self.classification_head(x, mask=mask, tasks=tasks)
            
            if labels is not None and isinstance(labels, dict):
                # Compute losses for each task
                losses = {}
                results = {}
                
                for task_name, task_preds in predictions.items():
                    if task_name in labels:
                        task_labels = labels[task_name]
                        # For now, we don't have a built-in loss in MultiTaskPredictionHead
                        # So we manually compute the loss
                        if self.task_configs[task_name].get('task_type') in ['classification', 'multi_label']:
                            loss = F.binary_cross_entropy_with_logits(task_preds, task_labels)
                        elif self.task_configs[task_name].get('task_type') == 'multi_class':
                            loss = F.cross_entropy(task_preds, task_labels)
                        else:  # regression, etc.
                            loss = F.mse_loss(task_preds, task_labels)
                        
                        losses[task_name] = loss
                        results[task_name] = (loss, task_preds)
                    else:
                        results[task_name] = task_preds
                
                if losses:
                    # Compute combined loss
                    combined_loss = sum(losses.values())
                    return combined_loss, predictions, results
                
                return predictions
            
            return predictions
        else:
            # Single-task forward pass
            return self.classification_head(x, mask, labels)


class RegressionModule(nn.Module):
    """
    Regression module for TAGAN.
    
    This module provides a unified interface for regression tasks,
    supporting different pooling strategies and loss functions.
    
    Attributes:
        hidden_dim (int): Hidden dimension size
        output_dim (int): Output dimension size
        pooling_type (str): How to pool temporal information
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        pooling_type: str = 'attention',
        dropout: float = 0.1,
        activation: str = 'relu',
        num_layers: int = 2,
        use_layer_norm: bool = True,
        loss_type: str = 'mse',
        huber_delta: float = 1.0
    ):
        """
        Initialize the regression module.
        
        Args:
            hidden_dim: Hidden dimension size
            output_dim: Output dimension size (default: 1)
            pooling_type: How to pool temporal information (default: 'attention')
                        Options: 'attention', 'mean', 'max', 'last', 'first'
            dropout: Dropout probability (default: 0.1)
            activation: Activation function (default: 'relu')
            num_layers: Number of layers in the regressor (default: 2)
            use_layer_norm: Whether to use layer normalization (default: True)
            loss_type: Type of regression loss (default: 'mse')
                      Options: 'mse', 'mae', 'huber', 'quantile'
            huber_delta: Delta parameter for Huber loss (default: 1.0)
        """
        super(RegressionModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
        # Temporal pooling
        if pooling_type == 'attention':
            # Attention-based pooling
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        
        # Regression layers
        regressor_layers = []
        
        for i in range(num_layers):
            # First layer takes hidden_dim, last layer outputs output_dim
            in_features = hidden_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            
            # Add linear layer
            regressor_layers.append(nn.Linear(in_features, out_features))
            
            # Add normalization, activation, and dropout except for the last layer
            if i < num_layers - 1:
                # Add layer normalization
                if use_layer_norm:
                    regressor_layers.append(nn.LayerNorm(out_features))
                
                # Add activation
                if activation == 'relu':
                    regressor_layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    regressor_layers.append(nn.LeakyReLU())
                elif activation == 'gelu':
                    regressor_layers.append(nn.GELU())
                elif activation == 'elu':
                    regressor_layers.append(nn.ELU())
                else:
                    regressor_layers.append(nn.ReLU())  # Default
                
                # Add dropout
                regressor_layers.append(nn.Dropout(dropout))
        
        # Create regressor
        self.regressor = nn.Sequential(*regressor_layers)
        
        # Set loss function
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(beta=huber_delta, reduction='mean')
        else:
            self.loss_fn = nn.MSELoss(reduction='mean')  # Default
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier/Glorot initialization."""
        # Apply to linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _pool_temporal(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool temporal information.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] (default: None)
            
        Returns:
            Pooled tensor [batch_size, hidden_dim]
        """
        if self.pooling_type == 'mean':
            # Mean pooling
            if mask is not None:
                # Apply mask and compute mean
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                output = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-10)
            else:
                output = x.mean(dim=1)
        
        elif self.pooling_type == 'max':
            # Max pooling
            if mask is not None:
                # Apply mask (set padded positions to large negative value)
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded + (1 - mask_expanded) * -1e9
                output, _ = x_masked.max(dim=1)
            else:
                output, _ = x.max(dim=1)
        
        elif self.pooling_type == 'last':
            # Last state
            if mask is not None:
                # Get the last valid position for each sequence
                seq_lengths = mask.sum(dim=1, keepdim=True).long() - 1
                seq_lengths = torch.clamp(seq_lengths, min=0)
                batch_size = x.size(0)
                
                # Gather the last valid state for each sequence
                indices = seq_lengths.view(-1, 1).expand(-1, x.size(-1))
                output = x[torch.arange(batch_size).unsqueeze(1), indices].squeeze(1)
            else:
                output = x[:, -1]
        
        elif self.pooling_type == 'first':
            # First state
            output = x[:, 0]
        
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            # [batch_size, seq_len, 1]
            attention_scores = self.attention(x)
            
            if mask is not None:
                # Apply mask (set padded positions to large negative value)
                mask_expanded = mask.unsqueeze(-1).float()
                attention_scores = attention_scores * mask_expanded + (1 - mask_expanded) * -1e9
            
            # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # [batch_size, hidden_dim]
            output = (x * attention_weights).sum(dim=1)
        
        else:
            # Default to mean pooling
            output = x.mean(dim=1)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the regression module.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] (default: None)
            targets: Optional targets for loss computation [batch_size, output_dim] (default: None)
            
        Returns:
            If targets is None:
                Predictions [batch_size, output_dim]
            If targets is provided:
                Tuple of (loss, predictions)
        """
        # Pool temporal information
        pooled = self._pool_temporal(x, mask)
        
        # Apply regressor
        predictions = self.regressor(pooled)
        
        # Compute loss if targets provided
        if targets is not None:
            loss = self.loss_fn(predictions, targets)
            return loss, predictions
        else:
            return predictions
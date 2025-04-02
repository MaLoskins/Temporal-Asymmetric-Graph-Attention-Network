"""
Configuration utilities for TAGAN.

This module provides configuration classes for setting up the TAGAN model
with different parameters.
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Union, List


class TAGANConfig:
    """
    Configuration class for TAGAN model.
    
    This class stores all parameters for initializing and training the TAGAN model.
    
    Attributes:
        # Model architecture parameters
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of geometric attention layers
        num_heads (int): Number of attention heads
        temporal_attention_dim (int): Dimension for temporal attention
        
        # Feature dimensions
        node_feature_dim (int): Node feature dimension
        edge_feature_dim (int): Edge feature dimension
        output_dim (int): Output dimension (number of classes)
        
        # Training parameters
        learning_rate (float): Learning rate for optimization
        weight_decay (float): L2 regularization weight
        dropout (float): Dropout probability
        memory_decay_factor (float): Decay factor for memory bank
        num_epochs (int): Number of training epochs
        
        # Architecture options
        use_layer_norm (bool): Whether to use layer normalization
        edge_importance (bool): Whether to use edge importance weighting
        gru_bias (bool): Whether to use bias in GRU cell
        leaky_relu_slope (float): Slope parameter for LeakyReLU
        
        # Loss parameters
        loss_type (str): Type of loss function to use
        focal_alpha (float): Alpha parameter for focal loss
        focal_gamma (float): Gamma parameter for focal loss
        
        # Other parameters
        device (str): Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        # Model architecture parameters
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        temporal_attention_dim: int = 64,
        
        # Feature dimensions
        node_feature_dim: int = 16,
        edge_feature_dim: int = 0,
        output_dim: int = 2,
        
        # Training parameters
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        memory_decay_factor: float = 0.8,
        max_inactivity: int = 5,
        gradient_clip_val: float = 1.0,
        
        # Architecture options
        use_layer_norm: bool = True,
        edge_importance: bool = True,
        gru_bias: bool = True,
        leaky_relu_slope: float = 0.2,
        use_edge_features: bool = False,
        concat_heads: bool = True,
        learnable_distance: bool = False,
        
        # Temporal propagation options
        time_aware: bool = True,
        bidirectional: bool = False,
        use_skip_connection: bool = True,
        use_gating: bool = True,
        temporal_window_size: int = 3,
        aggregation_method: str = 'mean',
        use_residual: bool = True,
        
        # Temporal attention options
        causal_attention: bool = False,
        asymmetric_temporal_bias: bool = True,
        window_size: int = 5,
        
        # Loss parameters
        loss_type: str = 'ce',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        
        # Training parameters
        num_epochs: int = 50,
        
        # Other parameters
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the TAGAN configuration.
        
        Args:
            hidden_dim: Hidden dimension size (default: 64)
            num_layers: Number of geometric attention layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            temporal_attention_dim: Dimension for temporal attention (default: 64)
            node_feature_dim: Node feature dimension (default: 16)
            edge_feature_dim: Edge feature dimension (default: 0)
            output_dim: Output dimension (default: 2)
            learning_rate: Learning rate for optimization (default: 0.001)
            weight_decay: L2 regularization weight (default: 1e-5)
            dropout: Dropout probability (default: 0.1)
            memory_decay_factor: Decay factor for memory bank (default: 0.8)
            max_inactivity: Maximum number of inactive steps before pruning (default: 5)
            use_layer_norm: Whether to use layer normalization (default: True)
            edge_importance: Whether to use edge importance weighting (default: True)
            gru_bias: Whether to use bias in GRU cell (default: True)
            leaky_relu_slope: Slope parameter for LeakyReLU (default: 0.2)
            use_edge_features: Whether to use edge features (default: False)
            concat_heads: Whether to concatenate multi-head outputs (default: True)
            loss_type: Type of loss function to use (default: 'ce')
            focal_alpha: Alpha parameter for focal loss (default: 0.25)
            focal_gamma: Gamma parameter for focal loss (default: 2.0)
            num_epochs: Number of training epochs (default: 50)
            device: Device to use (default: 'cuda' if available else 'cpu')
        """
        # Model architecture parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.temporal_attention_dim = temporal_attention_dim
        
        # Feature dimensions
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim if use_edge_features else 0
        self.output_dim = output_dim
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.memory_decay_factor = memory_decay_factor
        self.max_inactivity = max_inactivity
        self.gradient_clip_val = gradient_clip_val
        self.num_epochs = num_epochs
        
        # Architecture options
        self.use_layer_norm = use_layer_norm
        self.edge_importance = edge_importance
        self.gru_bias = gru_bias
        self.leaky_relu_slope = leaky_relu_slope
        self.use_edge_features = use_edge_features
        self.concat_heads = concat_heads
        self.learnable_distance = learnable_distance
        
        # Temporal propagation options
        self.time_aware = time_aware
        self.bidirectional = bidirectional
        self.use_skip_connection = use_skip_connection
        self.use_gating = use_gating
        self.temporal_window_size = temporal_window_size
        self.aggregation_method = aggregation_method
        self.use_residual = use_residual
        
        # Temporal attention options
        self.causal_attention = causal_attention
        self.asymmetric_temporal_bias = asymmetric_temporal_bias
        self.window_size = window_size
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Other parameters
        self.device = device
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate the configuration parameters."""
        # Check hidden dimension
        if self.hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {self.hidden_dim}")
        
        # Check number of layers
        if self.num_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got {self.num_layers}")
        
        # Check number of heads
        if self.num_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {self.num_heads}")
        
        # Check temporal attention dimension
        if self.temporal_attention_dim <= 0:
            raise ValueError(f"Temporal attention dimension must be positive, got {self.temporal_attention_dim}")
        
        # Check feature dimensions
        if self.node_feature_dim <= 0:
            raise ValueError(f"Node feature dimension must be positive, got {self.node_feature_dim}")
        if self.edge_feature_dim < 0:
            raise ValueError(f"Edge feature dimension must be non-negative, got {self.edge_feature_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"Output dimension must be positive, got {self.output_dim}")
        
        # Check training parameters
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {self.dropout}")
        if not 0 < self.memory_decay_factor <= 1:
            raise ValueError(f"Memory decay factor must be in (0, 1], got {self.memory_decay_factor}")
        if self.max_inactivity <= 0:
            raise ValueError(f"Maximum inactivity must be positive, got {self.max_inactivity}")
        if self.gradient_clip_val < 0:
            raise ValueError(f"Gradient clip value must be non-negative, got {self.gradient_clip_val}")
        
        # Check architecture options
        if self.leaky_relu_slope <= 0:
            raise ValueError(f"LeakyReLU slope must be positive, got {self.leaky_relu_slope}")
        
        # Check loss parameters
        valid_loss_types = ['ce', 'bce', 'mse', 'focal']
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"Loss type must be one of {valid_loss_types}, got {self.loss_type}")
        if self.focal_alpha <= 0 or self.focal_alpha >= 1:
            raise ValueError(f"Focal alpha must be in (0, 1), got {self.focal_alpha}")
        if self.focal_gamma <= 0:
            raise ValueError(f"Focal gamma must be positive, got {self.focal_gamma}")
        
        # Check device
        if self.device not in ['cpu', 'cuda']:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got {self.device}")
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA is not available, falling back to CPU")
            self.device = 'cpu'
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Keyword arguments to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        
        # Validate updated configuration
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration parameters
        """
        config_dict = {}
        
        for key, value in self.__dict__.items():
            config_dict[key] = value
        
        return config_dict
    
    def save(self, filepath: str):
        """
        Save configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Convert to dictionary and save as JSON
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TAGANConfig':
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            TAGANConfig instance
        """
        # Create new instance with default parameters
        config = cls()
        
        # Update with provided parameters
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        
        # Validate configuration
        config.validate()
        
        return config
    
    @classmethod
    def load(cls, filepath: str) -> 'TAGANConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            TAGANConfig instance
        """
        # Load JSON file
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create configuration from dictionary
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        config_str = "TAGANConfig(\n"
        
        for key, value in sorted(self.__dict__.items()):
            config_str += f"  {key}={value},\n"
        
        config_str += ")"
        
        return config_str


class TAGANExperimentConfig:
    """
    Configuration class for TAGAN experiments.
    
    This class extends TAGANConfig with additional parameters for
    running experiments, such as data paths, training settings, etc.
    
    Attributes:
        model_config (TAGANConfig): Model configuration
        experiment_name (str): Name of the experiment
        data_path (str): Path to the dataset
        output_path (str): Path to save outputs
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        early_stopping (bool): Whether to use early stopping
        patience (int): Patience for early stopping
        num_folds (int): Number of cross-validation folds
        random_seed (int): Random seed for reproducibility
        eval_interval (int): Interval for evaluation during training
        save_interval (int): Interval for saving checkpoints
        num_workers (int): Number of data loading workers
    """
    
    def __init__(
        self,
        model_config: Optional[TAGANConfig] = None,
        experiment_name: str = 'tagan_experiment',
        data_path: str = 'data/',
        output_path: str = 'outputs/',
        num_epochs: int = 100,
        batch_size: int = 32,
        early_stopping: bool = True,
        patience: int = 10,
        num_folds: int = 5,
        random_seed: int = 42,
        eval_interval: int = 1,
        save_interval: int = 10,
        num_workers: int = 4,
        scheduler_type: str = 'step',
        scheduler_step_size: int = 20,
        scheduler_gamma: float = 0.5
    ):
        """
        Initialize the experiment configuration.
        
        Args:
            model_config: Model configuration (default: None, creates default config)
            experiment_name: Name of the experiment (default: 'tagan_experiment')
            data_path: Path to the dataset (default: 'data/')
            output_path: Path to save outputs (default: 'outputs/')
            num_epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 32)
            early_stopping: Whether to use early stopping (default: True)
            patience: Patience for early stopping (default: 10)
            num_folds: Number of cross-validation folds (default: 5)
            random_seed: Random seed for reproducibility (default: 42)
            eval_interval: Interval for evaluation during training (default: 1)
            save_interval: Interval for saving checkpoints (default: 10)
            num_workers: Number of data loading workers (default: 4)
            scheduler_type: Type of learning rate scheduler (default: 'step')
            scheduler_step_size: Step size for learning rate scheduler (default: 20)
            scheduler_gamma: Gamma for learning rate scheduler (default: 0.5)
        """
        # Model configuration
        self.model_config = model_config if model_config is not None else TAGANConfig()
        
        # Experiment metadata
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.output_path = output_path
        
        # Training settings
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.num_workers = num_workers
        
        # Learning rate scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate the experiment configuration parameters."""
        # Validate model configuration
        self.model_config.validate()
        
        # Check experiment metadata
        if not self.experiment_name:
            raise ValueError("Experiment name cannot be empty")
        
        # Check training settings
        if self.num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.patience <= 0:
            raise ValueError(f"Patience must be positive, got {self.patience}")
        if self.num_folds <= 0:
            raise ValueError(f"Number of folds must be positive, got {self.num_folds}")
        if self.eval_interval <= 0:
            raise ValueError(f"Evaluation interval must be positive, got {self.eval_interval}")
        if self.save_interval <= 0:
            raise ValueError(f"Save interval must be positive, got {self.save_interval}")
        if self.num_workers < 0:
            raise ValueError(f"Number of workers must be non-negative, got {self.num_workers}")
        
        # Check scheduler settings
        valid_scheduler_types = ['step', 'exponential', 'cosine', 'plateau', 'none']
        if self.scheduler_type not in valid_scheduler_types:
            raise ValueError(f"Scheduler type must be one of {valid_scheduler_types}, got {self.scheduler_type}")
        if self.scheduler_step_size <= 0:
            raise ValueError(f"Scheduler step size must be positive, got {self.scheduler_step_size}")
        if self.scheduler_gamma <= 0 or self.scheduler_gamma > 1:
            raise ValueError(f"Scheduler gamma must be in (0, 1], got {self.scheduler_gamma}")
    
    def update(self, **kwargs):
        """
        Update experiment configuration parameters.
        
        Args:
            **kwargs: Keyword arguments to update
        """
        for key, value in kwargs.items():
            if key == 'model_config':
                if isinstance(value, dict):
                    self.model_config.update(**value)
                elif isinstance(value, TAGANConfig):
                    self.model_config = value
                else:
                    raise ValueError("model_config must be a dictionary or TAGANConfig instance")
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid experiment configuration parameter: {key}")
        
        # Validate updated configuration
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert experiment configuration to dictionary.
        
        Returns:
            Dictionary of experiment configuration parameters
        """
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if key == 'model_config':
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        
        return config_dict
    
    def save(self, filepath: str):
        """
        Save experiment configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Convert to dictionary and save as JSON
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TAGANExperimentConfig':
        """
        Create an experiment configuration from a dictionary.
        
        Args:
            config_dict: Dictionary of experiment configuration parameters
            
        Returns:
            TAGANExperimentConfig instance
        """
        # Extract model configuration
        model_config_dict = config_dict.pop('model_config', {})
        model_config = TAGANConfig.from_dict(model_config_dict)
        
        # Create new instance with model configuration
        config = cls(model_config=model_config)
        
        # Update with remaining parameters
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Invalid experiment configuration parameter: {key}")
        
        # Validate configuration
        config.validate()
        
        return config
    
    @classmethod
    def load(cls, filepath: str) -> 'TAGANExperimentConfig':
        """
        Load experiment configuration from a JSON file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            TAGANExperimentConfig instance
        """
        # Load JSON file
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create configuration from dictionary
        return cls.from_dict(config_dict)
    
    def get_output_dir(self) -> str:
        """
        Get the output directory for this experiment.
        
        Returns:
            Path to the output directory
        """
        return os.path.join(self.output_path, self.experiment_name)
    
    def __repr__(self) -> str:
        """Return a string representation of the experiment configuration."""
        config_str = "TAGANExperimentConfig(\n"
        
        for key, value in sorted(self.__dict__.items()):
            if key == 'model_config':
                config_str += f"  {key}=\n"
                for line in str(value).split('\n'):
                    config_str += f"    {line}\n"
            else:
                config_str += f"  {key}={value},\n"
        
        config_str += ")"
        
        return config_str
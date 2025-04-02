"""
Training module for TAGAN models.

This module provides a trainer class for training, evaluating, and testing TAGAN models,
including support for early stopping, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from tqdm import tqdm

from ..model import TAGAN
from ..utils.config import TAGANConfig
from ..utils.metrics import calculate_metrics, MetricsTracker
from ..data.data_loader import TemporalGraphDataLoader


class TAGANTrainer:
    """
    Trainer class for TAGAN models.
    
    This class handles training, evaluation, and testing of TAGAN models,
    including early stopping, checkpointing, and logging.
    
    Attributes:
        model (TAGAN): TAGAN model
        config (TAGANConfig): Configuration parameters
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        metrics_tracker (MetricsTracker): Tracker for performance metrics
        early_stopping_patience (int): Patience for early stopping
        early_stopping_counter (int): Counter for early stopping
        best_val_score (float): Best validation score
    """
    
    def __init__(
        self,
        model: TAGAN,
        config: TAGANConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        early_stopping_patience: int = 10
    ):
        """
        Initialize the TAGAN trainer.
        
        Args:
            model: TAGAN model
            config: Configuration parameters
            optimizer: Optimizer (default: None, uses Adam with config learning rate)
            lr_scheduler: Learning rate scheduler (default: None)
            device: Device to train on (default: None, uses config device)
            checkpoint_dir: Directory for saving checkpoints (default: './checkpoints')
            log_dir: Directory for saving logs (default: './logs')
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Ensure directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set device
        self.device = device if device is not None else torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Set learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        # Set up metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.best_val_score = float('-inf')
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Flag to track if training was completed
        self.training_completed = False
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for training.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger('tagan_trainer')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.log_dir, f'training_{time.strftime("%Y%m%d-%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train(
        self,
        train_loader: TemporalGraphDataLoader,
        val_loader: Optional[TemporalGraphDataLoader] = None,
        num_epochs: Optional[int] = None,
        validate_every: int = 1,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        Train the TAGAN model.
        
        Args:
            train_loader: Data loader for training data
            val_loader: Data loader for validation data (default: None)
            num_epochs: Number of epochs to train for (default: None, uses config)
            validate_every: Validate every n epochs (default: 1)
            save_best: Whether to save the best model (default: True)
            
        Returns:
            Dictionary of training results
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        
        # Reset early stopping
        self.early_stopping_counter = 0
        self.best_val_score = float('-inf')
        
        # Reset metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Reset training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            train_results = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_results['loss'])
            self.train_metrics.append(train_results['metrics'])
            
            # Validation
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_results = self.evaluate(val_loader, epoch=epoch)
                self.val_losses.append(val_results['loss'])
                self.val_metrics.append(val_results['metrics'])
                
                current_val_score = val_results['metrics']['f1']
                
                # Update metrics tracker
                self.metrics_tracker.update_epoch_metrics(val_results['metrics'], epoch)
                
                # Early stopping
                if current_val_score > self.best_val_score:
                    self.logger.info(f"Validation score improved from {self.best_val_score:.4f} to {current_val_score:.4f}")
                    self.best_val_score = current_val_score
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    if save_best:
                        self._save_checkpoint(
                            epoch, 
                            val_results['loss'], 
                            val_results['metrics'],
                            is_best=True
                        )
                else:
                    self.early_stopping_counter += 1
                    self.logger.info(f"Validation score did not improve, counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Update learning rate scheduler if provided
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_loader is not None:
                    self.lr_scheduler.step(val_results['loss'])
                else:
                    self.lr_scheduler.step()
            
            # Log progress
            self._log_epoch_progress(epoch, train_results, val_results if val_loader else None)
            
            # Save periodic checkpoint
            if save_best and (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    epoch, 
                    train_results['loss'], 
                    train_results['metrics'],
                    is_best=False, 
                    filename=f'checkpoint_epoch_{epoch}.pt'
                )
        
        # Training completed
        self.training_completed = True
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        if save_best:
            self._save_checkpoint(
                epoch, 
                train_results['loss'], 
                train_results['metrics'],
                is_best=False, 
                filename='final_model.pt'
            )
        
        # Plot training curves
        self._plot_training_curves()
        
        # Return training results
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_score': self.best_val_score,
            'total_time': total_time,
            'epochs_completed': epoch + 1
        }
    
    def _train_epoch(
        self,
        train_loader: TemporalGraphDataLoader,
        epoch: int
    ) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Data loader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training results for this epoch
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")
        
        for batch_idx, (sequences, labels) in enumerate(pbar):
            # Move labels to device
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences, labels)
            loss = outputs['loss']
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            
            self.optimizer.step()
            
            # Collect predictions and labels for metrics
            preds = outputs['predictions']
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            # Update loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute overall loss and metrics
        avg_loss = total_loss / len(train_loader)
        
        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Debug shapes before calculating metrics
        self.logger.info(f"Train - Labels shape: {all_labels.shape}, Predictions shape: {all_preds.shape}")
        
        # Make sure predictions and labels have the same batch dimension
        if all_preds.shape[0] != all_labels.shape[0]:
            # If predictions are per timestep but labels are per sequence
            if len(all_preds.shape) > 1 and all_preds.shape[0] > all_labels.shape[0]:
                # Take the prediction from the last timestep or average across timesteps
                all_preds = all_preds[-all_labels.shape[0]:]
            elif all_labels.shape[0] > all_preds.shape[0]:
                # If we have more labels than predictions, trim labels
                all_labels = all_labels[:all_preds.shape[0]]
                
        # Compute metrics
        metrics = calculate_metrics(all_labels, all_preds)
        
        return {'loss': avg_loss, 'metrics': metrics}
    
    def evaluate(
        self,
        data_loader: TemporalGraphDataLoader,
        epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on validation or test data.
        
        Args:
            data_loader: Data loader for evaluation data
            epoch: Current epoch number (default: None)
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Epoch {epoch+1 if epoch is not None else 'N/A'} (Eval)")
            
            for batch_idx, (sequences, labels) in enumerate(pbar):
                # Move labels to device
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, labels)
                loss = outputs['loss']
                
                # Collect predictions and labels for metrics
                preds = outputs['predictions']
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                # Update loss
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        # Compute overall loss and metrics
        if len(data_loader) > 0:
            avg_loss = total_loss / len(data_loader)
        else:
            self.logger.info("Evaluation dataset is empty, skipping evaluation")
            return {
                'loss': 0.0,
                'metrics': {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            }
        
        # Concatenate predictions and labels if we have any
        if not all_preds or not all_labels:
            self.logger.warning("No predictions or labels collected during evaluation")
            return {
                'loss': avg_loss,
                'metrics': {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            }
            
        # Handle case where we have only one batch
        if len(all_preds) == 1:
            all_preds = all_preds[0]
            all_labels = all_labels[0]
        else:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        
        # Debug shapes before calculating metrics
        self.logger.info(f"Eval - Labels shape: {all_labels.shape}, Predictions shape: {all_preds.shape}")
        
        # Make sure predictions and labels have the same batch dimension
        if all_preds.shape[0] != all_labels.shape[0]:
            # If predictions are per timestep but labels are per sequence
            if len(all_preds.shape) > 1 and all_preds.shape[0] > all_labels.shape[0]:
                # Take the prediction from the last timestep or average across timesteps
                all_preds = all_preds[-all_labels.shape[0]:]
            elif all_labels.shape[0] > all_preds.shape[0]:
                # If we have more labels than predictions, trim labels
                all_labels = all_labels[:all_preds.shape[0]]
        
        # Compute metrics
        metrics = calculate_metrics(all_labels, all_preds)
        
        return {'loss': avg_loss, 'metrics': metrics}
    
    def test(
        self,
        test_loader: TemporalGraphDataLoader,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test the model on test data.
        
        Args:
            test_loader: Data loader for test data
            model_path: Path to model checkpoint (default: None, uses current model)
            
        Returns:
            Dictionary of test results
        """
        # Load best model if path provided
        if model_path is not None:
            self.load_checkpoint(model_path)
        
        self.logger.info("Starting model testing")
        
        # Evaluate on test data
        test_results = self.evaluate(test_loader)
        
        self.logger.info(f"Test loss: {test_results['loss']:.4f}")
        self.logger.info(f"Test metrics: {test_results['metrics']}")
        
        return test_results
    
    def predict(
        self,
        data_loader: TemporalGraphDataLoader,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on data.
        
        Args:
            data_loader: Data loader for prediction data
            return_probs: Whether to return probabilities (default: True)
            
        Returns:
            Tuple of (predictions, labels)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in tqdm(data_loader, desc="Making predictions"):
                # Forward pass
                outputs = self.model.infer(sequences, return_probs=return_probs)
                
                # Collect predictions and labels
                preds = outputs['predictions']
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels)
        
        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_preds, all_labels
    
    def _log_epoch_progress(
        self,
        epoch: int,
        train_results: Dict[str, Any],
        val_results: Optional[Dict[str, Any]] = None
    ):
        """
        Log progress for an epoch.
        
        Args:
            epoch: Current epoch number
            train_results: Results from training
            val_results: Results from validation (default: None)
        """
        log_str = f"Epoch {epoch+1}/{self.config.num_epochs} - "
        log_str += f"Train Loss: {train_results['loss']:.4f}, "
        log_str += f"Train Accuracy: {train_results['metrics']['accuracy']:.4f}, "
        log_str += f"Train F1: {train_results['metrics']['f1']:.4f}"
        
        if val_results is not None:
            log_str += f" | Val Loss: {val_results['loss']:.4f}, "
            log_str += f"Val Accuracy: {val_results['metrics']['accuracy']:.4f}, "
            log_str += f"Val F1: {val_results['metrics']['f1']:.4f}"
        
        self.logger.info(log_str)
    
    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            metrics: Current metrics
            is_best: Whether this is the best model so far (default: False)
            filename: Custom filename for the checkpoint (default: None)
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Epoch number of the checkpoint
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load lr scheduler state
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Return epoch number
        return checkpoint['epoch']
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        if not self.train_losses:
            return
        
        # Create figure directory
        fig_dir = os.path.join(self.log_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, 'loss_curve.png'))
        
        # Plot accuracy curve
        plt.figure(figsize=(10, 5))
        train_acc = [m['accuracy'] for m in self.train_metrics]
        plt.plot(train_acc, label='Train Accuracy')
        if self.val_metrics:
            val_acc = [m['accuracy'] for m in self.val_metrics]
            plt.plot(val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, 'accuracy_curve.png'))
        
        # Plot F1 score curve
        plt.figure(figsize=(10, 5))
        train_f1 = [m['f1'] for m in self.train_metrics]
        plt.plot(train_f1, label='Train F1 Score')
        if self.val_metrics:
            val_f1 = [m['f1'] for m in self.val_metrics]
            plt.plot(val_f1, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, 'f1_score_curve.png'))
    
    def get_learning_rate(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
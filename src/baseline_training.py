"""Training utilities for baseline models.

This module provides training loops and utilities for baseline models,
including optimizers, learning rate schedules, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Callable
import numpy as np
from tqdm import tqdm

from .baselines import (
    BaselineVAE, 
    BaselineGAN, 
    CompositionalVAE,
    BaselineLSTM,
    BaselineTransformer
)


class VAETrainer:
    """Training loop for VAE models.
    
    Args:
        model: VAE model to train
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization weight
        beta: Weight for KL divergence term
        device: Device to train on
    """
    
    def __init__(
        self,
        model: BaselineVAE,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        beta: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            if len(batch) == 2:
                x, metadata = batch
                x = x.to(self.device)
                metadata = metadata.to(self.device)
            else:
                x = batch[0].to(self.device)
                metadata = None
            
            # Forward pass
            recon, mu, logvar = self.model(x, metadata)
            
            # Compute loss
            losses = self.model.loss_function(recon, x, mu, logvar, self.beta)
            loss = losses['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': losses['recon_loss'].item(),
                'kl': losses['kl_loss'].item()
            })
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        checkpoint_callback: Optional[Callable] = None
    ) -> Dict[str, list]:
        """Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            val_loader: Optional validation data loader
            checkpoint_callback: Optional callback for saving checkpoints
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': []
        }
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_recon'].append(train_metrics['recon_loss'])
            history['train_kl'].append(train_metrics['kl_loss'])
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}, val_loss={val_loss:.4f}')
            else:
                print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}')
            
            # Checkpoint
            if checkpoint_callback is not None:
                checkpoint_callback(self.model, epoch, train_metrics)
        
        return history
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, metadata = batch
                    x = x.to(self.device)
                    metadata = metadata.to(self.device)
                else:
                    x = batch[0].to(self.device)
                    metadata = None
                
                recon, mu, logvar = self.model(x, metadata)
                losses = self.model.loss_function(recon, x, mu, logvar, self.beta)
                total_loss += losses['loss'].item()
                num_batches += 1
        
        return total_loss / num_batches


class GANTrainer:
    """Training loop for GAN models with WGAN-GP.
    
    Args:
        model: GAN model to train
        learning_rate: Optimizer learning rate
        lambda_gp: Gradient penalty weight
        n_critic: Number of critic updates per generator update
        device: Device to train on
    """
    
    def __init__(
        self,
        model: BaselineGAN,
        learning_rate: float = 1e-4,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        self.optimizer_g = optim.Adam(
            model.generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9)
        )
        self.optimizer_d = optim.Adam(
            model.discriminator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9)
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, batch in enumerate(pbar):
            if len(batch) == 2:
                real_data, metadata = batch
                real_data = real_data.to(self.device)
                metadata = metadata.to(self.device)
            else:
                real_data = batch[0].to(self.device)
                metadata = None
            
            batch_size = real_data.size(0)
            
            # Train discriminator
            for _ in range(self.n_critic):
                self.optimizer_d.zero_grad()
                
                # Generate fake data
                z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
                if metadata is not None:
                    z_input = torch.cat([z, metadata], dim=1)
                else:
                    z_input = z
                
                fake_data = self.model.generator(z_input)
                fake_data = torch.softmax(fake_data, dim=1)
                
                # Discriminator scores
                real_score = self.model.discriminate(real_data, metadata)
                fake_score = self.model.discriminate(fake_data.detach(), metadata)
                
                # Gradient penalty
                gp = self.model.gradient_penalty(
                    real_data, 
                    fake_data.detach(), 
                    metadata, 
                    self.lambda_gp
                )
                
                # Wasserstein loss
                d_loss = fake_score.mean() - real_score.mean() + gp
                d_loss.backward()
                self.optimizer_d.step()
            
            # Train generator
            self.optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            if metadata is not None:
                z_input = torch.cat([z, metadata], dim=1)
            else:
                z_input = z
            
            fake_data = self.model.generator(z_input)
            fake_data = torch.softmax(fake_data, dim=1)
            fake_score = self.model.discriminate(fake_data, metadata)
            
            g_loss = -fake_score.mean()
            g_loss.backward()
            self.optimizer_g.step()
            
            # Accumulate metrics
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item()
            })
        
        return {
            'd_loss': total_d_loss / num_batches,
            'g_loss': total_g_loss / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        checkpoint_callback: Optional[Callable] = None
    ) -> Dict[str, list]:
        """Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            checkpoint_callback: Optional callback for saving checkpoints
            
        Returns:
            Dictionary with training history
        """
        history = {
            'd_loss': [],
            'g_loss': []
        }
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader, epoch)
            history['d_loss'].append(metrics['d_loss'])
            history['g_loss'].append(metrics['g_loss'])
            
            print(f'Epoch {epoch}: d_loss={metrics["d_loss"]:.4f}, g_loss={metrics["g_loss"]:.4f}')
            
            if checkpoint_callback is not None:
                checkpoint_callback(self.model, epoch, metrics)
        
        return history


class TemporalTrainer:
    """Training loop for temporal prediction models (LSTM/Transformer).
    
    Args:
        model: Temporal model to train
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization weight
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader (sequences)
            epoch: Current epoch number
            
        Returns:
            Dictionary with average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Expect (history, target) pairs
            history, target = batch
            history = history.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            if isinstance(self.model, BaselineLSTM):
                pred, _ = self.model(history)
            else:  # Transformer
                pred = self.model(history)
            
            # Compute loss (KL divergence for compositional data)
            loss = torch.nn.functional.kl_div(
                torch.log(pred + 1e-10),
                target,
                reduction='batchmean'
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return {'loss': total_loss / num_batches}
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        checkpoint_callback: Optional[Callable] = None
    ) -> Dict[str, list]:
        """Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            val_loader: Optional validation data loader
            checkpoint_callback: Optional callback for saving checkpoints
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}, val_loss={val_loss:.4f}')
            else:
                print(f'Epoch {epoch}: train_loss={train_metrics["loss"]:.4f}')
            
            if checkpoint_callback is not None:
                checkpoint_callback(self.model, epoch, train_metrics)
        
        return history
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history, target = batch
                history = history.to(self.device)
                target = target.to(self.device)
                
                if isinstance(self.model, BaselineLSTM):
                    pred, _ = self.model(history)
                else:
                    pred = self.model(history)
                
                loss = torch.nn.functional.kl_div(
                    torch.log(pred + 1e-10),
                    target,
                    reduction='batchmean'
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

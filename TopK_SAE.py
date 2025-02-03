"""
Implementation of TopK Sparse Autoencoder.
This autoencoder variant enforces sparsity by keeping only the k highest magnitude 
activations in the hidden layer.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import wandb
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
import numpy as np

class ActivationsDataset(Dataset):
    """Dataset for loading stored model activations."""
    def __init__(self, activations_path: str):
        self.file = h5py.File(activations_path, 'r')
        self.activations = self.file['activations']
        self.paths = self.file['paths']
        
    def __len__(self) -> int:
        return len(self.activations)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.activations[idx]).float()
    
    def __del__(self):
        self.file.close()

class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder implementation.
    
    Args:
        input_dim: Dimension of input features
        expansion_factor: Factor to multiply input dimension by for hidden layer size
        k: Number of top activations to keep (others set to zero)
        learning_rate: Initial learning rate
        lr_scheduler: Learning rate scheduler type ('cosine', 'linear', or None)
        pretrained_path: Optional path to load pretrained weights from
    """
    def __init__(
        self,
        input_dim: int,
        expansion_factor: float,
        k: int,
        learning_rate: float,
        lr_scheduler: Optional[str] = None,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * expansion_factor)
        self.k = k
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim)
        
        # Initialize with scaled random weights
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))
            
    def top_k_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsification to activations."""
        values, indices = torch.topk(x.abs(), self.k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1)
        return x * mask
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both hidden activations and reconstruction."""
        encoded = self.encoder(x)
        sparse_encoded = self.top_k_activation(encoded)
        decoded = self.decoder(sparse_encoded)
        return sparse_encoded, decoded
        
    def get_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        if self.lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_training_steps
            )
        elif self.lr_scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps
            )
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def dataloader(
        self,
        dataset_path: str,
        batch_size: int,
        num_workers: int,
        distributed: bool = False
    ) -> DataLoader:
        """Create dataloader for training/inference."""
        dataset = ActivationsDataset(dataset_path)
        
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        
    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        """Single training step."""
        optimizer.zero_grad()
        _, reconstruction = self(batch)
        loss = F.mse_loss(reconstruction, batch)
        loss.backward()
        optimizer.step()
        return loss
        
    def train_model(
        self,
        epochs: int,
        batch_size: int,
        model_activations_path: str,
        checkpoint_path: str,
        num_workers: int = 4,
        device: str = 'cuda',
        distributed: bool = False,
        local_rank: int = -1
    ):
        """Train the autoencoder."""
        # Initialize wandb
        if not distributed or (distributed and local_rank == 0):
            wandb.init(
                project="sparse_autoencoder",
                config={
                    "architecture": "TopK",
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "k": self.k,
                    "learning_rate": self.learning_rate,
                    "scheduler": self.lr_scheduler_type,
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            )
        
        # Setup distributed training if needed
        if distributed:
            if device == 'cuda':
                torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            self = DDP(self, device_ids=[local_rank])
        elif torch.cuda.device_count() > 1 and device == 'cuda':
            self = DataParallel(self)
            
        self.to(device)
        
        # Create dataloader
        dataloader = self.dataloader(
            model_activations_path,
            batch_size,
            num_workers,
            distributed
        )
        
        # Setup optimizer and scheduler
        total_steps = len(dataloader) * epochs
        optimizer, scheduler = self.get_optimizer_and_scheduler(total_steps)
        
        # Training loop
        for epoch in range(epochs):
            if distributed:
                dataloader.sampler.set_epoch(epoch)
                
            epoch_losses = []
            for batch in dataloader:
                batch = batch.to(device)
                loss = self.training_step(batch, optimizer)
                epoch_losses.append(loss.item())
                
                if scheduler:
                    scheduler.step()
                    
            # Log metrics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            if not distributed or (distributed and local_rank == 0):
                wandb.log({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pt')
                )
                
        if distributed:
            dist.destroy_process_group()
            
        if not distributed or (distributed and local_rank == 0):
            wandb.finish()
            
    def get_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Get latent representations for input images."""
        self.eval()
        with torch.no_grad():
            latents, _ = self(images)
        return latents
        
    def get_top_images(
        self,
        neuron_number: int,
        dataset_path: str,
        output_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = 'cuda',
        top_k: int = 9
    ):
        """Find and save images that most activate a specific neuron."""
        self.eval()
        self.to(device)
        
        # Setup dataloader
        dataloader = self.dataloader(dataset_path, batch_size, num_workers)
        
        # Track top activations and corresponding images
        top_activations = []
        top_image_paths = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                latents = self.get_latents(batch)
                activations = latents[:, neuron_number].cpu().numpy()
                
                # Get image paths for this batch
                paths = [str(p) for p in dataloader.dataset.paths[len(top_activations):len(top_activations) + len(batch)]]
                
                # Update top activations
                for act, path in zip(activations, paths):
                    if len(top_activations) < top_k:
                        top_activations.append(act)
                        top_image_paths.append(path)
                    else:
                        min_idx = np.argmin(top_activations)
                        if act > top_activations[min_idx]:
                            top_activations[min_idx] = act
                            top_image_paths[min_idx] = path
                            
        # Sort by activation strength
        sorted_indices = np.argsort(top_activations)[::-1]
        top_activations = [top_activations[i] for i in sorted_indices]
        top_image_paths = [top_image_paths[i] for i in sorted_indices]
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for idx, (act, img_path) in enumerate(zip(top_activations, top_image_paths)):
            if idx >= 9:  # Only show top 9
                break
            row, col = idx // 3, idx % 3
            img = plt.imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Activation: {act:.3f}')
            
        plt.tight_layout()
        plt.savefig(output_dir / f'neuron_{neuron_number}_top_activations.pdf')
        plt.close()
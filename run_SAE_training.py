"""
Script for training Sparse Autoencoders (SAE) on transformer activations.
Supports different SAE types (TopK, Gated, Jump ReLU) and training configurations.
Uses multiple GPUs on a local machine.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
import wandb
import h5py

# Import SAE implementations
from TopK_SAE import TopKSAE
# TODO: Import other SAE types once implemented
# from gated_sae import GatedSAE
# from jump_relu_sae import JumpReLUSAE

def train_sae(args):
    """Training function with local multi-GPU support."""
    # Detect available GPUs
    n_gpus = torch.cuda.device_count()
    is_multi_gpu = n_gpus > 1

    # Create SAE model based on type
    if args.sae_type.lower() == 'topk':
        sae = TopKSAE(
            input_dim=args.input_dim,
            expansion_factor=args.expansion_factor,
            k=args.k,
            learning_rate=args.learning_rate,
            lr_scheduler=args.lr_scheduler,
            pretrained_path=args.pretrained_path if hasattr(args, 'pretrained_path') else None
        )
    elif args.sae_type.lower() == 'gated':
        raise NotImplementedError("Gated SAE not implemented yet")
    elif args.sae_type.lower() == 'jumprelu':
        raise NotImplementedError("Jump ReLU SAE not implemented yet")
    else:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")

    # Move model to GPU and wrap with DataParallel if multiple GPUs available
    sae = sae.to('cuda')
    if is_multi_gpu:
        sae = DataParallel(sae)
        print(f"Using {n_gpus} GPUs")
    else:
        print("Using single GPU")

    # Construct paths
    activations_path = Path(args.activations_root) / \
                      args.dataset / \
                      args.split / \
                      args.model_library / \
                      args.model_version / \
                      str(args.block_index) / \
                      args.block_element / \
                      f"activations_{args.model_version}_{args.block_index}_{args.block_element}.h5"

    checkpoint_path = Path(args.checkpoint_root) / \
                     args.dataset / \
                     args.split / \
                     args.model_library / \
                     args.model_version / \
                     str(args.block_index) / \
                     args.block_element / \
                     args.sae_type / \
                     f"exp_{args.expansion_factor}_lr_{args.learning_rate}"
    
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="sparse_autoencoder",
        name=f"{args.dataset}_{args.split}_{args.model_library}_{args.model_version}_{args.block_index}_{args.block_element}",
        config={
            "architecture": args.sae_type,
            "input_dim": args.input_dim,
            "hidden_dim": int(args.input_dim * args.expansion_factor),
            "k": args.k,
            "learning_rate": args.learning_rate,
            "scheduler": args.lr_scheduler,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataset": args.dataset,
            "split": args.split,
            "model_library": args.model_library,
            "model_version": args.model_version,
            "block_index": args.block_index,
            "block_element": args.block_element,
            "num_gpus": n_gpus
        }
    )

    # Before creating the dataloader:
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations file not found at: {activations_path}")
    
    print(f"Loading activations from: {activations_path}")
    try:
        with h5py.File(str(activations_path), 'r') as f:
            print(f"HDF5 file contains keys: {list(f.keys())}")
            if 'activations' in f:
                print(f"Activations shape: {f['activations'].shape}")
    except Exception as e:
        print(f"Failed to inspect HDF5 file: {str(e)}")
        raise

    # Create dataloader
    dataloader = sae.module.dataloader(
        str(activations_path),
        args.batch_size,
        args.num_workers,
        distributed=False  # Using DataParallel instead of DDP
    ) if is_multi_gpu else sae.dataloader(
        str(activations_path),
        args.batch_size,
        args.num_workers,
        distributed=False
    )

    # Setup optimizer and scheduler
    total_steps = len(dataloader) * args.epochs
    optimizer, scheduler = sae.module.get_optimizer_and_scheduler(total_steps) if is_multi_gpu \
        else sae.get_optimizer_and_scheduler(total_steps)

    # Training loop
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to('cuda')
            
            # Training step
            loss = sae.module.training_step(batch, optimizer) if is_multi_gpu \
                else sae.training_step(batch, optimizer)
            epoch_losses.append(loss.item())

            if scheduler:
                scheduler.step()

            # Log batch info
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch": epoch * len(dataloader) + batch_idx,
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Log epoch info
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss
        })

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': sae.module.state_dict() if is_multi_gpu else sae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': {
                'input_dim': args.input_dim,
                'expansion_factor': args.expansion_factor,
                'k': args.k,
                'learning_rate': args.learning_rate,
                'lr_scheduler': args.lr_scheduler
            }
        }
        torch.save(
            checkpoint,
            checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
        )

    wandb.finish()

    print("Training complete")

def main():
    parser = argparse.ArgumentParser(description='Train Sparse Autoencoder')
    
    # SAE configuration
    parser.add_argument('--sae-type', type=str, required=True,
                      choices=['topk', 'gated', 'jumprelu'],
                      help='Type of SAE to train')
    parser.add_argument('--input-dim', type=int, required=True,
                      help='Input dimension of the activations')
    parser.add_argument('--expansion-factor', type=float, required=True,
                      help='Hidden layer expansion factor')
    parser.add_argument('--k', type=int,
                      help='K parameter for TopK SAE')
    parser.add_argument('--learning-rate', type=float, required=True,
                      help='Learning rate')
    parser.add_argument('--lr-scheduler', type=str,
                      choices=['cosine', 'linear', None],
                      help='Learning rate scheduler type')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, required=True,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    
    # Data configuration
    parser.add_argument('--activations-root', type=str, required=True,
                      help='Root directory containing model activations')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., imagenet)')
    parser.add_argument('--split', type=str, required=True,
                      choices=['train', 'val', 'test'],
                      help='Dataset split')
    parser.add_argument('--model-library', type=str, required=True,
                      choices=['clip', 'open_clip'],
                      help='Model library used for activations')
    parser.add_argument('--model-version', type=str, required=True,
                      help='Model version used for activations')
    parser.add_argument('--block-index', type=int, required=True,
                      help='Transformer block index')
    parser.add_argument('--block-element', type=str, required=True,
                      choices=['attention', 'mlp', 'residual', 'output'],
                      help='Block element type')
    
    # Output configuration
    parser.add_argument('--checkpoint-root', type=str, required=True,
                      help='Root directory for saving checkpoints')
    parser.add_argument('--pretrained-path', type=str,
                      help='Path to pretrained SAE weights')

    args = parser.parse_args()
    train_sae(args)

if __name__ == '__main__':
    '''
    Example usage:
    python run.py \
    --sae-type topk \
    --input-dim 768 \
    --expansion-factor 4.0 \
    --k 128 \
    --learning-rate 1e-3 \
    --lr-scheduler cosine \
    --epochs 100 \
    --batch-size 256 \
    --activations-root /path/to/activations \
    --dataset imagenet \
    --split train \
    --model-library clip \
    --model-version ViT-B/32 \
    --block-index 5 \
    --block-element mlp \
    --checkpoint-root /path/to/checkpoints
    '''
    main()
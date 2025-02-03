"""
Script to extract and save activations from ViT models (CLIP or OpenCLIP variants).
Saves activations from specified transformer blocks and elements to a designated output path.
"""

import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import h5py
from tqdm import tqdm
from typing import Optional, Tuple
from ViT_wrapper import ViTWrapper, ModelConfig, CLIPLibrary, BlockType

class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images."""
    def __init__(self, data_path: str, transform: Optional[transforms.Compose] = None):
        self.data_path = Path(data_path)
        self.image_files = [f for f in self.data_path.rglob("*") 
                           if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                  (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
            
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(image_path)

def save_activations(args):
    """Main function to extract and save model activations."""
    
    # Create output directory if it doesn't exist
    output_path = Path(os.path.join(
            args.output_path,  
            args.dataset_path.split("/")[-1],
            args.model_library,
            args.model_version,
            str(args.block_index),
            args.block_element
        ))
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure model
    config = ModelConfig(
        library=CLIPLibrary(args.model_library),
        model_name=args.model_version,
        pretrained=args.pretrained if hasattr(args, 'pretrained') else None,
        device=args.device
    )
    
    # Initialize model wrapper
    model = ViTWrapper(config)
    model.eval()
    
    # Register hook for activation extraction
    block_type = BlockType(args.block_element)
    model.register_activation_hook(
        block_idx=args.block_index,
        block_type=block_type
    )
    
    # Setup dataset and dataloader
    dataset = ImageDataset(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create HDF5 file for saving activations
    activation_file = output_path / f"activations_{args.model_version}_{args.block_index}_{args.block_element}.h5"
    
    with h5py.File(activation_file, 'w') as f:
        # Create datasets with maxshape for resizing
        activations_dataset = None
        paths_dataset = None
        
        with torch.no_grad():
            for batch_idx, (images, paths) in enumerate(tqdm(dataloader)):
                # Move images to device
                images = images.to(args.device)
                
                # Forward pass
                _ = model(images)
                
                # Get activations
                batch_activations = model.get_activations()
                key = f"block_{args.block_index}_{args.block_element}"
                activations = batch_activations[key].cpu().numpy()
                
                # Initialize datasets on first batch
                if activations_dataset is None:
                    activations_dataset = f.create_dataset(
                        'activations',
                        shape=(len(dataset), *activations.shape[1:]),
                        dtype='float32',
                        chunks=True
                    )
                    paths_dataset = f.create_dataset(
                        'paths',
                        shape=(len(dataset),),
                        dtype=h5py.special_dtype(vlen=str)
                    )
                
                # Calculate indices for this batch
                start_idx = batch_idx * args.batch_size
                end_idx = start_idx + activations.shape[0]
                
                # Save activations and paths
                activations_dataset[start_idx:end_idx] = activations
                paths_dataset[start_idx:end_idx] = paths
                
                # Clear stored activations
                model.clear_activations()
    
    # Clean up
    model.remove_hooks()

def main():
    parser = argparse.ArgumentParser(description='Save model activations from ViT models')
    
    # Model configuration
    parser.add_argument('--model-library', type=str, choices=['clip', 'open_clip'],
                      help='Which CLIP library to use')
    parser.add_argument('--model-version', type=str,
                      help='Model version/name (e.g., "ViT-B/32" for CLIP)')
    parser.add_argument('--pretrained', type=str,
                      help='Pretrained model name for OpenCLIP')
    
    # Activation extraction configuration
    parser.add_argument('--block-index', type=int,
                      help='Index of transformer block to extract from')
    parser.add_argument('--block-element', type=str,
                      choices=['attention', 'mlp', 'residual', 'output'],
                      help='Element of the block to extract')
    
    # Data and output configuration
    parser.add_argument('--dataset-path', type=str,
                      help='Path to dataset directory')
    parser.add_argument('--output-path', type=str,
                      help='Path to save activations')
    
    # Runtime configuration
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    save_activations(args)

if __name__ == '__main__':
    '''
    Run example: 
    python save_activations.py \
    --model-library clip \
    --model-version ViT-B/32 \
    --block-index 5 \
    --block-element mlp \
    --dataset-path /path/to/dataset \
    --output-path /path/to/output \
    --batch-size 64 \
    --num-workers 8 
    '''

    main()
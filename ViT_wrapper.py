"""
This wrapper is needed because:
1. We're working with two different CLIP implementations (OpenAI CLIP and OpenCLIP) that have
   different internal architectures and naming conventions
2. We need to extract activations from specific parts of the transformer blocks (MLP, attention, etc.)
   which requires careful hook placement based on the model's architecture
3. We want to select tokens with highest L2 norm, which needs to be done consistently across
   both implementations
4. We need a clean interface for registering hooks and collecting activations that works the same way
   regardless of which CLIP implementation we're using

This wrapper provides a unified interface for all these operations, hiding the complexity of dealing
with different model architectures behind a consistent API.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import open_clip  # for OpenCLIP
from transformers import CLIPModel  # for OpenAI CLIP
from dataclasses import dataclass
from enum import Enum
from torch import Tensor

class CLIPLibrary(Enum):
    OPENAI = "clip"
    OPEN_CLIP = "open_clip"

@dataclass
class ModelConfig:
    library: CLIPLibrary
    model_name: str
    pretrained: str
    device: str = "cuda"

class BlockType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"
    OUTPUT = "output"

def batch_highest_norm_token(x: torch.Tensor) -> torch.Tensor:
    """
    Select tokens with highest L2 norm for each sample in batch.
    
    Args:
        x: Input tensor of shape (batch_size, num_tokens, hidden_dim)
    Returns:
        Tensor of shape (batch_size, hidden_dim) containing highest norm tokens
    """
    norms = torch.norm(x, dim=2)  # [batch_size, num_tokens]
    max_norm_indices = torch.argmax(norms, dim=1)  # [batch_size]
    return torch.stack([x[i, idx] for i, idx in enumerate(max_norm_indices)])

class ViTWrapper(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.hooks = {}  # Initialize hooks dictionary
        self.stored_activations = {}
        self._load_model()

    def _load_model(self):
        """Load model based on specified library and configuration."""
        if self.config.library == CLIPLibrary.OPENAI:
            # Read HuggingFace token
            with open('hf_token', 'r') as f:
                token = f.read().strip()
            
            # Convert OpenAI CLIP model name to HuggingFace model name
            # e.g., 'ViT-B/32' -> 'openai/clip-vit-base-patch32'
            if self.config.model_name == 'ViT-B/32':
                hf_model_name = 'openai/clip-vit-base-patch32'
            elif self.config.model_name == 'ViT-L/14':
                hf_model_name = 'openai/clip-vit-large-patch14'
            elif self.config.model_name == 'ViT-L/14@336px':
                hf_model_name = 'openai/clip-vit-large-patch14-336'
            else:
                raise ValueError(f"Unsupported model name: {self.config.model_name}")
            
            self.model = CLIPModel.from_pretrained(hf_model_name, token=token)
            self.model.to(self.config.device)
            self.visual = self.model.vision_model
        else:  # OpenCLIP
            self.model, _, _ = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained,
                device=self.config.device
            )
            self.visual = self.model.visual
            
    def _get_block_mapping(self) -> Dict[str, Any]:
        """Get mapping of internal layer names based on library."""
        if self.config.library == CLIPLibrary.OPENAI:
            return {
                BlockType.ATTENTION: "attn",
                BlockType.MLP: "mlp",
                BlockType.RESIDUAL: "ln_1",  # Pre-normalization
                BlockType.OUTPUT: "ln_2"  # Final output
            }
        else:  # OpenCLIP
            return {
                BlockType.ATTENTION: "attn",  # Changed from self_attention to attn
                BlockType.MLP: "mlp",
                BlockType.RESIDUAL: "ln_1",  # Changed from ln1 to ln_1
                BlockType.OUTPUT: "ln_2"  # Changed from ln2 to ln_2
            } 

    def _get_block(self, block_idx: int) -> nn.Module:
        """Get transformer block by index."""
        if self.config.library == CLIPLibrary.OPENAI:
            return self.visual.transformer.resblocks[block_idx]
        else:
            return self.visual.transformer.resblocks[block_idx] 

    def _hook_fn(self, name: str):
        """Factory for creating named forward hooks."""
        def hook(module, input, output):
            self.stored_activations[name] = output
        return hook
            
    def register_activation_hook(self, 
                               block_idx: int, 
                               block_type: BlockType,
                               token_selector: Optional[callable] = batch_highest_norm_token):
        """
        Register a hook to store activations from specified block and type.
        
        Args:
            block_idx: Index of transformer block
            block_type: Type of activations to store
            token_selector: Function to select specific tokens (defaults to highest norm)
        """
        block = self._get_block(block_idx)
        block_map = self._get_block_mapping()
        
        # Get the specific submodule based on block type
        if block_type == BlockType.RESIDUAL:
            target_module = getattr(block, block_map[BlockType.RESIDUAL])
        elif block_type == BlockType.MLP:
            target_module = getattr(block, block_map[BlockType.MLP])
        elif block_type == BlockType.ATTENTION:
            if self.config.library == CLIPLibrary.OPENAI:
                target_module = getattr(block, block_map[BlockType.ATTENTION])
            else:
                target_module = getattr(block, block_map[BlockType.ATTENTION]).attn
        else:  # OUTPUT
            target_module = getattr(block, block_map[BlockType.OUTPUT])
            
        # Create hook name and register
        hook_name = f"block_{block_idx}_{block_type.value}"
        hook = target_module.register_forward_hook(self._hook_fn(hook_name))
        self.hooks[hook_name] = {
            "hook": hook,
            "token_selector": token_selector
        }
        
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get stored activations with token selection applied.
        
        Returns:
            Dictionary mapping hook names to selected token activations
        """
        result = {}
        for name, hook_info in self.hooks.items():
            if name in self.stored_activations:
                activations = self.stored_activations[name]
                if hook_info["token_selector"] is not None:
                    activations = hook_info["token_selector"](activations)
                result[name] = activations
        return result
        
    def clear_activations(self):
        """Clear stored activations."""
        self.stored_activations.clear()
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook_info in self.hooks.values():
            hook_info["hook"].remove()
        self.hooks.clear()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through visual encoder."""
        return self.visual(x)
        
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
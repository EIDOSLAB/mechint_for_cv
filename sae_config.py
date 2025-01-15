from dataclasses import dataclass
from typing import Optional, Literal, Union, Dict

@dataclass
class SAEConfig():
    # SAE Parameters
    sae_type: Literal['gated', 'top_k', 'vanilla']
    expansion_factor: int = 128
    d_in: int = 768
    init_norm: float = 1
    scaling_factor: float = 1
    name: str = ""
    
    # Top K specific parameters
    k: Optional[int] = None
    
    def __post_init__(self):
        self.d_sae = self.d_in * self.expansion_factor
        if self.sae_type == 'top_k' and self.k is None:
            raise ValueError("k must be specified for top_k SAE type")
    
    def to_dict(self):
        return {
            'sae_type': self.sae_type,
            'expansion_factor': self.expansion_factor,
            'd_in': self.d_in,
            'd_sae': self.d_sae,
            'init_norm': self.init_norm,
            'scaling_factor': self.scaling_factor,
            'name': self.name,
            'k': self.k,
        }
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class TrainingConfig():
    # Training Parameters
    batch_size: int = 2_048
    training_steps: int = 200_000
    lr: float = 5e-5
    l1: Optional[float] = 5
    max_grad_norm: float = 1
    num_workers: int = 4
    save_sae: bool = True
    save_frequency: int = 10_000
    checkpoints_path: str = '../../checkpoints/from_scratch_experiments'
    lr_warmup_proportion: float = 0.01
    lr_warmdown_proportion: float = 0.2
    l1_warmup_proportion: float = 0.05
    normalise_decoder: bool = False

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = 'sae_rad'
    wandb_log_frequency: int = 10
    histogram_window: int = 1e6
    
    # HuggingFace
    upload_to_huggingface: bool = True
    huggingface_org_name: str = 'your-org-here'
    
    def __post_init__(self):
        self.lr_warmup_steps: int = int(self.training_steps * self.lr_warmup_proportion)
        self.lr_warmdown_steps: int = int(self.training_steps * self.lr_warmdown_proportion)
        self.l1_warmup_steps: int = int(self.training_steps * self.l1_warmup_proportion)
    
    def to_dict(self):
        return {
            'batch_size' : self.batch_size,
            'training_steps' : self.training_steps,
            'lr' : self.lr,
            'l1 coefficient' : self.l1,
            'max_grad_norm' : self.max_grad_norm,
            'num_workers' : self.num_workers,
            'save_sae' : self.save_sae,
            'save_frequency' : self.save_frequency,
            'checkpoints_path': self.checkpoints_path,
            'lr_warmup_proportion' : self.lr_warmup_proportion,
            'lr_warmdown_proportion' : self.lr_warmdown_proportion,
            'l1_warmup_proportion' : self.l1_warmup_proportion,
            'lr_warmup_steps' : self.lr_warmup_steps,
            'lr_warmdown_steps' : self.lr_warmdown_steps,
            'l1_warmup_steps' : self.l1_warmup_steps,
            'log_to_wandb' : self.log_to_wandb,
            'wandb_project' : self.wandb_project,
            'wandb_log_frequency' : self.wandb_log_frequency,
            'histogram_window' : self.histogram_window,
            'upload_to_huggingface' : self.upload_to_huggingface,
            'huggingface_org_name' : self.huggingface_org_name,
        }
import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from typing import Optional
from dataclasses import dataclass

class LRScheduler:
    def __init__(self, training_conifg):
        self.steps = 1
        
        def scheduler_fn(step):
            if step <= training_conifg.lr_warmup_steps:
                current_lr = training_conifg.lr * (step/training_conifg.lr_warmup_steps)
            elif step > training_conifg.training_steps - training_conifg.lr_warmdown_steps:
                current_lr = max(training_conifg.lr * (training_conifg.training_steps - step)/training_conifg.lr_warmdown_steps, 0)
            else:
                current_lr = training_conifg.lr
            return current_lr
        
        self.scheduler_function = scheduler_fn
        self.lr = self.scheduler_function(self.steps)
    
    def step(self):
        self.steps += 1
        self.lr = self.scheduler_function(self.steps)


class L1Scheduler:
    def __init__(self, training_conifg):
        self.steps = 1
        
        def scheduler_fn(step):
            if step <= training_conifg.l1_warmup_steps:
                current_l1 = training_conifg.l1 * (step/training_conifg.l1_warmup_steps)
            else:
                current_l1 = training_conifg.l1
            return current_l1
        
        self.scheduler_function = scheduler_fn
        self.l1 = self.scheduler_function(self.steps)
    
    def step(self):
        self.steps += 1
        self.l1 = self.scheduler_function(self.steps)


@dataclass
class SAEOutput():
    weighted_hidden_acts: Float[Tensor, "batch d_sae"]
    reconstructed_acts: Float[Tensor, "batch d_in"]


@dataclass
class GatedTrainingOutput():
    weighted_hidden_acts: Float[Tensor, "batch d_sae"]
    reconstructed_acts: Float[Tensor, "batch d_in"]
    weighted_gated_acts: Float[Tensor, "batch d_sae"]
    aux_reconstructed_acts: Float[Tensor, "batch d_in"]
    

class TrainingModeError(Exception):
    """Exception raised for calling a method in training mode that should only be called in eval mode."""
    pass


def vanilla_loss(l1_scheduler, original_acts, output: SAEOutput):
    mse_loss = F.mse_loss(output.reconstructed_acts, original_acts, reduction='mean')
    sparsity_loss = output.weighted_hidden_acts.norm(p = 1, dim = -1).mean()
    l1 = l1_scheduler.l1
    total_loss = mse_loss + l1 * sparsity_loss
    return mse_loss, sparsity_loss, total_loss

def gated_loss(l1_scheduler, original_acts, output: GatedTrainingOutput):
    mse_loss = F.mse_loss(output.reconstructed_acts, original_acts, reduction='mean')
    aux_mse_loss = F.mse_loss(output.aux_reconstructed_acts, original_acts, reduction='mean')
    sparsity_loss = output.weighted_gated_acts.norm(p = 1, dim = -1).mean()
    l1 = l1_scheduler.l1
    total_loss = mse_loss + aux_mse_loss + l1 * sparsity_loss
    return mse_loss, sparsity_loss, total_loss

def calculate_explained_variance(original_acts, reconstructed_acts):
    residual_varaince = (original_acts - reconstructed_acts).var(dim=0, unbiased=False).sum().item()
    original_variance = original_acts.var(dim=0, unbiased=False).sum().item()
    explained_variance = 1 - residual_varaince/original_variance
    return explained_variance

def calculate_coefficient_of_determination(original_acts, reconstructed_acts):
    mean_square_residual = (original_acts - reconstructed_acts).square().mean(dim = 0).sum().item()
    original_variance = original_acts.var(dim=0, unbiased=False).sum().item()
    coefficient_of_determination = 1 - mean_square_residual/original_variance
    return coefficient_of_determination

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
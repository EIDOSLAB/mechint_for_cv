"""
To do:
    refactor this script to seperate WandB and training. Can I package all the information into a dataclass and pass that off to a seperate function? Currently it's a mess.
"""

from torch.utils.data import DataLoader
from utils_training import (
    L1Scheduler,
    LRScheduler,
    vanilla_loss,
    gated_loss,
    calculate_coefficient_of_determination,
    calculate_explained_variance,
    get_device)
from training_config import TrainingConfig
from sae import SAE
from huggingface_dataset import SAEHuggingFaceDataset
from functools import partial
from tqdm import trange
import torch
import wandb
import os

def train_sae(sae: SAE, training_config: TrainingConfig, dataset: SAEHuggingFaceDataset):
    sae.train()
    device = get_device()
    print(f'Beginning training on device: {device}.')
    sae.to(device)
    steps = 0
    dataset_length = len(dataset)
    epochs = int((training_config.training_steps * training_config.batch_size) / dataset_length)
    
    if training_config.log_to_wandb:
        wandb.init(project = training_config.wandb_project, config = training_config.to_dict() | sae.config.to_dict())
        num_data_points_fired = torch.zeros(sae.config.d_sae, device = device)
        num_data_points = 0
    
    lr_scheduler = LRScheduler(training_config)
    l1_scheduler = L1Scheduler(training_config)
    optimizer = torch.optim.Adam(sae.parameters(), lr = lr_scheduler.lr)
    if sae.config.sae_type == "vanilla":
        loss_fn = partial(vanilla_loss, l1_scheduler)
    elif sae.config.sae_type == 'gated':
        loss_fn = partial(gated_loss, l1_scheduler)
    elif sae.config.sae_type == "top_k":
        # Need to think about this in the future, currently a bit of a bodge. Eg we don't need an L1 scheduler/sparsity loss for TopK
        loss_fn = partial(vanilla_loss, l1_scheduler)
    else:
        raise Exception(f"SAE architecture {sae.config.sae_type} is not currently suported.")
    
    # train_dataloader = DataLoader(dataset, batch_size = training_config.batch_size, shuffle = True, num_workers=training_config.num_workers)
    train_dataloader = DataLoader(dataset.activations, batch_size = training_config.batch_size, shuffle = True, num_workers=training_config.num_workers)
    for epoch in trange(epochs, desc = 'Number of training epochs'):
        did_fire_in_epoch = torch.zeros(sae.config.d_sae, device = device, dtype = torch.bool)
        
        for original_acts in train_dataloader:
            if steps >= training_config.training_steps:
                break
            original_acts = original_acts.to(device)
            optimizer.zero_grad()
            output = sae(original_acts)
            mse_loss, sparsity_loss, total_loss = loss_fn(original_acts, output)
            if sae.config.sae_type == 'top_k':
                mse_loss.backward()
            else:
                total_loss.backward()
            if training_config.normalise_decoder:
                sae.decoder.normalise()
                sae.decoder.adjust_gradients_to_normalise()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), training_config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            l1_scheduler.step()
            #update the lr in the optimizer by hand:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scheduler.lr
            steps += 1
            
            with torch.no_grad():
                if training_config.log_to_wandb:
                    did_fire_in_batch = torch.any(output.weighted_hidden_acts>0, dim = 0)
                    did_fire_in_epoch = torch.logical_or(did_fire_in_epoch, did_fire_in_batch)
                    num_data_points_fired += (output.weighted_hidden_acts > 0).float().sum(0)
                    num_data_points += training_config.batch_size
            
                    if steps % training_config.wandb_log_frequency == 0:
                        explained_varaince = calculate_explained_variance(original_acts, output.reconstructed_acts)
                        coefficient_of_determination = calculate_coefficient_of_determination(original_acts, output.reconstructed_acts)
                        wandb.log({
                            'losses/l1': sparsity_loss.item(),
                            'losses/mse': mse_loss.item(),
                            'losses/total_loss': total_loss.item(),
                            'metrics/l0': (output.weighted_hidden_acts > 0).float().sum(dim = -1).mean().item(),
                            'metrics/explained_variance': explained_varaince,
                            'metrics/coefficient_of_determination': coefficient_of_determination,
                            'sae/mean_decoder_norm': sae.decoder.W_dec.data.norm(dim = 0).mean().item(),
                            'sae/mean_encoder_norm': sae.encoder.W_enc.data.norm(dim = -1).mean().item(),
                            'activations/mean_input_square_norm': original_acts.norm(dim = -1).square().mean().item(),
                            'activations/input_variance': original_acts.var(dim=0, unbiased=True).sum().item(),
                            'activations/mean_output_square_norm': output.reconstructed_acts.norm(dim = -1).square().mean().item(),
                            'activations/output_varaince': output.reconstructed_acts.var(dim=0, unbiased=True).sum().item(),
                            'training_params/lr': lr_scheduler.lr,
                            'training_params/l1': l1_scheduler.l1,
                        }, step = steps)
                        
                    if num_data_points >= training_config.histogram_window:
                        feature_sparsity = num_data_points_fired / num_data_points
                        log_feature_sparsity = torch.log10(feature_sparsity + 1e-7).cpu()
                        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                        wandb.log(
                            {   
                                "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                                "plots/feature_density_line_chart": wandb_histogram,
                            }, step=steps)
                        num_data_points = 0
                        num_data_points_fired = torch.zeros(sae.config.d_sae, device = device)
                
                if training_config.save_sae and steps % training_config.save_frequency == 0:
                    os.makedirs(training_config.checkpoints_path, exist_ok=True)
                    torch.save({
                        'sae_state_dict': sae.state_dict(),
                        'sae_config': sae.config,
                        'training_config': training_config,
                        'training_steps_taken': steps,
                    }, f'{training_config.checkpoints_path}/{sae.config.name}.pth')
                    
        if steps >= training_config.training_steps:
            break
                    
        with torch.no_grad():
            if training_config.log_to_wandb:
                wandb.log({
                    'metrics/number_of_dead_neurons': (did_fire_in_epoch == False).sum().item(),
                    }, step = steps)
                    
    if training_config.log_to_wandb:
        wandb.finish()
    
    if training_config.save_sae:
        os.makedirs(training_config.checkpoints_path, exist_ok=True)
        torch.save({
            'sae_state_dict': sae.state_dict(),
            'sae_config': sae.config,
            'training_config': training_config,
            'training_steps_taken': steps,
        }, f'{training_config.checkpoints_path}/{sae.config.name}.pth')
    
    if training_config.upload_to_huggingface:
        sae.push_to_hub(f"{training_config.huggingface_org_name}/{sae.config.name}", private=True)

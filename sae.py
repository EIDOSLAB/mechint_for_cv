import torch
import torch.nn.functional as F
import einops
from jaxtyping import Float
from typing import overload, Union
from typing import Optional
from torch import Tensor, nn
from tqdm import tqdm
from functools import partial
from sae_config import SAEConfig
from utils_training import TrainingModeError, SAEOutput, GatedTrainingOutput
from huggingface_hub import PyTorchModelHubMixin
from dataclasses import dataclass
import safetensors.torch as st


class VanillaEncoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU()
        self.W_enc = nn.Parameter(
            torch.empty(self.config.d_sae, self.config.d_in)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.config.d_sae)
        )
        
        with torch.no_grad():
            for row in range(config.d_sae):
                self.W_enc[row] = torch.randn(config.d_in)
            self.W_enc.data /= torch.norm(self.W_enc.data, dim = -1, keepdim = True)
            self.W_enc.data *= config.init_norm
    
    def initialise_biases_with_mean(self, mean: Float[Tensor, "d_in"]):
        """
        Sets the encoder biases to be minus W_enc @ mean.
        This has the effect of removing the mean from the data.
        """
        if mean.shape != (self.config.d_in,):
            raise ValueError(f"The tensor passed to initialise the encoder biases does not have the correct shape. The tensor must have shape [{self.config.d_in}]. Note that this is different to the shape of the encoder biases.")
        with torch.no_grad():
            self.b_enc.data = - mean.clone() @ self.W_enc.t()
            
    def forward(self, x):
        x = x @ self.W_enc.t() + self.b_enc
        hidden_acts = self.relu(x)
        return hidden_acts
        

class GatedEncoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU()
        self.W_enc = nn.Parameter(
            torch.empty(self.config.d_sae, self.config.d_in)
        )
        self.r_mag = nn.Parameter(
            torch.zeros(self.config.d_sae)
        )
        self.b_gate = nn.Parameter(
            torch.zeros(self.config.d_sae)
        )
        self.b_mag = nn.Parameter(
            torch.zeros(self.config.d_sae)
        )
        
        with torch.no_grad():
            for row in range(config.d_sae):
                self.W_enc[row] = torch.randn(config.d_in)
            self.W_enc.data /= torch.norm(self.W_enc.data, dim = -1, keepdim = True)
            self.W_enc.data *= config.init_norm
            
    def initialise_biases_with_mean(self, mean: Float[Tensor, "d_in"]):
        """
        Sets the encoder biases to be minus W_enc @ mean.
        This has the effect of removing the mean from the data.
        """
        if mean.shape != (self.config.d_in,):
            raise ValueError(f"The tensor passed to initialise the encoder biases does not have the correct shape. The tensor must have shape [{self.config.d_in}]. Note that this is different to the shape of the encoder biases.")
        with torch.no_grad():
            self.b_mag.data = - mean.clone() @ self.W_enc.t()
            self.b_gate.data = - mean.clone() @ self.W_enc.t()
            
    def forward(self, x):
        x = x @ self.W_enc.t()
        pi_gate = x + self.b_gate
        pi_mag = x * torch.exp(self.r_mag) + self.b_mag
        with torch.no_grad():
            f_gate = torch.heaviside(pi_gate.detach(), torch.zeros_like(pi_gate))
        f_mag = self.relu(pi_mag)
        hidden_acts = f_gate * f_mag
        if self.training:
            relu_gated_acts = self.relu(pi_gate)
            return hidden_acts, relu_gated_acts
        else:
            return hidden_acts

class TopKEncoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.W_enc = nn.Parameter(torch.empty(self.config.d_sae, self.config.d_in))
        self.b_enc = nn.Parameter(torch.zeros(self.config.d_sae))
        self.relu = nn.ReLU()
        
        with torch.no_grad():
            for row in range(config.d_sae):
                self.W_enc[row] = torch.randn(config.d_in)
            self.W_enc.data /= torch.norm(self.W_enc.data, dim=-1, keepdim=True)
            self.W_enc.data *= config.init_norm
    
    def initialise_biases_with_mean(self, mean: Float[Tensor, "d_in"]):
        """
        Sets the encoder biases to be minus W_enc @ mean.
        This has the effect of removing the mean from the data.
        """
        if mean.shape != (self.config.d_in,):
            raise ValueError(f"The tensor passed to initialise the encoder biases does not have the correct shape. The tensor must have shape [{self.config.d_in}]. Note that this is different to the shape of the encoder biases.")
        with torch.no_grad():
            self.b_enc.data = - mean.clone() @ self.W_enc.t()
    
    def forward(self, x):
        # Differentiable Top K activation
        z = x @ self.W_enc.t() + self.b_enc
        topk = torch.topk(z, k=self.config.k, dim=-1)
        # Note the use of relu: only the positive activations are kept. This is inline with OpenAI's implementation.
        values = self.relu(topk.values)
        result = torch.zeros_like(z)
        result.scatter_(-1, topk.indices, values)
        return result

class Decoder(nn.Module):
    def __init__(self, config: SAEConfig, decoder_weights: Optional[Float[Tensor, "d_in d_sae"]] = None):
        super().__init__()
        self.config = config
        self.W_dec = nn.Parameter(torch.empty(self.config.d_in, self.config.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.config.d_in))
        
        with torch.no_grad():
            if decoder_weights is None:
                for row in range(config.d_sae):
                    self.W_dec[:, row] = torch.randn(config.d_in)
                self.W_dec.data /= torch.norm(self.W_dec.data, dim=0, keepdim=True)
                self.W_dec.data *= config.init_norm
            else:
                self.W_dec.data = decoder_weights
                
    def initialise_biases_with_mean(self, mean: Float[Tensor, "d_in"]):
        if mean.shape != (self.config.d_in,):
            raise ValueError(f"The tensor passed to initialise the decoder biases does not have the correct shape. The tensor must have shape [{self.config.d_in}].")
        with torch.no_grad():
            self.b_dec.data = mean.clone()
    
    @torch.no_grad()
    def normalise(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=0)
    
    @torch.no_grad()
    def adjust_gradients_to_normalise(self):
        if self.W_dec.grad is None:
            raise ValueError('The decoder weights have no gradient.')
        
        gradient_dot_products = torch.sum(self.W_dec.grad * self.W_dec, dim = 0)
        squared_row_norms = torch.sum(self.W_dec * self.W_dec, dim = 0)
        projected_gradients = self.W_dec * (gradient_dot_products / squared_row_norms).unsqueeze(0)
        self.W_dec.grad -= projected_gradients
    
    def forward(self, x):
        return x @ self.W_dec.t() + self.b_dec


class SAE(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        if config.sae_type == 'vanilla':
            self.encoder = VanillaEncoder(config)
        elif config.sae_type == 'gated':
            self.encoder = GatedEncoder(config)
        elif config.sae_type == 'top_k':
            self.encoder = TopKEncoder(config)
        else:
            raise TypeError(f'The architecture {self.config.sae_type} is not recognised.')
        self.decoder = Decoder(config, decoder_weights=self.encoder.W_enc.data.t().clone())

    def initialise_biases_with_mean(self, mean: Float[Tensor, "d_in"]):
        """
        Initialises the biases in a way that has the effect of removing the mean from the data.
        """
        self.encoder.initialise_biases_with_mean(mean)
        self.decoder.initialise_biases_with_mean(mean)
        
    def top_k_forward(self, x):
        hidden_acts = self.encoder(x)
        reconstructed_acts = self.decoder(hidden_acts)
        return SAEOutput(hidden_acts, reconstructed_acts)
    
    def gated_forward(self, x):
        if self.training:
            hidden_acts, relu_gated_acts = self.encoder(x)
            weighted_gated_acts = relu_gated_acts * self.decoder.W_dec.norm(dim = 0)
            aux_reconstructed_acts = self.decoder(relu_gated_acts)
        else:
            hidden_acts = self.encoder(x)
            
        weighted_hidden_acts = hidden_acts * self.decoder.W_dec.norm(dim = 0)
        reconstructed_acts = self.decoder(hidden_acts)
        
        if self.training:
            return GatedTrainingOutput(weighted_hidden_acts, reconstructed_acts, weighted_gated_acts, aux_reconstructed_acts)
        else:
            return SAEOutput(weighted_hidden_acts, reconstructed_acts)
        
    def vanilla_forward(self, x):
        hidden_acts = self.encoder(x)
        weighted_hidden_acts = hidden_acts * self.decoder.W_dec.norm(dim = 0)
        reconstructed_acts = self.decoder(hidden_acts)
        return SAEOutput(weighted_hidden_acts, reconstructed_acts)
        
    def forward(self, x):
        if self.config.sae_type == 'vanilla':
            return self.vanilla_forward(x)
        elif self.config.sae_type == 'gated':
            return self.gated_forward(x)
        elif self.config.sae_type == 'top_k':
            return self.top_k_forward(x)
        else:
            raise TypeError(f'The architecture {self.config.sae_type} is not currently supported.')
            
    
    def encode(self, x):
        """
        Returns the decoder-normalised SAE features without passing through the decoder.
        """
        if self.training:
            raise TrainingModeError("The SAE.encode() method can only be called in evaluation mode.")
        hidden_acts = self.encoder(x)
        if self.config.sae_type in ['vanilla', 'gated']:
            weighted_hidden_acts = hidden_acts * self.decoder.W_dec.norm(dim=0)
        elif self.config.sae_type in ['top_k']:
            # We don't need to weight the hidden activations for the TopK SAE.
            # Actually, is this even true? Eg can't we map this back to the old SAEs?
            weighted_hidden_acts = hidden_acts
        return weighted_hidden_acts
    
    def contiguous(self):
        model_state_dict = self.state_dict()
        # Make each tensor contiguous
        for key, tensor in model_state_dict.items():
            model_state_dict[key] = tensor.contiguous()
    
    def push_to_hub(self, *args, **kwargs):
        """
        Overrides the base class method, to ensure contiguous memory for safetensors before pushing to the hub.
        """
        self.contiguous()
        super().push_to_hub(*args, **kwargs)
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Overrides the base class method with a monkey patch on safe tensors.
        This prevents an error when calling from_pretrained for reasons I don't understand!
        """ 
        def patched_end_ptr(tensor):
            stop = tensor.reshape(-1)[-1].data_ptr() + st._SIZE[tensor.dtype]
            return stop
        # Monkey patch the function
        st._end_ptr = patched_end_ptr
        
        instance = super().from_pretrained(*args, **kwargs)
        return instance
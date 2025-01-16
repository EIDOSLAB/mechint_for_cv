import os
from tqdm import tqdm
from PIL import Image
import wandb
import datasets
from datasets import Dataset
import sae_config
import sae
import torch
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from hooked_vit import HookedVisionTransformer
from imagenet_classes import IMAGENET2012_CLASSES

from transformers import CLIPProcessor, CLIPModel


# Config variables

class_token = True
batch_size = 1024
image_width = 224
image_height = 224
model_name =  "openai/clip-vit-large-patch14"
module_name = "resid"
block_layer = -2
dataset_path = "evanarlian/imagenet_1k_resized_256"
cached_activations_path = None
d_in = 1024
expansion_factor = 64
b_dec_init_method = "mean"
total_images = 2621440

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print('Device:')
print(device)

target_labels = list(IMAGENET2012_CLASSES.items())

# Dataset loading
if  os.path.exists(f'/scratch/datasets/{dataset_path}'):
    dataset = load_from_disk(f'/scratch/datasets/{dataset_path}')
else:
    dataset = load_dataset(dataset_path, split="train")
    dataset.save_to_disk(f'/scratch/datasets/{dataset_path}')

# ViT instance

hViT= HookedVisionTransformer(model_name)
print(hViT.model)
current_batch = 0
list_of_hook_locations = [(block_layer, module_name)]
dp = dataset_path.split('/')[-1]
mn = model_name.split('/')[-1]

# Here goes the creation of the SAE dataset
# The dataset is composed of the ViT activations in imagenet

for current_batch in tqdm(range((total_images//batch_size))):
    # file_path = f'../../sae_on_vit/{mn}_activations_{dp}_{current_batch}.pt'
    file_path = f'/scratch/sae_on_vit/{mn}_activations_{dp}_{current_batch}.pt'
    if not os.path.exists(file_path):
        vit_input = hViT.processor(images=dataset[batch_size*(current_batch):batch_size*(current_batch+1)]['image'], text = "", return_tensors="pt", padding = True).to(device)

        # Inferences
        activations = hViT.run_with_cache(
            list_of_hook_locations,
            **vit_input,
        )[1][(block_layer, module_name)]

        activations = activations[:, 0, :]
        torch.save(activations, file_path)

exit()




# The created dataset must be a HuggingFace Dataset object



scaling_factor = dataset.normalise(d_in ** 0.5) # As per Anthropic’s recommendation we need to normalise the dataset. I then calculate the scale factor that we have scaled cls tokens so that later at inference time we can scale tokens dynamically.

# SAE configuration object initialization
sae_config = sae_config.SAEConfig(
    sae_type='gated',
    expansion_factor=64,
    d_in=d_in,
    init_norm=1,
    scaling_factor=scaling_factor, # This is needed to scale inputs to the SAE at inference time after training – eg to normalise future cls tokens in the same way as the training data.
    name="gated_sae"
)

# SAE object from configuration object 
sae = sae.SAE(sae_config)


# Misc suggested by Hugo and Anthropic
sae.config.scaling_factor = scaling_factor # This is needed to scale inputs to the SAE at inference time after training – eg to normalise future cls tokens in the same way as the training data.
mean = dataset.get_mean()
sae.initialise_biases_with_mean(mean) # Maybe initialse the SAE biases to zero out the mean? This is optional though!

# SAE training
train_sae(sae, training_config, dataset) # Now train the SAE

# Save data/relevant images

print('Done.')

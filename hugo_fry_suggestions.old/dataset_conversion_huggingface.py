from glob import glob
from datasets import Dataset
import torch
import numpy as np
from huggingface_dataset import SAEHuggingFaceDataset

# Load the activations
base_name = 'clip-vit-large-patch14_activations_imagenet_1k_resized_256_*.pt'
print(base_name)
activations_files = glob('../../sae_on_vit/' + base_name)

# Turn the activations into a dataset
activations = []
for file in activations_files:
    print(file)
    activations.append(torch.load(file))
    break
activations = torch.cat(activations)
activations = activations.cpu().numpy()
activations = activations.astype(np.float32)

# Create Huggingface dataset from list
dataset = SAEHuggingFaceDataset(activations)

print(dataset.get_mean())
print('*'*10)
print(dataset.get_mean() == np.mean(activations, axis=0))
print('*'*10)
print(dataset.normalize(2))
print('*'*10)
print(dataset.dataset[0]['activations'])
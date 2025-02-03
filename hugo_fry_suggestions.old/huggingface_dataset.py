from datasets import Dataset
import numpy as np
import torch

# The constructor takes as argument a numpy array of activations and generates a dataset from list
# The class also has a get_mean method that returns the mean of the activations
# And also has a function that normalizes the values of the activations by dividing them for a passed argument

class SAEHuggingFaceDataset():
    def __init__(self, activations: np.array):
        self.activations = activations
        activations_dict = [{'activation' : torch.from_numpy(act)} for act in activations]
        # activations_list = [torch.from_numpy(act) for act in activations]
        # activations_list = [act.tolist() for act in activations]
        self.dataset = Dataset.from_list(activations_dict)
        
    def get_mean(self):
        return np.mean(self.activations, axis=0)
    
    def normalize(self, value):
        return self.activations/value

    # Add a len function that returns the number of activations
    def __len__(self):
        return len(self.activations)
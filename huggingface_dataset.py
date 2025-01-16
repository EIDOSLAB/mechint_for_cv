from datasets import Dataset
import numpy as np

# The constructor takes as argument a numpy array of activations and generates a dataset from list
# The class also has a get_mean method that returns the mean of the activations
# And also has a function that normalizes the values of the activations by dividing them for a passed argument

class SAEHuggingFaceDataset():
    def __init__(self, activations: np.array):
        self.activations = activations
        self.dataset = Dataset.from_dict({'activations' : activations})
        
    def get_mean(self):
        return np.mean(self.activations, axis=0)
    
    def normalize(self, value):
        return self.activations/value
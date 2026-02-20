import torch
import numpy as np
from torch.utils.data import Dataset
import os

class SpecTransform:
    def __call__(self,np_array):
        # Convert to float tensor
        tensor = torch.from_numpy(np_array).float()
        # Add the channel dim
        return tensor.unsqueeze(0)
        
class PopJazzDataset(Dataset):
    def __init__(self, pop_root, jazz_root):
        # Initialize root directory 
        self.pop_root = pop_root
        self.jazz_root = jazz_root
        
        # Take just the file names, for memory efficiency
        self.pop = os.listdir(pop_root)
        self.jazz = os.listdir(jazz_root)
        
        # Specify the len
        self.data_len = max(len(self.pop), len(self.jazz))
        self.pop_len = len(self.pop)
        self.jazz_len = len(self.jazz)
        
        # Initialize the transformation
        self.transform = SpecTransform()
        
    def __len__(self):
        return self.data_len
    def __getitem__(self,idx):
        # Use modulus operator to avoid indexing error
        pop_idx = idx % self.pop_len
        jazz_idx = idx % self.jazz_len
        
        # Construct the path to the img and load the path
        pop_path = os.path.join(self.pop_root, self.pop[pop_idx])
        jazz_path = os.path.join(self.jazz_root, self.jazz[jazz_idx])
        
        pop_file = np.load(pop_path)
        jazz_file = np.load(jazz_path)
        
        return {"pop": self.transform(pop_file), "jazz": self.transform(jazz_file)}
        
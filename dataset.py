import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import config

class AudioDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img = self.data[idx]
        
        # Convert to Float Tensor
        img = torch.tensor(img, dtype=torch.float32)
        # Add Channel Dimension
        img = img.unsqueeze(0) # (1,128,256)
        return img,0
    
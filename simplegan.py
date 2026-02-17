import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Create the Discriminator
# Linear layer: Linear (fanin, 128) -> LeakyRELU (0.1) -> Linear (128,1) -> Sigmoid

class Discriminator(nn.Module):
    def __init__(self, fanin):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(fanin, 2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.disc(x)

# Create the Generator

# Still linear: Linear (noise_dim, 128) -> LeakyRELU(0.1) -> Linear(128,img_dim) -> Tanh
# Tanh is for matching with the previous normalization step

class Generator(nn.Module):
    def __init__(self,noise_dim,img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # We need more neurons because the output is huge
            nn.Linear(noise_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 4096), # Increased hidden layer size
            nn.LeakyReLU(0.1),
            nn.Linear(4096, img_dim),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.gen(x)
    
# Setting hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
noise_dim = 100
img_dim = 128 * 256
batch_size = 32
epochs = 50

# Initializing 
disc = Discriminator(img_dim).to(device)
gen = Generator(noise_dim,img_dim).to(device)

opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)

transform = transforms.Compose([
    transforms.ToTensor(),
    # This normalization step match with the Tanh
    transforms.Normalize((0,5,), (0,5,)) 
])

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

dataset = AudioDataset('train_data.npy')
loader = DataLoader(dataset, batch_size= batch_size, shuffle = True)

criterion = nn.BCELoss()
# Trainning

for epoch in range(epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.view(-1, img_dim).to(device)
        # PHASE 1: Train the discriminator 
        
        # Get some fake images from the generator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_img = gen(noise)
        # Minimize two losses: First is the loss from fake_img (label = 0), and loss from real images(label = 1)
        # Detach the fake_img from disc_fake further usage
        disc_fake = disc(fake_img.detach()).view(-1)
        disc_real = disc(real).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD = (lossD_fake + lossD_real) / 2 # Division by 2 is kinda a best practice
        # Perform backprop
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()
        
        # PHASE 2: Train the Generator
        # Reuse the fake_img and put it in the discriminator again (after making it smarter)
        output = disc(fake_img).view(-1)
        # Pretend all fakes images are real (label = 1)-> Make the generator to learn in order to provide real images
        lossG = criterion(output, torch.ones_like(output))
        # Backprop
        gen.zero_grad()
        lossG.backward()
        opt_gen.step() 
        
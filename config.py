import torch
# Setting hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
noise_dim = 100
img_dim = 128 * 256
batch_size = 32
epochs = 50

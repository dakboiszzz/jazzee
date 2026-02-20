import torch
import random
import os
import numpy as np
import config
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint: {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Reset learning rate to the current config value
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_spectrogram_samples(gen_j, gen_p, pop, jazz, epoch, folder="saved_spectrograms"):
    """
    Saves a snapshot of real vs. fake spectrograms to visualize training progress.
    Since your shape is (N, 1, 128, 256), they map perfectly to grayscale images.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Put generators in eval mode to disable dropout/batchnorm updates
    gen_j.eval()
    gen_p.eval()
    
    with torch.no_grad():
        fake_jazz = gen_j(pop)
        fake_pop = gen_p(jazz)
        
        # Take up to 4 samples from the batch to avoid huge image files
        # We assume your outputs are normalized to [-1, 1] (using Tanh in Generator).
        # Multiplying by 0.5 and adding 0.5 brings them to [0, 1] for saving.
        save_image(pop[:4] * 0.5 + 0.5, f"{folder}/real_pop_epoch_{epoch}.png")
        save_image(fake_jazz[:4] * 0.5 + 0.5, f"{folder}/fake_jazz_epoch_{epoch}.png")
        save_image(jazz[:4] * 0.5 + 0.5, f"{folder}/real_jazz_epoch_{epoch}.png")
        save_image(fake_pop[:4] * 0.5 + 0.5, f"{folder}/fake_pop_epoch_{epoch}.png")
        
    # Revert to training mode
    gen_j.train()
    gen_p.train()

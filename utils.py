import torch
import random
import os
import numpy as np
import config
from torchvision.utils import save_image

import librosa
import soundfile as sf
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
def spec_to_sound(mel, sr=22050):
    """Reverses the normalization, converts to power, and estimates the audio waveform."""
    mel = (mel + 1) / 2 * 80 - 80
    mel_power = librosa.db_to_power(mel)
    return librosa.feature.inverse.mel_to_audio(mel_power, sr=sr)

def save_audio_samples(gen_j, gen_p, pop, jazz, epoch, folder="saved_audio"):
    """
    Saves a snippet of real vs. fake audio to listen to training progress.
    Grabs the first sample in the batch, converts tensor to numpy, and processes to .wav.
    """
    os.makedirs(folder, exist_ok=True)
    
    gen_j.eval()
    gen_p.eval()
    
    with torch.no_grad():
        fake_jazz = gen_j(pop)
        fake_pop = gen_p(jazz)
        
        # 1. Grab the first item in the batch [0] and strip the channel dimension [0]
        # 2. Move from GPU to CPU (.cpu()) and convert to NumPy (.numpy())
        real_pop_np = pop[0, 0].cpu().numpy()
        fake_jazz_np = fake_jazz[0, 0].cpu().numpy()
        real_jazz_np = jazz[0, 0].cpu().numpy()
        fake_pop_np = fake_pop[0, 0].cpu().numpy()
        
        # Convert NumPy arrays to audio waveforms
        audio_real_pop = spec_to_sound(real_pop_np)
        audio_fake_jazz = spec_to_sound(fake_jazz_np)
        audio_real_jazz = spec_to_sound(real_jazz_np)
        audio_fake_pop = spec_to_sound(fake_pop_np)
        
        # Save as .wav files (These will be short clips, ~5-6 seconds each)
        sf.write(f"{folder}/epoch_{epoch}_real_pop.wav", audio_real_pop, 22050)
        sf.write(f"{folder}/epoch_{epoch}_fake_jazz.wav", audio_fake_jazz, 22050)
        sf.write(f"{folder}/epoch_{epoch}_real_jazz.wav", audio_real_jazz, 22050)
        sf.write(f"{folder}/epoch_{epoch}_fake_pop.wav", audio_fake_pop, 22050)
        
    gen_j.train()
    gen_p.train()
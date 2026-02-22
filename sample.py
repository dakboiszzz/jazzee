import torch
import torch.optim as optim
import librosa
import numpy as np
import soundfile as sf
import os

# Import your custom modules
from model import Generator
import config
import utils

def audio_to_chunks(audio_path):
    """
    Reads a single .wav file and returns a tensor of shape (N, 1, 128, 256)
    """
    y, sr = librosa.load(audio_path, sr=22050)

    # Transform into mel_spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize the data (-1 to 1)
    mel_spectrogram_db = np.clip((mel_spectrogram_db + 80) / 80.0 * 2 - 1, -1, 1)

    # Window slicing 
    num_sam = mel_spectrogram_db.shape[1]
    wind_size = 256
    ovrlap = 128

    data = []
    for i in range(0, num_sam - wind_size, ovrlap):
        data.append(mel_spectrogram_db[:, i : i + wind_size])
    
    # Convert to numpy array: shape becomes (N, 128, 256)
    data = np.array(data)
    
    # Add the channel dimension for the Generator: shape becomes (N, 1, 128, 256)
    data = np.expand_dims(data, axis=1)
    
    return data

def spec_to_sound(mel, sr=22050):
    """Reverses the spectrogram back to audio using Griffin-Lim."""
    # Reverse normalization
    mel = (mel + 1) / 2 * 80 - 80
    
    # Convert dB back to power
    mel_power = librosa.db_to_power(mel)
    
    # Inverse mel to audio
    return librosa.feature.inverse.mel_to_audio(mel_power, sr=sr)

def sample_pop_to_jazz(input_audio_path, output_audio_path):
    """
    Loads a pretrained Pop->Jazz generator, processes a full song in chunks,
    and saves the resulting styled audio.
    """
    print(f"🎸 Initializing Jazz Generator on {config.DEVICE}...")
    
    gen_j = Generator().to(config.DEVICE)
    checkpoint = torch.load("gen_j.pth.tar", map_location=config.DEVICE, weights_only=False)
    gen_j.load_state_dict(checkpoint["state_dict"])
    gen_j.eval() 
    
    print(f"🎵 Preprocessing input audio: {input_audio_path}...")
    chunks = audio_to_chunks(input_audio_path) 
    chunks_tensor = torch.tensor(chunks, dtype=torch.float32).to(config.DEVICE)
    N = chunks_tensor.shape[0]
    
    print(f"🎷 Converting {N} chunks to Jazz style...")
    jazzy_chunks = []
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=config.DEVICE):
            for i in range(N):
                chunk = chunks_tensor[i].unsqueeze(0) # Keep batch dim: (1, 1, 128, 256)
                fake_jazz = gen_j(chunk)
                
                # Squeeze to (128, 256) and move to CPU for numpy operations
                jazzy_chunks.append(fake_jazz.to(torch.float32).squeeze().cpu().numpy())
                
    print("🪡 Stitching spectrograms together...")
    output_spec = []
    
    # Overlap logic
    for i in range(N - 1):
        output_spec.append(jazzy_chunks[i][:, :128])
    output_spec.append(jazzy_chunks[-1])
    
    # Concatenate along the time axis
    final_spec = np.concatenate(output_spec, axis=1)
    
    print("🔊 Synthesizing audio from spectrogram...")
    audio = spec_to_sound(final_spec, sr=22050)
    
    print(f"💾 Saving to {output_audio_path}")
    sf.write(output_audio_path, audio, 22050)
    print("Done bro!")
    
sample_pop_to_jazz('phepmaupop.wav', 'phepmaujazz.wav')
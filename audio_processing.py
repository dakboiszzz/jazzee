import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

def preprocess(folder_path, save_path):
    all_chunks = []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for file in files:
        audio_path = os.path.join(folder_path, file)
        # Load data
        y, sr = librosa.load(audio_path, sr = 22050)

        # Transform into mel_spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref = np.max)

        # Normalize the data (-1 to 1)
        mel_spectrogram_db = np.clip((mel_spectrogram_db + 80) / 80.0 * 2 - 1, -1,1)

        # Window slicing 
        num_sam = mel_spectrogram_db.shape[1]
        wind_size = 256
        ovrlap = 128

        data = []
        for i in range(0,num_sam - wind_size, ovrlap):
            data.append(mel_spectrogram_db[:, i : i + wind_size])
        
        all_chunks.extend(data)
        
        for idx, chunk in enumerate(data):
            np.save(os.path.join(save_path, f"{file[:-4]}_chunk_{idx}.npy"), chunk)
    
# preprocess('pop', 'pop_train')
# preprocess('jazz', 'jazz_train')
def spec_to_sound(mel, sr = 22050):
    # Reverse this
    # mel_spectrogram_db = np.clip((mel_spectrogram_db + 80) / 80.0 * 2 - 1, -1,1)
    mel = (mel + 1)/2 * 80 - 80
    # Convert to power
    mel_power = librosa.db_to_power(mel)
    
    return librosa.feature.inverse.mel_to_audio(mel_power, sr = sr)
def postprocess(file_path, save_path):
    data = np.load(file_path)
    batch_size = data.shape[0]
    output = []
    # Append just the first 128 (avoid overlapping)
    for i in range(batch_size - 1):
        output.append(data[i][:, :128])
    # Append the FULL last 
    output.append(data[-1])
    spec = np.concatenate(output, axis = 1)
    audio = spec_to_sound(spec)
    print(f'Saving to {save_path}')
    sf.write(save_path,audio,22050)
    print("Done bro!")
# postprocess('input.npy', 'constructed.wav')
"""
    # Display
    plt.figure(figsize= (10,4))
    librosa.display.specshow(mel_spectrogram_db, x_axis= 'time', y_axis= 'mel', sr = sr, cmap = 'viridis')
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()
"""
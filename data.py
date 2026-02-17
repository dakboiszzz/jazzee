import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
        
    final_data = np.array(all_chunks)
    np.save(save_path,final_data)
preprocess('jazz', 'train_data.npy')
"""
    # Display
    plt.figure(figsize= (10,4))
    librosa.display.specshow(mel_spectrogram_db, x_axis= 'time', y_axis= 'mel', sr = sr, cmap = 'viridis')
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()
"""
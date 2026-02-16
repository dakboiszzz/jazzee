import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = 'dataset_part001.wav'

y, sr = librosa.load(audio_path)

mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref = np.max)


plt.figure(figsize= (10,4))
librosa.display.specshow(mel_spectrogram_db, x_axis= 'time', y_axis= 'mel', sr = sr, cmap = 'viridis')
plt.colorbar(format = '+2.0f dB')
plt.title('Mel spectrogram')
plt.show()
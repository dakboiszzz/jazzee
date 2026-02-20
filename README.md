# Style-transfer with music using CycleGAN 

This is a fun project created by me to convert any songs into a jazz-like style. It's my first time touching in the filed of Audio Processing. 

## Project Overview
1. **Audio Processing**

Actually, this is the only thing that makes the project different from other CycleGAN projects (they normally dealing with images). What I did was:
- ***Convert the wav file:*** Split the audio files into smaller chunks (6 seconds), then I convert the `.wav` file into **spectrograms**, using Short Time Fourier Transform (included in the `librosa` library)
- ***Use the Spectrogram as images with 1 channel to feed in the model***
- ***Inverse Transform***: In order to hear the things that we created, we need to have an invese function to convert spectrograms back to `.wav` files. This can be done by the Griffin-Lim algorithm (actually i'm just using a built-in function in the `librosa` lib). But I think I need to work further with this because the generated audio files are of extremely poor quality. 

> You can see more in my `audio_processing.py` file

2. **Getting the data**

The training data is taken from the ***GTZAN Dataset*** in a Kaggle competition. This includes 100 audio files (30 seconds long) for each genres.

I took the `jazz` and `pop` folder in the Dataset, but this is for training only. I actually want to convert some of my favorite Vietnamese pop songs to jazz.

3. **Architecture choice**

I'm using ***CycleGAN*** for this project, specifically:

### For the model
I reconstruct the model architecture that was used in the original paper (You can see it [here](https://arxiv.org/pdf/1703.10593))

- ***Discriminator***: PatchGAN, `C64-C128-C256-C512` with `Ck` denoted as Conv2d-InstanceNorm-LeakyReLU (detail in the code)
- ***Generator***: Resembling U-Net, with 2 downsampling blocks, 9 residual blocks, and 2 upsampling blocks

### For training
- Two discriminators and two generators were used
- The loss for the Generator is divided into 3 losses: The Adversarial Loss, The Cycle Loss, and the Identity Loss, using MSE and L1

## How to use

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast package management and execution.

1. **Initialize the project** (if you haven't already):
```bash
uv init
```

2. **Add the required dependencies:**

```bash
uv add torch torchvision torchaudio numpy librosa soundfile matplotlib tqdm
```

3. **Prepare the Data**

Make sure that you have a folder of pop song and jazz songs in the format of `.wav`, then run the `aduio_preproccessing.py` to convert them into `.npy` files

```bash
uv run audio_preprocessing.py
```

This will create training data, at folder `pop_train` and `jazz_train` respectively.

4. **Train the model**

Once the data is ready, you can start the CycleGAN training loop. The script will automatically save checkpoints and generate sample .wav files at the end of each epoch so you can listen to the model learning over time!

```bash
uv run train.py
```

## Side notes

Actually, during my learning process, I have to go from the simplest GAN model to the Pix2Pix model and then finally the CycleGAN, so I have some implementations included in the project as well

1. **SimpleGan**

You can see the file in my project, basically the architecture involves:
- Generator and Discriminator: Fully Connected Deep Neural Nets
- Training: Initialize a noise to produce fake images, so that the model leanrs to produce realistic images from the latent space 

2. **Pix2Pix**

This is a strong GAN model, with complex architecture of the Generator and the Discriminator (you can see in the `model_pix2pix.py`)
- Discriminator (Similar to that of Cycle GAN): PatchGAN, consecutive blocks of Conv2d-BatchNorm-LeakyReLU
- Generator: U-Net with skip connections, 7 layers down and up, with a bottleneck layer at the bottom

I haven't tried CycleGAN using this model architecture, so maybe in the future I can test it.
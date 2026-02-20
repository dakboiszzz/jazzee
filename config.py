import torch
# Setting hyperparams
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
NOISE_DIM = 100
IMG_DIM = 128 * 256
BATCH_SIZE = 32
EPOCHS = 50

LAMBDA_CYC = 10
# I'm not sure with this lambda
LAMBDA_ID = 5
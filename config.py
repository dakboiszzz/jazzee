import torch
# Setting hyperparams
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
BATCH_SIZE = 1
EPOCHS = 100

LAMBDA_CYC = 10
# I'm not sure with this lambda
LAMBDA_ID = 5

SAVE_MODEL = True
LOAD_MODEL = False
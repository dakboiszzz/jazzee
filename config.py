import torch
# Check for NVIDIA (cuda), then Apple Silicon (mps), then fallback to cpu
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
LR = 2e-4
BATCH_SIZE = 1
EPOCHS = 100

LAMBDA_CYC = 10
# I'm not sure with this lambda
LAMBDA_ID = 5

SAVE_MODEL = True
LOAD_MODEL = False
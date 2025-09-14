"""
Separate runner script for training FairFace age classification with ResNet.
This simply wraps the CLI call in Python, so you can run training directly from here.

Usage:
    python run_fairface_training.py

Make sure to update the paths below before running.
"""

import subprocess

# Update these paths before running
DATA_ROOT = r"C:\Users\abhij\Desktop\cognizant hackathon\dataset"

TRAIN_CSV = r"C:\Users\abhij\Desktop\cognizant hackathon\dataset\fairface-img-margin025-trainval\train_annotations.csv"
VAL_CSV   = r"C:\Users\abhij\Desktop\cognizant hackathon\dataset\fairface-img-margin025-trainval\val_annotations.csv"

OUT_PATH  = r"C:\Users\abhij\Desktop\cognizant hackathon\model\resnet_age_best.pth"

# Image directories relative to DATA_ROOT
TRAIN_IMAGES_DIR = "fairface-img-margin025-trainval"
VAL_IMAGES_DIR = "fairface-img-margin025-trainval"



# Training parameters
EPOCHS = 12
BATCH_SIZE = 64
LR = 1e-3

cmd = [
    "python", "fairface_age_resnet.py",
    "--data-root", DATA_ROOT,
    "--train-csv", TRAIN_CSV,
    "--val-csv", VAL_CSV,
    "--train-images-dir", TRAIN_IMAGES_DIR,
    "--val-images-dir", VAL_IMAGES_DIR,
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--lr", str(LR),
    "--out", OUT_PATH,
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd, check=True)
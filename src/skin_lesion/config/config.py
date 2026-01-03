from pathlib import Path

IMAGE_SIZE = (1024, 1024)
TRAIN_RATIO = 0.8
TRAIN_SEED = 42
DATA_DIR = Path("PROJECT_Data/")
FEATURES_DIR = Path("artifacts/features.npz")
LABELS_CSV_DIR = Path("ISIC-2017_Data_GroundTruth_Classification.csv")
MODEL_DIR = Path("artifacts/mlp_features.pt")
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

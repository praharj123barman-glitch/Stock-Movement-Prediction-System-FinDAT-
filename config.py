# config.py
import os
# Directory structure
DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
WINDOWS_DIR = os.path.join(DATA_DIR, "windows")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = "models"
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"

# Create directories
for directory in [DATA_DIR, CLEANED_DIR, PROCESSED_DIR, WINDOWS_DIR, SPLITS_DIR, 
                 MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data parameters
WINDOW_SIZE = 5  # 5 days per week
TRAIN_DAYS = 400
VAL_DAYS = 160  
TEST_DAYS = 180  # Remaining days
NUM_WEEKS = 4  # Number of weeks for long-term pattern (1,2,3,4)

HIDDEN_DIMS = [64]  
EMBEDDING_DIM = 8 
GRU_HIDDEN_DIM = 8 #  GRU hidden dimension
TRANSFORMER_HIDDEN_DIM = 64  # transformer hidden
TRANSFORMER_HEADS = 4  # Number of attention heads in transformer  
TRANSFORMER_LAYERS = 2  # Number of transformer layers  #to indirectly learn more features from indirect stocks
LEARNING_RATE = 0.001  # Default learning rate
BATCH_SIZE = 4 # Batch size for training
EPOCHS = 2  # Number of epochs
DELTA = 0.5  # Balancing parameter between rank and move loss
LAMBDA = 1e-5  # L2 regularization parameter

# Results directory
RESULTS_DIR_Ablation = "AblationDir"   #this is for ablation study
# Features to add as specified in Task2
FEATURES = [
    "open", "close", "high", "low",  # Base features
    "return_ratio",
    "pct_change_open", "pct_change_high", "pct_change_low",
    "ma_5", "ma_10", "ma_15", "ma_20", "ma_25", "ma_30",
    "up_or_down",
]

# Sector mapping will be loaded or created during preprocessing
SECTOR_MAPPING = {}


"""
Configuration for Habitat Classification Model

This file contains configuration parameters for the deep learning pipeline.
Paths are placeholders and should be updated according to your environment.
"""

import os


class Config:
    
    TRAIN_MASK_RAW_DIR = "./data/train/mask_raw"
    TRAIN_MASK_ONEHOT_DIR = "./data/train/mask_onehot"
    TRAIN_METADATA_DIR = "./data/train/metadata"

    
    VAL_MASK_RAW_DIR = "./data/val/mask_raw"
    VAL_MASK_ONEHOT_DIR = "./data/val/mask_onehot"
    VAL_METADATA_DIR = "./data/val/metadata"

    
    EXTERNAL_MASK_RAW_DIR = "./data/external/mask_raw"
    EXTERNAL_MASK_ONEHOT_DIR = "./data/external/mask_onehot"
    EXTERNAL_METADATA_DIR = "./data/external/metadata"

    
    TRAIN_LABEL_EXCEL_PATH = "./data/train_labels.xlsx"
    VAL_LABEL_EXCEL_PATH = "./data/val_labels.xlsx"
    EXTERNAL_LABEL_EXCEL_PATH = "./data/external_labels.xlsx"

    
    SAVE_DIR = "./results"
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    LOG_DIR = os.path.join(SAVE_DIR, "logs")
    FIG_DIR = os.path.join(SAVE_DIR, "figures")

    
    MASK_TYPE = 'onehot'  # 'raw' or 'onehot'
    INPUT_DEPTH = 4
    IN_CHANNELS = 15
    IMG_SIZE = 128
    NUM_CLASSES = 2

    
    BATCH_SIZE = 4
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    CLASS_WEIGHTS = [1.0, 2.0]
    EARLY_STOPPING_PATIENCE = 30  
    LR_SCHEDULER = 'cosine'
    USE_AUC_LOSS = True
    VAL_RATIO = 0.33

    
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    MIXED_PRECISION = True
    USE_AUGMENTATION = True

    @classmethod
    def create_directories(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.FIG_DIR, exist_ok=True)
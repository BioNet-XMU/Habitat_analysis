"""
Dataset and DataLoader for Habitat Classification

Handles loading of onehot/raw mask data for training and validation.
"""

import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class HabitatDataset(Dataset):
    """
    Dataset for loading preprocessed .pt files (supports raw and onehot formats).
    """

    def __init__(self, data_list, config):
        self.data_list = data_list
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        filepath, label, patient_id = self.data_list[idx]

        # Load tensor
        tensor = torch.load(filepath, map_location='cpu').float()

        # Handle onehot format: [4, H, W, 5] -> [20, H, W]
        if self.config.MASK_TYPE == 'onehot':
            if tensor.dim() == 4:
                n_class, h, w, n_slice = tensor.shape
                tensor = tensor.permute(0, 3, 1, 2).reshape(-1, h, w)

        # Handle raw format: [H, W, C] -> [C, H, W]
        else:
            if tensor.dim() == 3 and tensor.shape[-1] == self.config.IN_CHANNELS:
                tensor = tensor.permute(2, 0, 1)

        # Ensure correct number of channels
        if tensor.shape[0] != self.config.IN_CHANNELS:
            if tensor.shape[0] > self.config.IN_CHANNELS:
                tensor = tensor[:self.config.IN_CHANNELS]
            else:
                pad = torch.zeros(
                    self.config.IN_CHANNELS - tensor.shape[0],
                    *tensor.shape[1:]
                )
                tensor = torch.cat([tensor, pad], dim=0)

        # Resize to target size
        if tensor.shape[1] != self.config.IMG_SIZE or tensor.shape[2] != self.config.IMG_SIZE:
            tensor = tensor.unsqueeze(0)
            mode = 'nearest' if self.config.MASK_TYPE == 'onehot' else 'bilinear'
            tensor = F.interpolate(tensor, size=(self.config.IMG_SIZE, self.config.IMG_SIZE), mode=mode)
            tensor = tensor.squeeze(0)

        # Normalize
        if self.config.MASK_TYPE == 'raw':
            tensor = tensor / 3.0
        else:
            tensor = torch.clamp(tensor, 0, 1)

        return tensor, torch.tensor(label, dtype=torch.long), patient_id


def collate_fn(batch):
    """Custom collate function to handle batch data."""
    tensors = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    pids = [item[2] for item in batch]
    return tensors, labels, pids


def create_data_loaders(config):
    """
    Create training and validation data loaders.

    Args:
        config: Configuration object with paths and settings

    Returns:
        train_loader, val_loader, None
    """
    print("\n" + "=" * 50)
    print("Loading data...")
    print(f"Mask type: {config.MASK_TYPE}")
    print("=" * 50)

    # Load labels from Excel
    df_train = pd.read_excel(config.TRAIN_LABEL_EXCEL_PATH)
    df_val = pd.read_excel(config.VAL_LABEL_EXCEL_PATH)

    train_labels = df_train['label'].tolist()
    val_labels = df_val['label'].tolist()

    # Select directory and suffix based on mask type
    if config.MASK_TYPE == 'onehot':
        train_dir = config.TRAIN_MASK_ONEHOT_DIR
        val_dir = config.VAL_MASK_ONEHOT_DIR
        file_suffix = '_onehot.pt'
    else:
        train_dir = config.TRAIN_MASK_RAW_DIR
        val_dir = config.VAL_MASK_RAW_DIR
        file_suffix = '_raw.pt'

    # Get sorted file lists
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(file_suffix)])
    val_files = sorted([f for f in os.listdir(val_dir) if f.endswith(file_suffix)])

    # Pair files with labels
    train_data = []
    for i, filename in enumerate(train_files):
        if i < len(train_labels) and train_labels[i] in [0, 1]:
            filepath = os.path.join(train_dir, filename)
            patient_id = filename.replace(file_suffix, '')
            train_data.append((filepath, train_labels[i], patient_id))

    val_data = []
    for i, filename in enumerate(val_files):
        if i < len(val_labels) and val_labels[i] in [0, 1]:
            filepath = os.path.join(val_dir, filename)
            patient_id = filename.replace(file_suffix, '')
            val_data.append((filepath, val_labels[i], patient_id))

    # Print statistics
    print(f"\nTraining samples: {len(train_data)} (sensitive: {sum(1 for _, l, _ in train_data if l == 0)}, "
          f"resistant: {sum(1 for _, l, _ in train_data if l == 1)})")
    print(f"Validation samples: {len(val_data)} (sensitive: {sum(1 for _, l, _ in val_data if l == 0)}, "
          f"resistant: {sum(1 for _, l, _ in val_data if l == 1)})")

    # Create datasets and loaders
    train_dataset = HabitatDataset(train_data, config)
    val_dataset = HabitatDataset(val_data, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"\nTraining batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"Input channels: {config.IN_CHANNELS}")

    return train_loader, val_loader, None
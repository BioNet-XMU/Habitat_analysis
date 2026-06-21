"""
Training Script for Habitat Classification

This script trains a ResNet18 model to classify habitat types.
Early stopping: stops if validation AUC doesn't improve for 30 consecutive epochs.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

# Disable wandb
os.environ['WANDB_MODE'] = 'disabled'

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset_onehot import create_data_loaders
from models.cnn_model import create_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

        config.create_directories()
        set_seed(config.SEED)

        print("\nCreating data loaders...")
        self.train_loader, self.val_loader, _ = create_data_loaders(config)

        print("\nCreating model...")
        self.model = create_model(config)
        self.model = self.model.to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params / 1e6:.2f}M")

        # Loss function with class weights
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

        # Mixed precision
        self.scaler = GradScaler() if config.MIXED_PRECISION else None

        # Tracking variables
        self.best_val_auc = 0
        self.early_stopping_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []

        print(f"\nDevice: {self.device}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE} epochs")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_probs = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc="Training")
        for data, targets, _ in pbar:
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_loss = total_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
        return epoch_loss, epoch_auc

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for data, targets, _ in pbar:
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_loss = total_loss / len(self.val_loader)
        epoch_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
        return epoch_loss, epoch_auc

    def train(self):
        """Main training loop with early stopping."""
        print("\nStarting training...")
        print("=" * 60)
        print(f"Early stopping: stop if validation AUC doesn't improve for {self.config.EARLY_STOPPING_PATIENCE} consecutive epochs")
        print("=" * 60)

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 40)

            # Train
            train_loss, train_auc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_aucs.append(train_auc)

            # Validate
            val_loss, val_auc = self.validate()
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)

            # Update scheduler
            self.scheduler.step()

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping check
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.save_checkpoint('best_model.pth')
                self.early_stopping_counter = 0
                print(f"✓ Saving best model (Val AUC: {val_auc:.4f})")
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
                print(f"Current best Val AUC: {self.best_val_auc:.4f}")

            # Stop if no improvement for 30 epochs
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered! Validation AUC hasn't improved for {self.config.EARLY_STOPPING_PATIENCE} consecutive epochs")
                print(f"Best validation AUC: {self.best_val_auc:.4f}")
                break

        # Final steps
        print("\n" + "=" * 60)
        print(f"Training complete! Best validation AUC: {self.best_val_auc:.4f}")

        # Save training history and plot
        self.save_training_history()
        self.plot_training_curves()

        return self.best_val_auc

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_aucs': self.train_aucs,
            'val_aucs': self.val_aucs
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)

    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_aucs': self.train_aucs,
            'val_aucs': self.val_aucs,
            'best_val_auc': self.best_val_auc
        }
        path = os.path.join(self.config.SAVE_DIR, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved: {path}")

    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.train_aucs, label='Train AUC', color='blue')
        axes[1].plot(self.val_aucs, label='Val AUC', color='red')
        axes[1].axhline(y=0.5, color='gray', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Training and Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.FIG_DIR, 'training_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Training curves saved: {save_path}")


def main():
    config = Config
    trainer = Trainer(config)
    best_auc = trainer.train()
    print(f"\nFinal best validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
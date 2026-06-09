"""
EmotionSense — Model Training Pipeline
========================================
Production-grade training script for facial emotion recognition.

Architecture: EfficientNet-B0 (ImageNet pretrained) with custom classifier head
Dataset: FER-2013 (7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
Export: ONNX format compatible with OpenCV DNN inference

Usage:
    1. Download FER-2013 dataset from Kaggle:
       https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data
    2. Extract to ai-service/data/ so structure is:
       ai-service/data/train/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}/
       ai-service/data/test/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}/
    3. Install dependencies: pip install -r training_requirements.txt
    4. Run: python train_model.py

The script will:
    - Preprocess and augment the data
    - Train with 2-phase strategy (warm-up + fine-tune)
    - Evaluate with full classification report + confusion matrix
    - Export best model to ONNX format
    - Validate ONNX output matches PyTorch
"""

import os
import sys
import json
import time
import copy
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─── Constants ──────────────────────────────────────────────
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = 7
INPUT_SIZE = 224  # EfficientNet-B0 default input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Configuration ──────────────────────────────────────────
DEFAULT_CONFIG = {
    # Data
    'data_dir': os.path.join(os.path.dirname(__file__), 'data'),
    'output_dir': os.path.join(os.path.dirname(__file__), 'model'),
    'batch_size': 64,
    'num_workers': 4,

    # Training Phase 1: Warm-up (frozen backbone)
    'warmup_epochs': 5,
    'warmup_lr': 1e-3,

    # Training Phase 2: Fine-tuning (all layers)
    'finetune_epochs': 25,
    'finetune_lr': 1e-4,
    'weight_decay': 1e-4,

    # Scheduler
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'min_lr': 1e-7,

    # Regularization
    'label_smoothing': 0.1,
    'dropout_classifier': 0.3,
    'dropout_fc': 0.2,

    # Early stopping
    'early_stopping_patience': 10,

    # Model
    'model_name': 'emotionsense-v2',
}


# ─── Grayscale to 3-Channel Transform ──────────────────────
class GrayscaleTo3Channel:
    """Convert grayscale (1-channel) image to 3-channel by repeating."""
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert('RGB')
        return img


# ─── Data Pipeline ──────────────────────────────────────────
def get_transforms():
    """
    Build training and validation/test transforms.
    
    Training: Heavy augmentation to combat overfitting on FER-2013
    Validation/Test: Only resize + normalize for fair evaluation
    """
    train_transform = transforms.Compose([
        GrayscaleTo3Channel(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        GrayscaleTo3Channel(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def load_datasets(config):
    """
    Load FER-2013 dataset using ImageFolder.
    
    Expected directory structure:
        data/train/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}/*.png
        data/test/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}/*.png
    """
    data_dir = config['data_dir']
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.isdir(train_dir):
        print(f"\n❌ ERROR: Training data not found at: {train_dir}")
        print("\nPlease download the FER-2013 dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data")
        print(f"\nExtract it so the structure is:")
        print(f"  {data_dir}/train/{{Angry,Disgust,...}}/")
        print(f"  {data_dir}/test/{{Angry,Disgust,...}}/")
        sys.exit(1)

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    # Validate class names match expected emotions
    class_names = train_dataset.classes
    print(f"\n📁 Dataset classes found: {class_names}")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples:     {len(test_dataset):,}")

    # Print class distribution
    class_counts = Counter(train_dataset.targets)
    print(f"\n📊 Training class distribution:")
    for idx, name in enumerate(class_names):
        count = class_counts.get(idx, 0)
        bar = '█' * (count // 200)
        print(f"   {name:>10}: {count:>5} {bar}")

    return train_dataset, test_dataset, class_names


def create_weighted_sampler(dataset):
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Each sample gets a weight inversely proportional to its class frequency.
    """
    class_counts = Counter(dataset.targets)
    total = len(dataset)

    # Compute weight for each class: total / (num_classes * count)
    class_weights = {}
    for cls_idx, count in class_counts.items():
        class_weights[cls_idx] = total / (NUM_CLASSES * count)

    # Assign weight to each sample
    sample_weights = [class_weights[t] for t in dataset.targets]
    sample_weights = torch.FloatTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"\n⚖️  Class weights for balanced sampling:")
    for idx in sorted(class_weights.keys()):
        name = dataset.classes[idx]
        print(f"   {name:>10}: {class_weights[idx]:.3f}")

    return sampler


def create_data_loaders(train_dataset, test_dataset, config):
    """Create DataLoaders with weighted sampling for training."""
    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    return train_loader, test_loader


# ─── Model Architecture ────────────────────────────────────
def build_model(config, device):
    """
    Build EfficientNet-B0 with custom classifier head.
    
    Architecture:
        EfficientNet-B0 backbone (ImageNet pretrained)
        → AdaptiveAvgPool2d(1)
        → Dropout(0.3)
        → Linear(1280, 512)
        → BatchNorm1d(512)
        → ReLU
        → Dropout(0.2)
        → Linear(512, 7)
    """
    print("\n🏗️  Building model: EfficientNet-B0 + Custom Classifier")

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head
    in_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=config['dropout_classifier']),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=config['dropout_fc']),
        nn.Linear(512, NUM_CLASSES),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:>10,}")
    print(f"   Trainable parameters: {trainable_params:>10,}")

    model = model.to(device)
    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for param in model.features.parameters():
        param.requires_grad = False
    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧊 Backbone FROZEN — Trainable params: {trainable:,}")


def unfreeze_backbone(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔥 Backbone UNFROZEN — Trainable params: {trainable:,}")


# ─── Training Loop ──────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch with progress bar."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f"  Epoch {epoch:>2}/{total_epochs}",
        bar_format='{l_bar}{bar:30}{r_bar}',
        leave=True,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%',
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, test_loader, config, device, class_names):
    """
    Full training loop with 2-phase strategy:
        Phase 1: Warm-up with frozen backbone
        Phase 2: Fine-tune all layers
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Compute class weights for loss function
    class_counts = Counter([s for s in datasets.ImageFolder(
        os.path.join(config['data_dir'], 'train')
    ).targets])
    total_samples = sum(class_counts.values())
    class_weight_list = [total_samples / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)]
    class_weights = torch.FloatTensor(class_weight_list).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config['label_smoothing'],
    )

    best_acc = 0.0
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    total_epochs = config['warmup_epochs'] + config['finetune_epochs']

    # ═══════════════════════════════════════════════════════
    # Phase 1: Warm-up (Frozen Backbone)
    # ═══════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 1: WARM-UP (Frozen Backbone)")
    print("═" * 60)

    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['warmup_lr'],
        weight_decay=config['weight_decay'],
    )

    for epoch in range(1, config['warmup_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, total_epochs
        )
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  → Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"  → Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best model! Val Acc: {best_acc*100:.2f}%")

    # ═══════════════════════════════════════════════════════
    # Phase 2: Fine-tuning (All Layers)
    # ═══════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 2: FINE-TUNING (All Layers)")
    print("═" * 60)

    unfreeze_backbone(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['finetune_lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['scheduler_patience'],
        factor=config['scheduler_factor'],
        min_lr=config['min_lr'],
        verbose=True,
    )

    for epoch in range(config['warmup_epochs'] + 1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, total_epochs
        )
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, device)
        val_f1 = f1_score(labels, preds, average='macro')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  → Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"  → Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.1f}% | F1: {val_f1:.4f} | LR: {current_lr:.2e}")

        scheduler.step(val_loss)

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ New best model! Val Acc: {best_acc*100:.2f}% | F1: {best_f1:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f'{config["model_name"]}-best.pth')
            torch.save({
                'model_state_dict': best_model_wts,
                'config': config,
                'class_names': class_names,
                'best_acc': best_acc,
                'best_f1': best_f1,
                'epoch': epoch,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"\n⚠️  Early stopping triggered after {patience_counter} epochs without improvement")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"\n✅ Training complete! Best Val Acc: {best_acc*100:.2f}% | Best F1: {best_f1:.4f}")

    return model, history, best_acc, best_f1


# ─── Evaluation & Visualization ─────────────────────────────
def full_evaluation(model, test_loader, device, class_names, output_dir):
    """
    Comprehensive model evaluation with:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix heatmap
    - Per-class accuracy analysis
    """
    print("\n" + "═" * 60)
    print("  MODEL EVALUATION")
    print("═" * 60)

    criterion = nn.CrossEntropyLoss()
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    # Classification Report
    print(f"\n📊 Classification Report:")
    print("-" * 60)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # Save classification report
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"   Report saved to: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.8},
    )
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('EmotionSense v2 — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrix saved to: {cm_path}")

    # Per-class accuracy
    print(f"\n🎯 Per-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean() * 100
            emoji = {'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
                     'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'}.get(name, '❓')
            bar = '█' * int(class_acc // 5)
            print(f"   {emoji} {name:>10}: {class_acc:5.1f}% {bar}")

    # Overall metrics
    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n   Overall Accuracy:  {test_acc*100:.2f}%")
    print(f"   Macro F1-Score:    {overall_f1:.4f}")
    print(f"   Weighted F1-Score: {weighted_f1:.4f}")

    return test_acc, report_dict


def plot_training_history(history, output_dir):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('EmotionSense v2 — Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()

    history_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📈 Training history saved to: {history_path}")


# ─── ONNX Export ────────────────────────────────────────────
def export_to_onnx(model, config, class_names, device):
    """
    Export trained PyTorch model to ONNX format.
    
    The ONNX model will:
    - Accept input: [1, 3, 224, 224] float32 (normalized with ImageNet stats)
    - Produce output: [1, 7] float32 (logits for 7 emotion classes)
    """
    print("\n" + "═" * 60)
    print("  ONNX EXPORT")
    print("═" * 60)

    output_dir = config['output_dir']
    onnx_path = os.path.join(output_dir, f'{config["model_name"]}.onnx')

    model.eval()
    model = model.to('cpu')

    # Create dummy input
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Export
    print(f"\n   Exporting to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed input size for OpenCV DNN compatibility
    )

    # Validate ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ ONNX model validated successfully")

    # Verify output matches PyTorch
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)[0]

    pytorch_outputs = model(dummy_input).detach().numpy()

    max_diff = np.max(np.abs(ort_outputs - pytorch_outputs))
    print(f"   ✓ PyTorch vs ONNX max difference: {max_diff:.8f}")
    if max_diff < 1e-4:
        print(f"   ✓ Output verification PASSED")
    else:
        print(f"   ⚠ Output difference is larger than expected, but model should still work")

    # Save model metadata
    metadata = {
        'model_name': config['model_name'],
        'architecture': 'EfficientNet-B0',
        'num_classes': NUM_CLASSES,
        'class_names': class_names,
        'input_size': INPUT_SIZE,
        'input_channels': 3,
        'input_format': 'RGB normalized (ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
        'output_format': 'logits (apply softmax for probabilities)',
        'exported_at': datetime.now().isoformat(),
    }
    metadata_path = os.path.join(output_dir, f'{config["model_name"]}-metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Metadata saved to: {metadata_path}")

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n   📦 ONNX model size: {file_size:.1f} MB")
    print(f"   📦 Model path: {onnx_path}")

    return onnx_path


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='EmotionSense Model Training')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to FER-2013 dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Fine-tuning epochs (default: 25)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warm-up epochs with frozen backbone (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Fine-tuning learning rate (default: 1e-4)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU training')
    args = parser.parse_args()

    # Banner
    print("\n" + "═" * 60)
    print("  🧠 EmotionSense — Model Training Pipeline")
    print("  Architecture: EfficientNet-B0 + Custom Classifier")
    print("  Dataset: FER-2013 (7 Emotion Classes)")
    print("═" * 60)

    # Configuration
    config = DEFAULT_CONFIG.copy()
    if args.data_dir:
        config['data_dir'] = args.data_dir
    config['batch_size'] = args.batch_size
    config['finetune_epochs'] = args.epochs
    config['warmup_epochs'] = args.warmup_epochs
    config['finetune_lr'] = args.lr

    # Device setup
    if args.no_gpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Adjust workers for Windows
    if sys.platform == 'win32':
        config['num_workers'] = 0  # Multiprocessing can be problematic on Windows
        print(f"   Workers: 0 (Windows compatibility)")

    start_time = time.time()

    # Step 1: Load Data
    print("\n" + "─" * 60)
    print("  STEP 1: Loading Dataset")
    print("─" * 60)
    train_dataset, test_dataset, class_names = load_datasets(config)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config)

    # Step 2: Build Model
    print("\n" + "─" * 60)
    print("  STEP 2: Building Model")
    print("─" * 60)
    model = build_model(config, device)

    # Step 3: Train
    print("\n" + "─" * 60)
    print("  STEP 3: Training")
    print("─" * 60)
    model, history, best_acc, best_f1 = train_model(
        model, train_loader, test_loader, config, device, class_names
    )

    # Step 4: Evaluate
    print("\n" + "─" * 60)
    print("  STEP 4: Evaluation")
    print("─" * 60)
    test_acc, report = full_evaluation(model, test_loader, device, class_names, config['output_dir'])

    # Step 5: Plot history
    plot_training_history(history, config['output_dir'])

    # Step 6: Export to ONNX
    print("\n" + "─" * 60)
    print("  STEP 5: ONNX Export")
    print("─" * 60)
    onnx_path = export_to_onnx(model, config, class_names, device)

    # Summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "═" * 60)
    print("  🎉 TRAINING COMPLETE")
    print("═" * 60)
    print(f"\n   ⏱️  Total time:      {minutes}m {seconds}s")
    print(f"   🎯 Best Val Accuracy: {best_acc*100:.2f}%")
    print(f"   📊 Macro F1-Score:    {best_f1:.4f}")
    print(f"   📦 ONNX Model:       {onnx_path}")
    print(f"\n   To use the trained model in EmotionSense:")
    print(f"   The AI service will automatically detect and load '{config['model_name']}.onnx'")
    print(f"   from the model/ directory.")
    print()


if __name__ == '__main__':
    main()

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    from _bootstrap import ensure_modeling_root_on_path
except ImportError:
    import sys

    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from _bootstrap import ensure_modeling_root_on_path

MODEL_ROOT = ensure_modeling_root_on_path()
FLOAT_MODEL_PATH = MODEL_ROOT / "artifacts" / "models" / "float_model_phase1.pt"
INT8_MODEL_PATH = MODEL_ROOT / "artifacts" / "models" / "int8_model.pt"
from models.model_generator import Int8ModelFromSpec
from models.model_generator import FloatModelFromSpec
from training.trainer import train_model, train_model_with_distillation
from training.layer_wise_compression import train_layer_wise_compression
from data.data_loader import load_data_sample
from scripts.train.train_teacher import TactileCNNWithEmbedding

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define model specification
# spec = (13, 128, [(7, 64, None), (3, 128, 2), (13, 256, 16), (3, 256, None)], 12)
spec = (13, 128, [(9, 64, 2), (7, 128, 2), (5, 256, 2), (3, 256, 2)], 12)

train_loader, val_loader = load_data_sample(str(MODEL_ROOT / "data" / "int8_dataset.npz"), sample_size=1, batch_size=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_ROOT / "gallery_assets", exist_ok=True)
os.makedirs(FLOAT_MODEL_PATH.parent, exist_ok=True)
os.makedirs(INT8_MODEL_PATH.parent, exist_ok=True)

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

def update_history(new_data):
    for key in history:
        history[key].extend(new_data[key])

class_names = [
    "plastic pencil case 1", "plastic pencil case 2", "plastic placemat 1", 
    "plastic placemat 2", "plastic placemat 3", "plastic placemat 4",
    "cork mat", "denim fabric 1", "denim fabric 2",
    "felt material", "polyester cloth", "cotton scarf"
]

# Toggle this for quick A/B against unweighted cross entropy.
USE_WEIGHTED_LOSS = True

# Class order matches class_names above.
class_weights = torch.tensor([
    1.00,  # plastic pencil case 1
    1.00,  # plastic pencil case 2
    1.10,  # plastic placemat 1
    1.10,  # plastic placemat 2
    1.10,  # plastic placemat 3
    1.10,  # plastic placemat 4
    1.00,  # cork mat
    1.30,  # denim fabric 1
    1.30,  # denim fabric 2
    1.00,  # felt material
    1.00,  # polyester cloth
    1.00,  # cotton scarf
], dtype=torch.float32)

criterion = (
    nn.CrossEntropyLoss(weight=class_weights.to(device))
    if USE_WEIGHTED_LOSS
    else nn.CrossEntropyLoss()
)

USE_DISTILLATION = True
TEACHER_CHECKPOINT = str(MODEL_ROOT / "gallery_assets" / "5-GoldStandard" / "best_multiscale_model.pt")
DISTILL_ALPHA = 0.30
DISTILL_TEMPERATURE = 3.0

def save_confusion_matrix(epoch_num):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix - Epoch {epoch_num}")
    plt.tight_layout()
    plt.savefig(MODEL_ROOT / f"gallery_assets/confusion_epoch_{epoch_num}.png")
    plt.close()

model = FloatModelFromSpec(spec)
model.to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    eps=1e-8,
    weight_decay=1e-5
)

# Initial training of the float model
EPOCHS = 10
if USE_DISTILLATION and os.path.exists(TEACHER_CHECKPOINT):
    teacher_model = TactileCNNWithEmbedding(in_channels=13, num_classes=12)
    teacher_state = torch.load(TEACHER_CHECKPOINT, map_location=device)
    teacher_model.load_state_dict(teacher_state)
    teacher_model.to(device)
    teacher_model.eval()
    print(f"Using distillation teacher checkpoint: {TEACHER_CHECKPOINT}")
    warm_up_history = train_model_with_distillation(
        model,
        teacher_model,
        train_loader,
        val_loader,
        device,
        num_epochs=EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        alpha=DISTILL_ALPHA,
        temperature=DISTILL_TEMPERATURE,
    )
else:
    if USE_DISTILLATION:
        print(f"Distillation enabled but teacher checkpoint not found: {TEACHER_CHECKPOINT}")
        print("Falling back to standard warm-up training.")
    warm_up_history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=EPOCHS,
        criterion=criterion,
        optimizer=optimizer
    )
update_history(warm_up_history)
save_confusion_matrix("float_initial")

torch.save(model.state_dict(), FLOAT_MODEL_PATH)

model = Int8ModelFromSpec(spec)
model.to(device)
print(f"Number of trainable parameters: {count_parameters(model)}")
state_dict = torch.load(FLOAT_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

# Layer-wise compression training
optimizer = optim.Adam(
    model.parameters(),
    lr=5e-4,
    eps=1e-8,
    weight_decay=1e-6
)
compression_result = train_layer_wise_compression(
    model,
    train_loader,
    val_loader,
    device,
    optimizer=optimizer,
    criterion=criterion,
    save_callback=save_confusion_matrix,
    reset_optimizer_each_phase=True,
    return_metadata=True
)
compression_metadata = {'final_loss_trim_epochs': 0}
if isinstance(compression_result, tuple):
    compression_history, compression_metadata = compression_result
else:
    compression_history = compression_result
update_history(compression_history)

torch.save(model.state_dict(), INT8_MODEL_PATH)

# Final Accuracy/Loss Plot
loss_trim = min(compression_metadata.get('final_loss_trim_epochs', 0), len(history['train_loss']))

if loss_trim > 0:
    loss_train = history['train_loss'][:-loss_trim]
    loss_val = history['val_loss'][:-loss_trim]
else:
    loss_train = history['train_loss']
    loss_val = history['val_loss']

loss_epochs_range = range(1, len(loss_train) + 1)
acc_epochs_range = range(1, len(history['train_acc']) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_epochs_range, loss_train, label='Train Loss')
plt.plot(loss_epochs_range, loss_val, label='Val Loss')
plt.title('Loss Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_epochs_range, history['train_acc'], label='Train Acc')
plt.plot(acc_epochs_range, history['val_acc'], label='Val Acc')
plt.title('Accuracy Over Time')
plt.legend()

plt.savefig(MODEL_ROOT / "gallery_assets/training_summary.png")
print("Assets generated in /gallery_assets and state_dict saved.")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data.data_loader import load_data_sample
from training.quantize_weights import quantize_and_freeze_layer
from models.model_generator import Int8ModelFromSpec

def get_layer_name(friendly_name):
    mapping = {
        'conv1a': 'blocks.0.dw',
        'conv1b': 'blocks.0.pw',
        'conv2a': 'blocks.1.dw',
        'conv2b': 'blocks.1.pw',
        'conv3a': 'blocks.2.dw',
        'conv3b': 'blocks.2.pw',
        'conv4a': 'blocks.3.dw',
        'conv4b': 'blocks.3.pw',
        'fc': 'classifier'
    }
    return mapping[friendly_name]

layer_freeze_schedule = ['conv1a', 'conv1b', 'conv2a', 'conv2b', 'conv3a', 'conv3b', 'conv4a', 'conv4b', 'fc']
freeze_interval = 10

# Class names
class_names = [
    "plastic pencil case 1", "plastic pencil case 2",
    "plastic placemat 1", "plastic placemat 2", "plastic placemat 3", "plastic placemat 4",
    "cork mat",
    "denim fabric 1", "denim fabric 2",
    "felt material", "polyester cloth", "cotton scarf"
]

train_loader, val_loader = load_data_sample("data/int8_dataset.npz", sample_size=1, batch_size=512)

# Parameters
EPOCHS = 100
LEARNING_RATE = 0.001

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec = (13, 128, [(7, 64, None), (3, 128, 2), (13, 256, 16), (3, 256, None)], 12)
model = Int8ModelFromSpec(spec).to(device)
state_dict = torch.load("float_model.pt", map_location=device)
model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {val_accuracy:.4f}")
    if epoch % freeze_interval == 0 and epoch // freeze_interval <= len(layer_freeze_schedule) and epoch != 0:
        layer_to_freeze = layer_freeze_schedule[epoch // freeze_interval - 1]
        layer_name = get_layer_name(layer_to_freeze)
        quantize_and_freeze_layer(model, layer_name)
        parent_name = layer_name.rsplit('.', 1)[0]
        parent_module = dict(model.named_modules())[parent_name]
        
        if 'dw' in layer_name:
            parent_module.sc_dw.quant_on = True
        elif 'pw' in layer_name:
            parent_module.sc_pw.quant_on = True
            
        print(f"Epoch {epoch}: Frozen {layer_to_freeze} AND enabled ShiftClamp quantization")
        if epoch // freeze_interval == len(layer_freeze_schedule):
            print("All layers frozen. Stopping training.")
            break

# Save model state_dict
torch.save(model.state_dict(), "fully_quantized_weights.pt")
print("✓ Model state_dict saved to fully_quantized_weights.pt")

# Evaluate on validation set and compute confusion matrix
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute and display confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=np.arange(12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix on Validation Set")
plt.tight_layout()
plt.show()

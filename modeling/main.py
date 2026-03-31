import torch
import torch.nn as nn
import torch.optim as optim
from models.model_generator import Int8ModelFromSpec
from models.model_generator import FloatModelFromSpec
from training.trainer import train_quantized_pipeline
from training.trainer import train_model
from data.data_loader import load_data_sample

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define model specification
spec = (13, 128, [(7, 64, None), (3, 128, 2), (13, 256, 16), (3, 256, None)], 12)
train_loader, val_loader = load_data_sample("data/int8_dataset.npz", sample_size=1, batch_size=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
model = FloatModelFromSpec(spec)
model.to(device)

lr = 1e-2
optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9
)

# Initial training of the float model
EPOCHS = 10
train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=EPOCHS,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer
)

torch.save(model.state_dict(), "int8_model.pt")
"""


model = Int8ModelFromSpec(spec)
model.to(device)
print(f"Number of trainable parameters: {count_parameters(model)}")
#state_dict = torch.load("int8_model.pt", map_location=device)
#model.load_state_dict(state_dict)

"""
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9
)

phases = ["scale", "freeze"]

for phase in phases:

    train_quantized_pipeline(
        model,
        train_loader,
        val_loader,
        device,
        optimizer=optimizer,
        mode=phase
    )

torch.save(model.state_dict(), "int8_model.pt")
"""
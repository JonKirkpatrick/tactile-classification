import torch
import copy
from utils.model_utils import ShiftClamp

def train_model(model, train_loader, val_loader, device, num_epochs, criterion, optimizer, scheduler):
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        weight_deltas = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_deltas[name] = {'old': param.data.clone(), 'delta_sum': 0.0, 'count': 0}


        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_max = param.grad.abs().max().item()
                    if not torch.isfinite(param.grad).all():
                        print(f"❗ Non-finite gradient in {name}")
                    print(f"{name}: grad max = {grad_max:.4e}")
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    delta = (param.data - weight_deltas[name]['old']).abs().mean().item()
                    weight_deltas[name]['delta_sum'] += delta
                    weight_deltas[name]['count'] += 1
                    weight_deltas[name]['old'] = param.data.clone()  # update for next batch
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print("Average weight deltas this epoch:")
        for name, stats in weight_deltas.items():
            if stats['count'] > 0:
                avg_delta = stats['delta_sum'] / stats['count']
                print(f"  {name}: {avg_delta:.6f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_samples += inputs.size(0)

        val_loss /= val_samples
        val_acc = val_corrects / val_samples
        if scheduler:
            scheduler.step(val_loss)

        print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if epoch_loss < 0.04:
            print("Early stopping due to low validation loss.")
            break

    print(f'\nBest val Acc: {best_val_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model
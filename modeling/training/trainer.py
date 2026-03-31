import torch
from training.quantize_weights import quantize_and_freeze_layer, scale_weights_in_place, quantize_weights_in_place


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    criterion,
    optimizer
):    

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}"
        )

def get_quantization_schedule(model):
    """
    Returns a list of tuples: (layer_name, type, module_ref)
    Discovers all DW and PW layers within QuantBlocks dynamically.
    """
    schedule = []
    
    # 1. Iterate through the ModuleList blocks
    for i, block in enumerate(model.blocks):
        # We know every QuantBlock has a 'dw' and a 'pw'
        schedule.append((f"blocks.{i}.dw", "dw", block))
        schedule.append((f"blocks.{i}.pw", "pw", block))
        
    # 2. Add the final classifier
    schedule.append(("classifier", "fc", model))
    
    return schedule

def set_quantization_state(layer_info, enabled=True):
    """
    layer_info is the tuple from get_quantization_schedule
    """
    path, layer_type, parent_module = layer_info
    
    if layer_type == "dw":
        parent_module.sc_dw.quant_on = enabled
    elif layer_type == "pw":
        parent_module.sc_pw.quant_on = enabled
    """
    elif layer_type == "fc":
        # Final global shift before the classifier
        # In your builder, this is self.final_shift
        parent_module.final_shift.quant_on = enabled
    """

def train_quantized_pipeline(model, train_loader, val_loader, device, optimizer, mode='freeze'):
    schedule = get_quantization_schedule(model)
    criterion = torch.nn.CrossEntropyLoss()
    
    # We maintain the global layer_scales across the passes
    if not hasattr(model, 'layer_scales'):
        model.layer_scales = {}

    for i, layer_info in enumerate(schedule):
        layer_path, layer_type, _ = layer_info
        
        # 1. Inject Noise (Common to all modes except final freeze)
        set_quantization_state(layer_info, enabled=True)
        print(f"\n>>> Training with Quantization Noise on {layer_path}...")
        
        train_model(model, train_loader, val_loader, device, num_epochs=5, 
                    criterion=criterion, optimizer=optimizer)

        # 2. Handle Weights based on Pass Mode
        if mode == 'scale':
            # Phase 1: Maximize dynamic range, but keep float precision
            scale_weights_in_place(model, layer_path, model.layer_scales)
            print(f"\n>>> Scaled: {layer_path}")
            
        elif mode == 'quantize':
            # Phase 2: Snap to integer grid, but keep gradients live
            quantize_weights_in_place(model, layer_path, model.layer_scales)
            print(f"\n>>> Quantized (Live): {layer_path}")
            
        elif mode == 'freeze':
            # Phase 3: Snap and lock
            quantize_and_freeze_layer(model, layer_path)
            print(f"\n>>> Frozen: {layer_path}")

        if i == len(schedule) - 1:
            print(f"\n✓ Finished working {layer_path}. No parameters left to train.")
            break            

        # 3. Train the "Current State"
        train_model(model, train_loader, val_loader, device, num_epochs=5, 
                    criterion=criterion, optimizer=optimizer)

    print("\n✓ Grand Unified Training Complete.")
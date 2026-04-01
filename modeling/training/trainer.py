
import torch
from training.quantize_weights import quantize_and_freeze_layer, scale_weights_in_place, quantize_weights_in_place


def train_model(model, train_loader, val_loader, device, num_epochs, criterion, optimizer):    
    local_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # ---- Validation ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()

        # Store metrics
        local_history['train_loss'].append(running_loss / total)
        local_history['train_acc'].append(correct / total)
        local_history['val_loss'].append(v_loss / v_total)
        local_history['val_acc'].append(v_correct / v_total)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {v_correct/v_total:.4f}")

    return local_history # Return the data to the caller


def train_model_with_distillation(
    model,
    teacher_model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    criterion,
    optimizer,
    alpha=0.3,
    temperature=3.0,
):
    local_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    kl_div = torch.nn.KLDivLoss(reduction='batchmean')

    teacher_model.eval()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            student_logits = model(inputs)
            with torch.no_grad():
                teacher_out = teacher_model(inputs)
                # Support teacher outputs that are either logits or (logits, embedding).
                teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out

            ce_loss = criterion(student_logits, targets)
            student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
            distill_loss = kl_div(student_log_probs, teacher_probs) * (temperature ** 2)
            loss = (1.0 - alpha) * ce_loss + alpha * distill_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()

        local_history['train_loss'].append(running_loss / total)
        local_history['train_acc'].append(correct / total)
        local_history['val_loss'].append(v_loss / v_total)
        local_history['val_acc'].append(v_correct / v_total)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Val Acc: {v_correct/v_total:.4f} | Distill alpha={alpha:.2f}, T={temperature:.2f}"
        )

    return local_history

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

def train_quantized_pipeline(model, train_loader, val_loader, device, optimizer, mode='freeze', save_callback=None):
    schedule = get_quantization_schedule(model)
    criterion = torch.nn.CrossEntropyLoss()
    local_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    def update_history(new_data):
        for key in local_history:
            local_history[key].extend(new_data[key])
    
    # We maintain the global layer_scales across the passes
    if not hasattr(model, 'layer_scales'):
        model.layer_scales = {}

    for i, layer_info in enumerate(schedule):
        layer_path, layer_type, _ = layer_info
        
        # 1. Inject Noise (Common to all modes except final freeze)
        set_quantization_state(layer_info, enabled=True)
        print(f"\n>>> Training with Quantization Noise on {layer_path}...")
        
        quant_history = train_model(model, train_loader, val_loader, device, num_epochs=5, 
                    criterion=criterion, optimizer=optimizer)
        update_history(quant_history)

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
        phase_history = train_model(model, train_loader, val_loader, device, num_epochs=5, 
                    criterion=criterion, optimizer=optimizer)
        update_history(phase_history)
        if save_callback:
            save_callback(f"{mode}_{layer_path.replace('.', '_')}")

    print("\n✓ Grand Unified Training Complete.")
    return local_history
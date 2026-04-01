import torch
from training.quantize_weights import quantize_and_freeze_layer, scale_weights_in_place
from training.trainer import train_model, get_quantization_schedule, set_quantization_state


def _build_optimizer_from_template(base_optimizer, model):
    """Rebuild optimizer from template defaults using only trainable params."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return None
    return type(base_optimizer)(trainable_params, **base_optimizer.defaults)


def train_layer_wise_compression(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    criterion=None,
    save_callback=None,
    reset_optimizer_each_phase=False,
    phase_epochs=5,
    recovery_epochs=5,
    return_metadata=False,
):
    
    schedule = get_quantization_schedule(model)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    local_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    metadata = {'final_loss_trim_epochs': 0}

    def update_local_history(new_data):
        for key in local_history:
            local_history[key].extend(new_data[key])

    for i, layer_info in enumerate(schedule):
        layer_path, _, _ = layer_info
        print(f"\n>>> COMPRESSING LAYER: {layer_path}")

        if reset_optimizer_each_phase:
            current_optimizer = _build_optimizer_from_template(optimizer, model)
            if current_optimizer is None:
                print(">>> No trainable parameters remain. Stopping compression training.")
                break
        else:
            current_optimizer = optimizer

        set_quantization_state(layer_info, enabled=True)
        scale_weights_in_place(model, layer_path, model.layer_scales)
        
        h1 = train_model(model, train_loader, val_loader, device, phase_epochs, criterion, current_optimizer)
        update_local_history(h1)

        # The final schedule layer (classifier) often has inflated loss due to scaling/quantization dynamics.
        # Record exactly how many epochs were run so plotting can trim this artifact automatically.
        if i == len(schedule) - 1:
            metadata['final_loss_trim_epochs'] = len(h1['train_loss'])

        quantize_and_freeze_layer(model, layer_path)
        print(f">>> Frozen and Locked: {layer_path}")

        if any(p.requires_grad for p in model.parameters()):
            if reset_optimizer_each_phase:
                current_optimizer = _build_optimizer_from_template(optimizer, model)
                if current_optimizer is None:
                    print(">>> No trainable parameters remain after freezing.")
                    break
            h2 = train_model(model, train_loader, val_loader, device, recovery_epochs, criterion, current_optimizer)
            update_local_history(h2)
        else:
            print(">>> All layers frozen. Skipping final recovery training.")

        if save_callback:
            save_callback(f"frozen_{layer_path.replace('.', '_')}")

    if return_metadata:
        return local_history, metadata
    return local_history
import torch

def quantize_and_freeze_layer(model, layer_name):
    modules = dict(model.named_modules())
    module = modules[layer_name]

    scale = None  # To hold the scale computed from weights

    # Quantize and freeze weights and bias
    for name, param in module.named_parameters(recurse=False):
        with torch.no_grad():
            if 'weight' in name and param.dtype == torch.float32:
                max_val = torch.max(torch.abs(param))
                scale = 1.0 if max_val == 0 else 127.0 / max_val
                quantized = torch.floor(param * scale).clamp(-128, 127)
                param.copy_(quantized)
                param.requires_grad = False

            elif 'bias' in name and param.dtype == torch.float32:
                if scale is None:
                    raise ValueError("Bias encountered before weight; scale is undefined.")
                quantized = torch.floor(param * scale).clamp(-128, 127)
                param.copy_(quantized)
                param.requires_grad = False


def quantize_weights_in_place(model, layer_name, layer_scales):
    modules = dict(model.named_modules())
    module = modules[layer_name]

    scale = None  # Local scale computed from weights

    for name, param in module.named_parameters(recurse=False):
        with torch.no_grad():
            if 'weight' in name and param.dtype == torch.float32:
                max_val = torch.max(torch.abs(param))
                scale = 1.0 if max_val == 0 else 127.0 / max_val
                previous_scale = layer_scales.get(layer_name, 1.0)
                updated_scale = previous_scale * scale
                layer_scales[layer_name] = updated_scale
                quantized = torch.floor(param * scale).clamp(-128, 127)
                param.copy_(quantized)
                # Do not freeze

            elif 'bias' in name and param.dtype == torch.float32:
                if scale is None:
                    raise ValueError("Bias encountered before weight; scale is undefined.")
                quantized = torch.floor(param * scale).clamp(-128, 127)
                param.copy_(quantized)

def scale_weights_in_place(model, layer_name, layer_scales):
    modules = dict(model.named_modules())
    module = modules[layer_name]

    scale = None  # Local scale computed from weights

    for name, param in module.named_parameters(recurse=False):
        with torch.no_grad():
            if 'weight' in name and param.dtype == torch.float32:
                max_val = torch.max(torch.abs(param))
                scale = 1.0 if max_val == 0 else 127.0 / max_val
                previous_scale = layer_scales.get(layer_name, 1.0)
                updated_scale = previous_scale * scale
                layer_scales[layer_name] = updated_scale
                quantized = param * scale
                param.copy_(quantized)
                # Do not freeze

            elif 'bias' in name and param.dtype == torch.float32:
                if scale is None:
                    raise ValueError("Bias encountered before weight; scale is undefined.")
                quantized = param * scale
                param.copy_(quantized)
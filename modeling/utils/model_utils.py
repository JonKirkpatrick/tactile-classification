import torch
import torch.nn as nn

class ShiftClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quant_on):
        max_val = x.abs().max()
        if max_val == 0:
            scale_exp = torch.tensor(0.0, device=x.device)
            scale = torch.tensor(1.0, device=x.device)
        else:
            scale_exp = torch.ceil(torch.log2(max_val / 127.0))
            scale = torch.pow(2.0, scale_exp)

        ctx.scale = scale  # Save scale for backward

        x_scaled = x / scale
        if quant_on:
            x_scaled = torch.floor(x_scaled)
        clamped = torch.clamp(x_scaled, -128, 127)

        return clamped

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        grad_x = grad_output / scale
        return grad_x, None

class ShiftClamp(nn.Module):
    def __init__(self, quant_on=False):
        super().__init__()
        self.quant_on = quant_on

    def forward(self, x):
        return ShiftClampFunction.apply(x, self.quant_on)
    
def initialize_weights(module, scale_dict={}, quantize=False):
    """
    Rescales existing weights of a module to saturate int8 range [-128, 127].
    
    Parameters:
    - module: torch.nn.Module whose parameters will be rescaled.
    - scale_dict: dict to store scale factors keyed by layer prefix (e.g., 'conv1')
    - quantize: bool, if True, values are rounded to simulate int8.
    """
    for name, param in module.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                max_val = param.data.abs().max()
                if max_val > 0:
                    scale = 127.0 / max_val
                    param.data.mul_(scale)
                if quantize:
                    param.data.round_()
                
                # Strip off '.weight', '.bias', or any other suffix
                prefix = name.rsplit('.', 1)[0]
                scale_dict[prefix] = scale
                
                print(f"✓ Initialized {name} with scale {scale:.4f}, quantize={quantize}")

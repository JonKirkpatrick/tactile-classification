import torch.nn as nn
from utils.model_utils import ShiftClamp

class QuantBlock(nn.Module):
    def __init__(self, in_channels, dw_kernel, out_channels, pool_kernel=None, block_index=None, layer_scales=None):
        super().__init__()
        padding = (dw_kernel - 1) // 2

        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=dw_kernel,
                      padding=padding, groups=in_channels, bias=False)
        self.sc_dw = ShiftClamp(quant_on=False)
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sc_pw = ShiftClamp(quant_on=False)
        self.relu = nn.ReLU()
        if pool_kernel:
            self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        else:
            self.maxpool = None

    def forward(self, x):
        x = self.dw(x)
        x = self.sc_dw(x)
        x = self.pw(x)
        x = self.sc_pw(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.maxpool(x)
        return x

class Int8ModelFromSpec(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        c_in, L_0, block_specs, num_classes = spec
        self.layer_scales = {}
        self.permute_input = True
        self.blocks = nn.ModuleList()
        for block_index, (k, c_out, p) in enumerate(block_specs):
            block = QuantBlock(
                c_in, k, c_out, pool_kernel=p,
                block_index = block_index,
                layer_scales = self.layer_scales
            )
            self.blocks.append(block)
            c_in = c_out

        self.final_shift = ShiftClamp( quant_on=False )
        self.classifier = nn.Linear(c_in, num_classes, bias=True)

    def __repr__(self):
        c_in, L_0, block_specs, num_classes = self.spec
        block_strs = ', '.join(f"({k}, {c}, {p})" for k, c, p in block_specs)
        return (
            f"{self.__class__.__name__}(\n"
            f"  C_in={c_in}, L_0={L_0},\n"
            f"  Blocks=[{block_strs}],\n"
            f"  NUM_CLASSES={num_classes}\n"
            f")"
        )

    def get_arch_spec(self):
        return self.spec
    
    def forward(self, x):
        if self.permute_input:
            x = x.permute(0, 2, 1)  # (B, L, C) to (B, C, L)

        for block in self.blocks:
            x = block(x)

        x = x.sum(dim=2, keepdim=False)
        x = self.final_shift(x)
        x = self.classifier(x)
        return x
    
class FloatBlock(nn.Module):
    def __init__(self, in_channels, dw_kernel, out_channels, pool_kernel=None, block_index=None, layer_scales=None):
        super().__init__()
        padding = (dw_kernel - 1) // 2

        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=dw_kernel,
                      padding=padding, groups=in_channels, bias=False)
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        if pool_kernel:
            self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        else:
            self.maxpool = None

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.maxpool(x)
        return x

class FloatModelFromSpec(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        c_in, L_0, block_specs, num_classes = spec
        self.layer_scales = {}
        self.permute_input = True
        self.blocks = nn.ModuleList()
        for block_index, (k, c_out, p) in enumerate(block_specs):
            block = FloatBlock(
                c_in, k, c_out, pool_kernel=p,
                block_index = block_index,
                layer_scales = self.layer_scales
            )
            self.blocks.append(block)
            c_in = c_out

        self.classifier = nn.Linear(c_in, num_classes, bias=True)

    def __repr__(self):
        c_in, L_0, block_specs, num_classes = self.spec
        block_strs = ', '.join(f"({k}, {c}, {p})" for k, c, p in block_specs)
        return (
            f"{self.__class__.__name__}(\n"
            f"  C_in={c_in}, L_0={L_0},\n"
            f"  Blocks=[{block_strs}],\n"
            f"  NUM_CLASSES={num_classes}\n"
            f")"
        )

    def get_arch_spec(self):
        return self.spec
    
    def forward(self, x):
        if self.permute_input:
            x = x.permute(0, 2, 1)  # (B, L, C) to (B, C, L)

        for block in self.blocks:
            x = block(x)

        x = x.sum(dim=2, keepdim=False)
        x = self.classifier(x)
        return x
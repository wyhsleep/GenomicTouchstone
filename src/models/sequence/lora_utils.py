import torch
import torch.nn as nn

lora_target_modules_mapping = {
    'dnabert': ['query','value'],
    'dnabert2': ['query','value'],
    'dnaberts': ['query','value'],
    'NT': ['query','value'],
    'DNAHLM': ['c_attn'],
    'genalm': ['query','value'],
    'grover': ['query','value'],
    'caduceus': ['x_proj','in_proj','out_proj','dt_proj'],
    'rnafm': ['query','value'],
    'rnabert': ['query','value'],
    'rnaernie': ['query','value'],
    'splicebert': ['query','value'],
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())





class LoRAConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank=4, stride=1, padding=0, dilation=1, lora_alpha=32, lora_dropout=0.1):
        super(LoRAConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.rank = rank
        self.lora_alpha = lora_alpha
        # Define LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels, kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        # LoRA dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        # Scaling factor
        self.scaling = self.lora_alpha / self.rank

    def forward(self, x):
        conv_output = self.conv(x)
       
        lora_output = self.lora_dropout(x) 
        lora_output = nn.functional.conv1d(lora_output, self.lora_A, stride=self.conv.stride, padding=self.conv.padding)
        lora_output = nn.functional.conv1d(lora_output, self.lora_B, stride=1, padding=0)
        lora_output = lora_output * self.scaling  # Scale LoRA output
        return conv_output + lora_output
    
def replace_conv1d_with_lora(model, rank=4, lora_alpha=32, lora_dropout=0.1):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv1d):
            
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding
            dilation = module.dilation[0]

            lora_layer = LoRAConv1D(
                in_channels, out_channels, kernel_size, rank, stride=stride,
                padding=padding, dilation=dilation, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            setattr(model, name, lora_layer)  
        elif len(list(module.children())) > 0:
            # Recursively replace sub-modules
            replace_conv1d_with_lora(module, rank, lora_alpha, lora_dropout)


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True  
        else:
            param.requires_grad = False  

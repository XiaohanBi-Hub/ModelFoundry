import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import Union, List, Dict, Any, cast

cfgs: Dict[str, List[Union[str, int]]] = {
    "resnet20": [3]*3, "resnet32": [5]*3, "resnet44": [7]*3, "resnet56": [9]*3
}

# model_name[layer_idx(0～2)][block_idx(0/0～2)][conv_idx(0～2)]
resnet_conv_configs = {
    "resnet20": [[[(3, 16)]], [[(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)]], [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]], [[(32, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)]]],
    "resnet32": [[[(3, 16)]], [[(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)]], [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]], [[(32, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)]]],
    "resnet44": [[[(3, 16)]], [[(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)]], [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]], [[(32, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)]]],
    "resnet56": [[[(3, 16)]], [[(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)], [(16, 16), (16, 16)]], [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]], [[(32, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)], [(64, 64), (64, 64)]]]
}

cfgs: Dict[str, List[Union[str, int]]] = {
    "resnet20": [3]*3, "resnet32": [5]*3, "resnet44": [7]*3, "resnet56": [9]*3
}


def init_param(module_init_type, size):
    if module_init_type == 'random':
        param = torch.randn(size).to(device)
    elif module_init_type == 'ones':
        param = torch.ones(size).to(device)
    elif module_init_type == 'zeros':
        param = torch.zeros(size).to(device)
    else:
        raise ValueError
    return param

def get_res_target_name(layer_idx, block_idx, layer_configs):
    # if layer_idx == 0 -> block_idx_max=0
    if layer_idx <= 0 or block_idx < 0:
        return ValueError
    if block_idx > 0:
        return f'layer{layer_idx}_{block_idx-1}_conv_1'
    else:   # block_idx == 0
        if layer_idx == 1:
            return 'layer0_0_conv_0'
        else:
            block_idx_max = layer_configs[layer_idx-1]
            return f'layer{layer_idx - 1}_{block_idx_max-1}_conv_1'


class GradSplitter(nn.Module):
    def __init__(self, model, module_init_type):
        super(GradSplitter, self).__init__()
        self.model = model
        self.n_class = model.num_classes
        self.sign = MySign.apply
        for p in model.parameters():
            p.requires_grad = False

        self.model_name = self.model.model_name
        self.conv_configs = model.conv_configs #？
        self.module_params = []
        self.init_modules(module_init_type)

    def init_modules(self, module_init_type):
        for module_idx in range(self.n_class):
            for layer_idx, layer_cfg in enumerate(self.conv_configs):   # layer_cfg = [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]]
                for block_idx, block_cfg in enumerate(layer_cfg):       # block_cfg = [(16, 32), (32, 32), (16, 32)]
                    for conv_idx, conv_cfg in enumerate(block_cfg):
                        # print(conv_cfg)                               # conv_cfg = (16, 32)
                        param = init_param(module_init_type, cast(int, conv_cfg[1]))
                        # module_*_layer*_*_conv_*
                        setattr(self, f'module_{module_idx}_layer{layer_idx}_{block_idx}_conv_{conv_idx}', nn.Parameter(param))
                        # print(f'module_{module_idx}_layer{layer_idx}_{block_idx}_conv_{conv_idx}')

            # multi-layer head 10 -> 10 -> 1
            param = nn.Sequential(
                nn.Linear(self.n_class, 10),
                nn.ReLU(),
                nn.Linear(10, 1), 
            ).to(device)

            # param = nn.Linear(self.n_class, 1).to(device)
            setattr(self, f'module_{module_idx}_head', param)
        print(getattr(self, f'module_{0}_head'))

    def forward(self, inputs):
        predicts = []
        for module_idx in range(self.n_class):
            each_module_pred = self.module_predict(inputs, module_idx)
            predicts.append(each_module_pred)
        predicts = torch.cat(predicts, dim=1)
        return predicts

    def module_predict(self, x, module_idx):
        # part1
        x = self.model.relu(self.model.conv_0(x))
        layer_param_init = getattr(self, f'module_0_layer0_0_conv_0')
        layer_param_proc = self.sign(layer_param_init)
        x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)

        # part2
        res_cfg = self.model.layer_configs                      # res_cfg = [3]*3
        for l_idx in range(len(res_cfg)):                       # l_idx = 0,1,2
            layer_idx = l_idx+1                                 # layer_idx = 1,2,3
            layer = getattr(self.model, f'layer{layer_idx}')    # layer = model.layer1 / model.layer2 / model.layer3
            
            for block_idx in range(len(layer)):                 # block_idx = 0,1,2
                block = layer[block_idx]
                #print('*'*20)
                #print(f'module_{module_idx}_layer{layer_idx}_block{block_idx}')
                #print(block)
                identity = x

                # conv_0
                out = block.relu(block.conv_0(x))
                layer_param_init = getattr(self, f'module_{module_idx}_layer{layer_idx}_{block_idx}_conv_0')
                layer_param_proc = self.sign(layer_param_init)
                out = torch.einsum('j, ijkl->ijkl', layer_param_proc, out)

                # conv_1
                out = block.conv_1(out)
                # module_name = get_res_target_name(layer_idx, block_idx, self.model.layer_configs)
                # layer_param_init = getattr(self, f'module_{module_idx}_{module_name}') 
                layer_param_init = getattr(self, f'module_{module_idx}_layer{layer_idx}_{block_idx}_conv_1')
                layer_param_proc = self.sign(layer_param_init)
                out = torch.einsum('j, ijkl->ijkl', layer_param_proc, out)

                if block_idx==0 and layer_idx>1: # downsample = True
                    identity = block.downsample(x)

                out += identity
                out = block.relu(out)

                x = out
            

        # part 3
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        pred = self.model.fc(x)

        module_head = getattr(self, f'module_{module_idx}_head')
        pred = torch.relu(pred)
        head_output = torch.sigmoid(module_head(pred))
        return head_output

    def get_module_params(self):
        module_params = OrderedDict()
        total_params = self.state_dict()
        for layer_name in total_params:
            if layer_name.startswith('module'):
                if 'conv' in layer_name:
                    module_params[layer_name] = (total_params[layer_name] > 0).int()
                else:
                    module_params[layer_name] = total_params[layer_name]
        return module_params

    def get_module_kernels(self):
        module_used_kernels = []
        for module_idx in range(self.n_class):
            each_module_kernels = []
            for l_idx, layer_idx in enumerate(self.conv_configs):
                for b_idx, block_idx in enumerate(layer_idx):
                    for idx, conv_idx in enumerate(block_idx):
                        # module_*_layer*_*_conv_*
                        conv_param = getattr(self,  f'module_{module_idx}_layer{l_idx}_{b_idx}_conv_{idx}')
                        each_module_kernels.append(self.sign(conv_param))
            module_used_kernels.append(torch.cat(each_module_kernels))
        return torch.stack(module_used_kernels)
                        


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
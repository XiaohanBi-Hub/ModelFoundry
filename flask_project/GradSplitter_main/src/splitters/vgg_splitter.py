import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import Union, List, Dict, Any, cast

cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class GradSplitter(nn.Module):
    def __init__(self, model, module_init_type):
        super(GradSplitter, self).__init__()
        self.model = model
        self.n_class = model.num_classes
        self.sign = MySign.apply
        for p in model.parameters():
            p.requires_grad = False

        model_name = model.model_name
        if model_name.endswith("_bn"):
            batch_norm = True
            model_name = model_name.split("_")[0]
            
        self.conv_configs = cfgs[model_name]
        self.module_params = []
        self.init_modules(module_init_type)

    def init_modules(self, module_init_type):
        for module_idx in range(self.n_class):
            layer_idx=0
            for v in self.conv_configs:
            # for layer_idx in range(len(self.conv_configs)):
                if v == 'M':
                    continue
                v = cast(int, v)

                if module_init_type == 'random':
                    param = torch.randn(v).to(device)
                elif module_init_type == 'ones':
                    param = torch.ones(v).to(device)
                elif module_init_type == 'zeros':
                    param = torch.zeros(v).to(device)
                else:
                    raise ValueError

                setattr(self, f'module_{module_idx}_conv_{layer_idx}', nn.Parameter(param))
                # print(f'module_{module_idx}_conv_{layer_idx}')
                layer_idx=layer_idx+1

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
        conv_idx = 0
        max_pool_idx = 0
        for v in self.conv_configs:
            if v == 'M':
                max_pool_layer = getattr(self.model, f'avg_pool_{max_pool_idx}')
                x=max_pool_layer(x)
                max_pool_idx=max_pool_idx+1
            else:
                v = cast(int, v)
                conv_layer = getattr(self.model, f'conv_{conv_idx}')
                x = conv_layer(x)

                layer_param_init = getattr(self, f'module_{module_idx}_conv_{conv_idx}')
                layer_param_proc = self.sign(layer_param_init)

                x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)
                conv_idx=conv_idx+1

        # x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        pred = self.model.classifier(x)

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
            # for layer_idx in range(len(self.conv_configs)):
            layer_idx = 0
            for v in self.conv_configs:
                if v == 'M':
                    continue
                v = cast(int, v)
                layer_param = getattr(self, f'module_{module_idx}_conv_{layer_idx}')
                each_module_kernels.append(self.sign(layer_param))
                layer_idx=layer_idx+1 

            module_used_kernels.append(torch.cat(each_module_kernels))
        return torch.stack(module_used_kernels)


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
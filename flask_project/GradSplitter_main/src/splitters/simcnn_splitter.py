import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradSplitter(nn.Module):
    def __init__(self, model, module_init_type):
        super(GradSplitter, self).__init__()
        self.model = model
        self.n_class = model.num_classes
        self.n_modules = self.n_class
        self.sign = MySign.apply
        for p in model.parameters():
            p.requires_grad = False

        self.conv_configs = [64, 64,
                             128, 128,
                             256, 256, 256,
                             512, 512, 512,
                             512, 512, 512]

        self.max_pooling_idx = [1, 3, 6, 9, 12]

        self.module_params = []
        self.init_modules(module_init_type)

    def init_modules(self, module_init_type):
        for module_idx in range(self.n_modules):
            for layer_idx in range(len(self.conv_configs)):
                if module_init_type == 'random':
                    param = torch.randn(self.conv_configs[layer_idx]).to(device)
                elif module_init_type == 'ones':
                    param = torch.ones(self.conv_configs[layer_idx]).to(device)
                elif module_init_type == 'zeros':
                    param = torch.zeros(self.conv_configs[layer_idx]).to(device)
                else:
                    raise ValueError

                setattr(self, f'module_{module_idx}_conv_{layer_idx}', nn.Parameter(param))

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
        for module_idx in range(self.n_modules):
            each_module_pred = self.module_predict(inputs, module_idx)
            predicts.append(each_module_pred)
        predicts = torch.cat(predicts, dim=1)
        return predicts

    def module_predict(self, x, module_idx):
        for layer_idx in range(len(self.conv_configs)):
            conv_layer = getattr(self.model, f'conv_{layer_idx}')
            x = torch.relu(conv_layer(x))

            layer_param_init = getattr(self, f'module_{module_idx}_conv_{layer_idx}')
            layer_param_proc = self.sign(layer_param_init)

            x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)
            if layer_idx in self.max_pooling_idx:
                x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.shape[0], -1)
        x = torch.relu(self.model.fc_13(x))
        x = torch.relu(self.model.fc_14(x))
        pred = self.model.fc_15(x)

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
        for module_idx in range(self.n_modules):
            each_module_kernels = []
            for layer_idx in range(len(self.conv_configs)):
                layer_param = getattr(self, f'module_{module_idx}_conv_{layer_idx}')
                each_module_kernels.append(self.sign(layer_param))
            module_used_kernels.append(torch.cat(each_module_kernels))
        return torch.stack(module_used_kernels)


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
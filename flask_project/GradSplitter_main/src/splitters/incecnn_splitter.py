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
        self.sign = MySign.apply
        for p in model.parameters():
            p.requires_grad = False

        self.conv_configs = [pair[1] for pair in model.conv_configs]

        self.module_params = []
        self.init_modules(module_init_type)

    def init_modules(self, module_init_type):
        for module_idx in range(self.n_class):
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
            # param = nn.Linear(self.n_class, 1).to(device)

            # multi-layer head 10 -> 10 -> 1
            param = nn.Sequential(
                nn.Linear(self.n_class, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            ).to(device)

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
        # 3 pre layers
        x = torch.relu(self.model.conv_0(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_0')
        layer_param_proc = self.sign(layer_param_init)
        x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)

        x = torch.relu(self.model.conv_1(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_1')
        layer_param_proc = self.sign(layer_param_init)
        x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)

        x = torch.relu(self.model.conv_2(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_2')
        layer_param_proc = self.sign(layer_param_init)
        x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)

        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # first 3-layer
        out_1 = torch.relu(self.model.conv_3(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_3')
        layer_param_proc = self.sign(layer_param_init)
        out_1 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_1)

        out_2 = torch.relu(self.model.conv_4(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_4')
        layer_param_proc = self.sign(layer_param_init)
        out_2 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_2)

        out_3 = torch.relu(self.model.conv_5(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_5')
        layer_param_proc = self.sign(layer_param_init)
        out_3 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_3)

        x = torch.cat([out_1, out_2, out_3], dim=1)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # second 3-layer
        out_1 = torch.relu(self.model.conv_6(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_6')
        layer_param_proc = self.sign(layer_param_init)
        out_1 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_1)

        out_2 = torch.relu(self.model.conv_7(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_7')
        layer_param_proc = self.sign(layer_param_init)
        out_2 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_2)

        out_3 = torch.relu(self.model.conv_8(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_8')
        layer_param_proc = self.sign(layer_param_init)
        out_3 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_3)

        x = torch.cat([out_1, out_2, out_3], dim=1)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # third 3-layer
        out_1 = torch.relu(self.model.conv_9(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_9')
        layer_param_proc = self.sign(layer_param_init)
        out_1 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_1)

        out_2 = torch.relu(self.model.conv_10(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_10')
        layer_param_proc = self.sign(layer_param_init)
        out_2 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_2)

        out_3 = torch.relu(self.model.conv_11(x))
        layer_param_init = getattr(self, f'module_{module_idx}_conv_11')
        layer_param_proc = self.sign(layer_param_init)
        out_3 = torch.einsum('j, ijkl->ijkl', layer_param_proc, out_3)

        x = torch.cat([out_1, out_2, out_3], dim=1)

        # final predict
        x = self.model.avg_pool_12(x)
        x = x.view(x.size(0), -1)
        pred = self.model.fc_12(x)

        # head
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
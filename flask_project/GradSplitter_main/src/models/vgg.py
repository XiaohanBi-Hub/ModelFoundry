"""
Based on:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
https://github.com/chenyaofo/pytorch-cifar-models.
2024-01-11
"""

import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast

cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
conv_cfgs = {
    "vgg11": [(3, 64), (64, 128), (128, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512)],
    "vgg13": [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512)],
    "vgg16": [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512)],
    "vgg19": [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512)]
}

class VGG(nn.Module):
    def __init__(self, model_name="vgg16_bn", num_classes=10, conv_configs=None, 
                 batch_norm: bool = False):
        super(VGG, self).__init__()
        # print(conv_configs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.conv_num = 0
        self.maxpool_num = 0
        self.is_modular = True
        if model_name.endswith("_bn"):
            batch_norm = True
            model_name = model_name.split("_")[0]
            # print(model_name)

        if conv_configs is None:
            # layer_configs -> conv_configs
            self.is_modular = False
            conv_configs = conv_cfgs[model_name]
            layer_configs = cfgs[model_name]
        else:
            # conv_configs -> layer_configs
            layer_configs = []
            indx = 0
            for v in cfgs[model_name]:
                if v == "M":
                    layer_configs.append(v)
                else:
                    cfg = conv_configs[indx][1]
                    layer_configs.append(cast(int, cfg))
                    indx = indx + 1

        self.layer_configs = layer_configs
        self.conv_configs = conv_configs
        self._layers = self._make_layer(self.layer_configs, batch_norm=batch_norm)

        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096)
            nn.Linear(conv_configs[-1][-1], 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        
        if self.is_modular:
            # self.module_head = nn.Linear(num_classes, 1)
            module_head_dim = 1
            self.module_head = nn.Sequential(
                nn.Linear(self.num_classes, 10),
                nn.ReLU(),
                nn.Linear(10, module_head_dim),
            )

        if not self.is_modular:
            self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        conv_idx = 0
        max_pool_idx = 0
        for v in self.layer_configs:
            if v == 'M':
                max_pool_layer = getattr(self, f'avg_pool_{max_pool_idx}')
                x=max_pool_layer(x)
                max_pool_idx=max_pool_idx+1
            else:
                v = cast(int, v)
                conv_layer = getattr(self, f'conv_{conv_idx}')
                x = conv_layer(x)
                conv_idx=conv_idx+1 

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.is_modular:
            x = torch.relu(x)
            x = torch.sigmoid(self.module_head(x))

        return x

    def _make_layer(self, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        conv_idx = 0
        max_pool_idx = 0
        for v in cfg:
            if v == 'M':
                max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
                setattr(self, f'avg_pool_{max_pool_idx}', max_pool)
                layers += [max_pool]
                max_pool_idx=max_pool_idx+1
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    setattr(self, f'conv_{conv_idx}', nn.Sequential(conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)))
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    setattr(self, f'conv_{conv_idx}', nn.Sequential(conv2d, nn.ReLU(inplace=True)))
                    layers += [conv2d, nn.ReLU(inplace=True)]
                conv_idx=conv_idx+1
                in_channels = v
        # return nn.Sequential(*layers)
        self.conv_num = conv_idx
        self.maxpool_num = max_pool_idx
        return layers

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            
   


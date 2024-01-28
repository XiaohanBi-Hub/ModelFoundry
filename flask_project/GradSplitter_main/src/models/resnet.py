"""
Based on:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
https://github.com/chenyaofo/pytorch-cifar-models.
2024-01-11
"""

import sys
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    '''
    input: conv_configs, stride=1, downsample
        -> conv_configs: [(inplanes, planes), (planes, planes)]
    '''
    expansion: int = 1

    def __init__(self, conv_configs, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        conv_0_cfg = conv_configs[0]
        conv_1_cfg = conv_configs[1]
        # self.conv_0 = nn.Sequential(conv3x3(inplanes, planes, stride), nn.BatchNorm2d(planes))
        self.conv_0 = nn.Sequential(conv3x3(conv_0_cfg[0], conv_0_cfg[1], stride), nn.BatchNorm2d(conv_0_cfg[1]))
        self.relu = nn.ReLU(inplace=True)
        # self.conv_1 = nn.Sequential(conv3x3(planes, planes), nn.BatchNorm2d(planes))
        self.conv_1 = nn.Sequential(conv3x3(conv_1_cfg[0], conv_1_cfg[1]), nn.BatchNorm2d(conv_1_cfg[1]))

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.relu(self.conv_0(x))
        out = self.conv_1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, model_name="resnet20", num_classes=10, conv_configs=None):
        super(ResNet, self).__init__()
        self.model_name = model_name
        block = BasicBlock
        self.num_classes = num_classes
        self.is_modular = True
        if conv_configs is None:
            conv_configs = resnet_conv_configs[model_name]
            self.is_modular = False
        else:
            # print(conv_configs)
            idx = 0
            ref_conv_cfg = resnet_conv_configs[model_name]
            for layer_idx, layer_cfg in enumerate(ref_conv_cfg):        # layer_cfg = [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]]
                for block_idx, block_cfg in enumerate(layer_cfg):       # block_cfg = [(16, 32), (32, 32), (16, 32)]
                    for conv_idx, conv_cfg in enumerate(block_cfg):     # conv_cfg = (16, 32)
                        block_cfg[conv_idx] = conv_configs[idx]
                        idx = idx +1
            conv_configs = ref_conv_cfg
            # [(3, 1), (1, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 32), (32, 32), (32, 32), (32, 32), (32, 32), (32, 32), (32, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 62)]


        self.conv_configs = conv_configs
        layers = cfgs[model_name]
        self.layer_configs = layers

        self.inplanes = 16

        conv0_cfg=self.conv_configs[0][0][0]
        # print(conv0_cfg)    # (3,16)

        # self.conv_0 = nn.Sequential(conv3x3(3, 16), nn.BatchNorm2d(16))
        self.conv_0 = nn.Sequential(conv3x3(conv0_cfg[0], conv0_cfg[1]), nn.BatchNorm2d(conv0_cfg[1]))
        self.relu = nn.ReLU(inplace=True)
        
        # < -- ! torchvision -- >
        #   > The head conv structure should be modified (stride=1 and no pooling,) 
        #   > because the image size of CIFAR-10 is very small.
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # [def _make_layer] input: self, block, block_configs, blocks
        self.layer1 = self._make_layer(block, self.conv_configs[1], layers[0])
        self.layer2 = self._make_layer(block, self.conv_configs[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.conv_configs[3], layers[2], stride=2)
        # self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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

        # print(self.conv_configs)

    def forward(self, x):
        x = self.relu(self.conv_0(x))
        # <!-- torchvision -- >
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, block_configs, blocks, stride=1):
    # def _make_layer(self, block, planes, blocks, stride=1):
        '''
        input: self, block, block_configs, blocks
            -> block: BasicBlock
            -> block_configs: [conv_config1, conv_config2, ...]
                -> conv_config* = [(conv1_in, conv1_out), (,) ,...]
            -> blocks: cfgs
        '''
        ### block 0
        first_block_conv_configs = block_configs[0] 
        # first_block_conv_configs = [(conv1_in, conv1_out), (,) ,...]
        
        downsample = None
        if stride != 1 or self.inplanes != first_block_conv_configs[-1][1] * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, first_block_conv_configs[-1][1] * block.expansion, stride),
                nn.BatchNorm2d(first_block_conv_configs[-1][1] * block.expansion),
            )
        layers = []
        
        # [BasicBlock] input: conv_configs, stride=1, downsample
        new_block = block(block_configs[0], stride, downsample)
        layers.append(new_block)
        
        ### other blocks
        self.inplanes = first_block_conv_configs[-1][1] * block.expansion
        for i in range(1, blocks):
            new_block = block(block_configs[i])
            layers.append(new_block)
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    '''
    def _make_layer(self, block, planes, blocks, stride=1):
        ### block 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        
        new_block = block(self.inplanes, planes, stride, downsample)
        layers.append(new_block)
        
        ### other blocks
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            new_block = block(self.inplanes, planes)
            layers.append(new_block)
        return nn.Sequential(*layers)
        '''
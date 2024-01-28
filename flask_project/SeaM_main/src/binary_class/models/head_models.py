import torch
import torch.nn as nn
import torchvision.models as models

import os
import sys
sys.path.append("../..")
sys.path.append("..")
print(sys.path)
try:
    from SeaM_main.src.binary_class.models.nn_layers import MaskConv, MaskLinear, Binarization
except ModuleNotFoundError:
    from models.nn_layers import MaskConv, MaskLinear, Binarization


class HeadVGG(nn.Module):
    # This is for adding a Head after the output layer for modularization
    def __init__(self, 
                 original_model, 
                 num_classes: int = 2, 
                 is_reengineering: bool = False):
        super(HeadVGG, self).__init__()
        self.is_reengineering = is_reengineering
        self.features = original_model.features
        if hasattr(original_model, 'avgpool'):
            self.avgpool = original_model.avgpool
        else:
            self.avgpool = None
        self.classifier = original_model.classifier

        if self.is_reengineering:
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(in_features=original_model.classifier[-1].out_features,\
                          out_features=num_classes))

    def forward(self,x):
        x = self.features(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.is_reengineering:
            x = self.module_head(x)
        return x
    
    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def count_weight_ratio(self):
        masks = []
        for n, layer in self.named_modules():
            if hasattr(layer, 'weight_mask'):
                masks.append(torch.flatten(layer.weight_mask))
                if layer.bias_mask is not None:
                    masks.append(torch.flatten(layer.bias_mask))

        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        weight_ratio = torch.mean(bin_masks)
        return weight_ratio

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head
    
class HeadResNet(nn.Module):
    # This is for adding a Head after the output layer for modularization
    def __init__(self, 
                 original_model, 
                 num_classes: int = 2, 
                 is_reengineering: bool = False):
        super(HeadResNet, self).__init__()
        self.is_reengineering = is_reengineering
        self.original_model = original_model
        # Original output features dim
        num_ftrs = original_model.fc.out_features

        if self.is_reengineering:
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(in_features=num_ftrs,\
                          out_features=num_classes))
            
    def forward(self,x):
        x = self.original_model(x)
        x = self.module_head(x)
        return x

    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def count_weight_ratio(self):
        masks = []
        for n, layer in self.named_modules():
            if hasattr(layer, 'weight_mask'):
                masks.append(torch.flatten(layer.weight_mask))
                if layer.bias_mask is not None:
                    masks.append(torch.flatten(layer.bias_mask))

        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        weight_ratio = torch.mean(bin_masks)
        return weight_ratio

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head

class HeadCNNs(nn.Module):
    # This is for adding a Head after the output layer for modularization
    def __init__(self, 
                 original_model, 
                 num_classes: int = 2, 
                 is_reengineering: bool = False):
        super(HeadCNNs, self).__init__()
        self.is_reengineering = is_reengineering
        self.original_model = original_model
        # Original output features dim
        # num_ftrs = original_model.fc_15.out_features
        # SimCNN,InceCNN,ResCNN are both 10 classes on cifar10 and svhn
        num_ftrs = 10

        if self.is_reengineering:
            self.module_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(in_features=num_ftrs,\
                          out_features=num_classes))
            
    def forward(self, x):
        x = self.original_model(x)
        x = self.module_head(x)
        return x
    
    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def count_weight_ratio(self):
        masks = []
        for n, layer in self.named_modules():
            if hasattr(layer, 'weight_mask'):
                masks.append(torch.flatten(layer.weight_mask))
                if layer.bias_mask is not None:
                    masks.append(torch.flatten(layer.bias_mask))

        masks = torch.cat(masks, dim=0)
        bin_masks = Binarization.apply(masks)
        weight_ratio = torch.mean(bin_masks)
        return weight_ratio

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head
            
    
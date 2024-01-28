import torch
import torch.nn as nn
import torchvision.models as models
import copy
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")
sys.path.append("..")
print(sys.path)
try:
    from SeaM_main.src.binary_class.models.nn_layers import MaskConv, MaskLinear, Binarization
    from SeaM_main.src.binary_class.models.head_models import HeadVGG
except ModuleNotFoundError:
    from models.nn_layers import MaskConv, MaskLinear, Binarization
    from SeaM_main.src.binary_class.models.head_models import HeadVGG




class MaskCNN():
    # This is for masking Conv and Linear layer in CNN models
    def __init__(self) -> None:
        pass
    def replace(self, model, is_reengineering: bool = False):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                maskconv = MaskConv(module.in_channels, 
                                    module.out_channels, 
                                    module.kernel_size, 
                                    stride=module.stride, 
                                    padding=module.padding, 
                                    dilation=module.dilation, 
                                    groups=module.groups, 
                                    bias=(module.bias is not None),
                                    is_reengineering=is_reengineering)
                # copy weight and bias
                maskconv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    maskconv.bias.data = module.bias.data.clone()
                setattr(model, name, maskconv)
            elif isinstance(module, nn.Linear):
                masklinear = MaskLinear(module.in_features, 
                                        module.out_features, 
                                        bias=(module.bias is not None),
                                        is_reengineering=is_reengineering)
                
                # copy weight and bias
                masklinear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    masklinear.bias.data = module.bias.data.clone()
                setattr(model, name, masklinear)
            elif len(list(module.children())) > 0:
                # recursively replace layers in sub-modules
                self.replace(module, is_reengineering=is_reengineering)  
        return model
    
maskcnn = MaskCNN()

if __name__ == "__main__":
    vgg16_bn = models.vgg16_bn()
    print(vgg16_bn)
    vgg16_replaced = maskcnn.replace(vgg16_bn)
    headvgg = HeadVGG(vgg16_replaced, is_reengineering=True)
    print(headvgg)

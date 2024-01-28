import numpy as np
import torch
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def module_info(model_name):
    if model_name == 'simcnn':
        getter = get_target_module_info_for_simcnn
    elif model_name == 'rescnn':
        getter = get_target_module_info_for_rescnn
    elif model_name == 'incecnn':
        getter = get_target_module_info_for_incecnn
    elif model_name.startswith("vgg"):
        getter = get_target_module_info_for_vggs
    elif model_name.startswith("resnet"):
        getter = get_target_module_info_for_resnets
    else:
        raise ValueError()
    return getter


def get_target_module_info_for_simcnn(modules_info, target_class, trained_model, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    for conv_idx in range(len(modules_info)):
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break

    if f'module_{target_class}_head.weight' in modules_info:  # head with one layer
        module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
        module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    elif f'module_{target_class}_head.0.weight' in modules_info:  # head with multi-layer
        module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
        module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
        module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
        module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    else:
        raise KeyError
    return module_conv_info, module_head_para


def get_target_module_info_for_rescnn(modules_info, target_class, trained_model, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    residual_layer_indices = trained_model.residual_idx
    for conv_idx in range(len(modules_info)):
        if conv_idx in residual_layer_indices:
            layer_name = f'module_{target_class}_conv_{conv_idx - 2}'
        else:
            layer_name = f'module_{target_class}_conv_{conv_idx}'

        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break
    # module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
    # module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
    module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
    module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
    module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    return module_conv_info, module_head_para


def get_target_module_info_for_incecnn(modules_info, target_class, trained_model, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    for conv_idx in range(len(modules_info)):
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break
    # module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
    # module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
    module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
    module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
    module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    return module_conv_info, module_head_para


def get_target_module_info_for_vggs(modules_info, target_class, trained_model, handle_warning):
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    for conv_idx in range(len(modules_info)):
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break
    module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
    module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
    module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
    module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    return module_conv_info, module_head_para


def get_target_module_info_for_resnets(modules_info, target_class, trained_model, handle_warning):
    print("get_target_module_info_for_resnets")
    module_conv_info = []  # {[1 0 0 1..]... [0 0 1 1]} -> [[0,1,2,5,9,63], ..., [1,34,100,111,...]] indices of retained kernels.
    module_head_para = OrderedDict()
    # print(modules_info.keys())

    for layer_idx, layer_cfg in enumerate(trained_model.conv_configs):
        for block_idx, block_cfg in enumerate(layer_cfg):       # block_cfg = [(16, 32), (32, 32), (16, 32)]
            for conv_idx, conv_cfg in enumerate(block_cfg):     # conv_cfg = (16, 32)
                layer_name = f'module_{target_class}_layer{layer_idx}_{block_idx}_conv_{conv_idx}'
                if layer_name in modules_info:
                    each_conv_info = modules_info[layer_name]
                    each_conv_info = each_conv_info.numpy()
                    idx_info = np.argwhere(each_conv_info == 1)
                    if idx_info.size == 0 and handle_warning:
                        idx_info = np.array([[0]]).astype('int64')
                    module_conv_info.append(np.squeeze(idx_info, axis=-1))
                else:
                    break
    
    module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
    module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
    module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
    module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    return module_conv_info, module_head_para

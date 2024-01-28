import numpy as np
import torch
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _extract_module_sim_res(module_conv_info, module_head_para, trained_model, model_network):
    """
    for SimCNN and ResCNN.
    conv_info: tensor [[1 0 0 1] ...[0 0 1 1]]
    """
    # get the configures of module from the update_conv_info
    conv_configs = []
    cin = 3
    for each_conv_layer in module_conv_info:
        n_kernels = each_conv_layer.size
        conv_configs.append((cin, n_kernels))
        cin = n_kernels

    module = model_network(num_classes=trained_model.num_classes, conv_configs=conv_configs)
    # print(conv_configs)
    # extract the parameters of active kernels from model
    active_kernel_param = {}
    model_param = trained_model.state_dict()
    for i in range(len(conv_configs)):
        conv_weight = model_param[f'conv_{i}.0.weight']
        conv_bias = model_param[f'conv_{i}.0.bias']
        bn_weight = model_param[f'conv_{i}.1.weight']
        bn_bias = model_param[f'conv_{i}.1.bias']
        bn_running_mean = model_param[f'conv_{i}.1.running_mean']
        bn_running_var = model_param[f'conv_{i}.1.running_var']

        cur_conv_active_kernel_idx = module_conv_info[i]  # active Cout
        pre_conv_active_kernel_idx = module_conv_info[i-1] if i > 0 else list(range(3))  # active Cin

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

        active_kernel_param[f'conv_{i}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]

    assert model_param[f'fc_{len(conv_configs)}.weight'].size(1) == model_param[f'conv_{len(conv_configs)-1}.0.bias'].size(0)
    first_fc_weight = model_param[f'fc_{len(conv_configs)}.weight']
    pre_conv_active_kernel_idx = module_conv_info[-1]
    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'fc_{len(conv_configs)}.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    model_param.update(module_head_para)
    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module


def _extract_module_ince(module_conv_info, module_head_para, trained_model, model_network):
    pre_layers = [0, 1, 2]
    ince_layer_1 = [3, 4, 5]
    ince_layer_2 = [6, 7, 8]
    ince_layer_3 = [9, 10, 11]

    conv_configs = []
    for layer_idx in range(len(module_conv_info)):
        if layer_idx in pre_layers:
            cin = conv_configs[layer_idx - 1][1] if layer_idx > 0 else 3
            cout = module_conv_info[layer_idx].size
        elif layer_idx in ince_layer_1:
            cin = conv_configs[2][1]
            cout = module_conv_info[layer_idx].size
        elif layer_idx in ince_layer_2:
            cin = sum([conv_configs[i][1] for i in ince_layer_1])
            cout = module_conv_info[layer_idx].size
        elif layer_idx in ince_layer_3:
            cin = sum([conv_configs[i][1] for i in ince_layer_2])
            cout = module_conv_info[layer_idx].size
        else:
            raise ValueError
        conv_configs.append((cin, cout))

    module = model_network(num_classes=trained_model.num_classes, conv_configs=conv_configs)

    # extract the parameters of active kernels from model
    active_kernel_param = {}
    model_param = trained_model.state_dict()
    for i in range(len(conv_configs)):
        conv_weight = model_param[f'conv_{i}.0.weight']
        conv_bias = model_param[f'conv_{i}.0.bias']
        bn_weight = model_param[f'conv_{i}.1.weight']
        bn_bias = model_param[f'conv_{i}.1.bias']
        bn_running_mean = model_param[f'conv_{i}.1.running_mean']
        bn_running_var = model_param[f'conv_{i}.1.running_var']

        cur_conv_active_kernel_idx = module_conv_info[i]  # active Cout

        # active Cin
        if i in pre_layers:
            pre_conv_active_kernel_idx = module_conv_info[i - 1] if i > 0 else list(range(3))
        elif i in ince_layer_1:
            pre_conv_active_kernel_idx = module_conv_info[2]
        elif i in ince_layer_2:
            tmp_last_conv_len = [model_param[f'conv_{tmp_idx}.0.bias'].size(0) for tmp_idx in ince_layer_1]
            pre_3_conv_active_kernel_idx = [module_conv_info[tmp_idx] for tmp_idx in ince_layer_1]
            pre_conv_active_kernel_idx = [
                pre_3_conv_active_kernel_idx[0],
                pre_3_conv_active_kernel_idx[1] + tmp_last_conv_len[0],
                pre_3_conv_active_kernel_idx[2] + tmp_last_conv_len[0] + tmp_last_conv_len[1]
            ]
            pre_conv_active_kernel_idx = np.concatenate(pre_conv_active_kernel_idx, axis=0)
        elif i in ince_layer_3:
            tmp_last_conv_len = [model_param[f'conv_{tmp_idx}.0.bias'].size(0) for tmp_idx in ince_layer_2]
            pre_3_conv_active_kernel_idx = [module_conv_info[tmp_idx] for tmp_idx in ince_layer_2]
            pre_conv_active_kernel_idx = [
                pre_3_conv_active_kernel_idx[0],
                pre_3_conv_active_kernel_idx[1] + tmp_last_conv_len[0],
                pre_3_conv_active_kernel_idx[2] + tmp_last_conv_len[0] + tmp_last_conv_len[1]
            ]
            pre_conv_active_kernel_idx = np.concatenate(pre_conv_active_kernel_idx, axis=0)
        else:
            raise ValueError

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

        active_kernel_param[f'conv_{i}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]

    # check
    tmp_fc_len = model_param[f'fc_{len(conv_configs)}.weight'].size(1)
    tmp_last_conv_len = [model_param[f'conv_{len(conv_configs) - 3}.0.bias'].size(0),
                         model_param[f'conv_{len(conv_configs) - 2}.0.bias'].size(0),
                         model_param[f'conv_{len(conv_configs) - 1}.0.bias'].size(0)]
    assert tmp_fc_len == sum(tmp_last_conv_len)

    first_fc_weight = model_param[f'fc_{len(conv_configs)}.weight']
    conv_a_active_kernel_idx = module_conv_info[-3]
    conv_b_active_kernel_idx = np.array(module_conv_info[-2]) + tmp_last_conv_len[0]
    conv_c_active_kernel_idx = np.array(module_conv_info[-1]) + tmp_last_conv_len[0] + tmp_last_conv_len[1]
    pre_conv_active_kernel_idx = np.concatenate(
        [conv_a_active_kernel_idx, conv_b_active_kernel_idx, conv_c_active_kernel_idx],
        axis=0
    )

    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'fc_{len(conv_configs)}.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    model_param.update(module_head_para)
    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module


def _extract_module_vggs(module_conv_info, module_head_para, trained_model, model_network):
    """
    for VGG
    conv_info: tensor [[1 0 0 1] ...[0 0 1 1]]
    """
    # get the configures of module from the update_conv_info
    conv_configs = [] 
    cin = 3
    for each_conv_layer in module_conv_info:
        n_kernels = each_conv_layer.size
        conv_configs.append((cin, n_kernels))
        cin = n_kernels

    module = model_network(model_name=trained_model.model_name, num_classes=trained_model.num_classes, conv_configs=conv_configs)

    active_kernel_param = {}
    model_param = trained_model.state_dict()
    for i in range(len(conv_configs)):
        conv_weight = model_param[f'conv_{i}.0.weight']
        conv_bias = model_param[f'conv_{i}.0.bias']
        bn_weight = model_param[f'conv_{i}.1.weight']
        bn_bias = model_param[f'conv_{i}.1.bias']
        bn_running_mean = model_param[f'conv_{i}.1.running_mean']
        bn_running_var = model_param[f'conv_{i}.1.running_var']

        cur_conv_active_kernel_idx = module_conv_info[i]  # active Cout
        pre_conv_active_kernel_idx = module_conv_info[i-1] if i > 0 else list(range(3))  # active Cin

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

        active_kernel_param[f'conv_{i}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]

    first_classifier_weight = model_param[f'classifier.0.weight']
    pre_conv_active_kernel_idx = module_conv_info[-1]
    active_first_classifier_weight = first_classifier_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'classifier.0.weight'] = active_first_classifier_weight
    
    model_param.update(active_kernel_param)
    model_param.update(module_head_para)
    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module


def _extract_module_resnets(module_conv_info, module_head_para, trained_model, model_network):
    print("TODO") 
    """
    for ResNets
    conv_info: tensor [[1 0 0 1] ...[0 0 1 1]]
    """
    conv_configs = [] 
    cin = 3
    for each_conv_layer in module_conv_info:
        n_kernels = each_conv_layer.size
        conv_configs.append((cin, n_kernels))
        cin = n_kernels
    # print(conv_configs) # [(3, 1), (1, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 32), (32, 32), (32, 32), (32, 32), (32, 32), (32, 32), (32, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 62)]

    module = model_network(model_name=trained_model.model_name, num_classes=trained_model.num_classes, conv_configs=conv_configs)
    # for para in module.named_parameters(): 
    #     print(para[0],'\t',para[1].size())

    active_kernel_param = {}
    model_param = trained_model.state_dict()
    # print(model_param.keys())
    module_conv_config = module.conv_configs
    m_conv_info_idx = 0
    last_conv=[0,0,0]
    for layer_idx, layer_cfg in enumerate(module_conv_config):  # layer_cfg = [[(16, 32), (32, 32)], [(32, 32), (32, 32)], [(32, 32), (32, 32)]]
        last_conv[0]=layer_idx
        for block_idx, block_cfg in enumerate(layer_cfg):       # block_cfg = [(16, 32), (32, 32), (16, 32)]
            last_conv[1]=block_idx
            for conv_idx, conv_cfg in enumerate(block_cfg):     # conv_cfg = (16, 32)
                last_conv[2]=conv_idx
                if layer_idx == 0:
                    # 'conv_0.0.weight'
                    conv_weight = model_param['conv_0.0.weight']    
                    # conv_bias = model_param['conv_0.0.bias']  #conv.bias=False
                    bn_weight = model_param['conv_0.1.weight']
                    bn_bias = model_param['conv_0.1.bias']
                    bn_running_mean = model_param['conv_0.1.running_mean']
                    bn_running_var = model_param['conv_0.1.running_var']
                else:
                    # 'layer1.0.conv_0.0.weight'
                    conv_weight = model_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.0.weight']
                    # conv_bias = model_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.0.bias']
                    bn_weight = model_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.weight']
                    bn_bias = model_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.bias']
                    bn_running_mean = model_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.running_mean']
                    bn_running_var = model_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.running_var']
                        
                cur_conv_active_kernel_idx = module_conv_info[m_conv_info_idx]  # active Cout
                pre_conv_active_kernel_idx = module_conv_info[m_conv_info_idx-1] if m_conv_info_idx > 0 else list(range(3))  # active Cin
                
                tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
                if layer_idx == 0 :
                    active_kernel_param[f'conv_0.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
                    # active_kernel_param[f'conv_0.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

                    active_kernel_param[f'conv_0.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
                    active_kernel_param[f'conv_0.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
                    active_kernel_param[f'conv_0.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
                    active_kernel_param[f'conv_0.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]
                else:
                    active_kernel_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
                    # active_kernel_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

                    active_kernel_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
                    active_kernel_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
                    active_kernel_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
                    active_kernel_param[f'layer{layer_idx}.{block_idx}.conv_{conv_idx}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]       
    

    # assert model_param[f'fc.weight'].size(1) == model_param[f'layer{last_conv[0]}.{last_conv[1]}.conv_{last_conv[2]}.0.bias'].size(0)
    first_fc_weight = model_param[f'fc.weight']
    pre_conv_active_kernel_idx = module_conv_info[-1]
    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'fc.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    model_param.update(module_head_para)

    # for para in model_param:
    #     print(para,'\t',model_param[para].size())

    module.load_state_dict(model_param)
    module = module.to(device).eval()
    return module
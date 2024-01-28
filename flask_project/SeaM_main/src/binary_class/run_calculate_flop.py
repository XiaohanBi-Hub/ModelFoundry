import argparse
import numpy as np
import re
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from numbers import Number
from datasets.dataset_loader_bc import load_dataset
from config import load_config
from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import Any, Callable, List, Optional, Union
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from SeaM_main.src.binary_class.utils.mask_layer import maskcnn
from SeaM_main.src.binary_class.models.head_models import HeadVGG, HeadResNet, HeadCNNs

from models.vgg import cifar10_vgg16_bn as cifar10_vgg16
# from models.vgg import cifar100_vgg16_bn as cifar100_vgg16
from models.vgg import cifar10_vgg11_bn,cifar10_vgg13_bn,cifar10_vgg16_bn,cifar10_vgg19_bn
from models.vgg import cifar100_vgg11_bn,cifar100_vgg13_bn,cifar100_vgg16_bn,cifar100_vgg19_bn
from models.resnet import cifar10_resnet20, cifar100_resnet20

from SeaM_main.src.binary_class.utils.model_loader import load_model, load_trained_model
from GradSplitter_main.src.utils.configure_loader import load_configure as grad_load_config


class CalculateFLOP:
    def __init__(self, model):
        self.model = model
        self.model_modules = dict(list(model.named_modules()))
        self.calculated_convs = []

    @torch.no_grad()
    def calculate_flop_fvcore(self, img_size, is_sparse, show_details=False):
        x = torch.randn(1, 3, img_size, img_size)
        fca = FlopCountAnalysis(self.model, x)
        if is_sparse:
            handlers = {'aten::_convolution': self.masked_conv_flop_jit,
                        'aten::addmm': self.masked_fc_flop_jit}
            fca.set_op_handle(**handlers)

        flops = fca.total()
        if show_details:
            print(flop_count_table(fca))

        return flops

    def masked_conv_flop_jit(self, inputs: List[Any], outputs: List[Any]) -> Number:
        x = inputs[0]
        x_shape = self.get_shape(x)
        batch_size = x_shape[0]
        assert batch_size == 1

        # For vgg16 and resnet20 model. other models may use different inputs[x].
        conv_name = re.match(r'.+scope: /(.+)', str(inputs[3]).strip()).groups(1)[0].strip()
        conv_name = conv_name.split('/')[-1]

        self.calculated_convs.append(conv_name)

        weight = self.model_modules[conv_name].weight.numpy()
        vector_length = np.count_nonzero(weight)

        out_shape = self.get_shape(outputs[0])
        output_H, output_W = out_shape[2], out_shape[3]

        flop_mults = vector_length * output_H * output_W

        # Note: Same as fvcore, we only calculate multiplication, without calculating addition.
        flop_adds = 0
        # flop_adds = (vector_length-1) * output_H * output_W
        flop = flop_mults + flop_adds
        return flop

    def masked_fc_flop_jit(self, inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for fully connected layers.
        """
        # Count flop for nn.Linear
        # inputs is a list of length 3.
        input_shapes = [self.get_shape(v) for v in inputs[1:3]]
        # input_shapes[0]: [batch size, input feature dimension]
        # input_shapes[1]: [batch size, output feature dimension]
        assert len(input_shapes[0]) == 2, input_shapes[0]
        assert len(input_shapes[1]) == 2, input_shapes[1]
        batch_size, input_dim = input_shapes[0]
        assert batch_size == 1


        # For vgg16 and resnet20 model. other models may use different inputs[x].
        fc_name = re.match(r'.+scope: /(.+) #.*', str(inputs[3]).strip()).groups(1)[0].strip()
        fc_name = fc_name.split('/')[-1]

        weight =  self.model_modules[fc_name].weight.numpy()
        flop = np.count_nonzero(weight)

        return flop

    @staticmethod
    def get_shape(val: Any) -> Optional[List[int]]:
        if val.isCompleteTensor():
            return val.type().sizes()
        else:
            return None


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, choices=['vgg16', 'resnet20'], required=True)
#     parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True)
#     parser.add_argument('--target_class', type=int, required=True)

#     parser.add_argument('--lr_head', type=float, default=0.1)
#     parser.add_argument('--lr_mask', type=float, default=0.1)
#     parser.add_argument('--alpha', type=float, default=1,
#                         help='the weight for the weighted sum of two losses in re-engineering.')

#     parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')
#     parser.add_argument('--seed', type=int, default=0,
#                         help='the random seed for sampling ``num_superclasses'' classes as target classes.')

#     args = parser.parse_args()
#     return args

def get_args(model, dataset, target_class, lr_head=0.1, lr_mask=0.1, 
             alpha=1.0, shots=-1, seed=0):
    args = argparse.Namespace()
    args.model = model
    args.dataset = dataset
    args.target_class = int(target_class)
    args.lr_head = float(lr_head)
    args.lr_mask = float(lr_mask)
    args.alpha = float(alpha)
    args.shots = int(shots)
    args.seed = int(seed)
    return args

@torch.no_grad()
def eval_reengineered_model(reengineered_model, test_loader):
    reengineered_model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cpu'), batch_labels.to('cpu')
        batch_outputs = reengineered_model(batch_inputs)
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)
        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def eval_pretrained_model(args, num_workers, pin_memory):
    if "cnn" in args.model:
        # 这里是simcnn，rescnn，incecnn三种情况
        # datasets有svhn和cifar10两种
        grad_config = grad_load_config(args.model,args.dataset)
        grad_config.set_estimator_idx(1)
        model = load_trained_model(args.model,n_classes=10,
                trained_model_path=grad_config.trained_model_path).to('cpu')
    else:
        model = eval(f'{args.dataset}_{args.model}')\
            (pretrained=True).to('cpu')
    dataset_test = load_dataset(args.dataset, is_train=False, 
                                target_class=args.target_class, reorganize=False)
    test_loader = DataLoader(dataset_test, batch_size=64, 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=pin_memory)

    model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cpu'), batch_labels.to('cpu')
        batch_outputs = model(batch_inputs)
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)

        if args.target_class >= 0:
            # Transfer the multiclass classification into the binary classification.
            batch_preds = batch_preds == args.target_class
            batch_labels = batch_labels == args.target_class

        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def main_func(args, num_workers, pin_memory, config):
    dataset_test = load_dataset(args.dataset, is_train=False, 
                                target_class=args.target_class, reorganize=True)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)

    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/tc_{args.target_class}'
    save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'

    model = eval(f'{args.dataset}_{args.model}')(pretrained=True).to('cpu')

    model = maskcnn.replace(model,is_reengineering=True)
    if "cnn" in args.model:
        model = HeadCNNs(model, is_reengineering=True)
    elif "vgg" in args.model:
        model = HeadVGG(model,is_reengineering=True)
    elif "resnet" in args.model:
        model = HeadResNet(model, is_reengineering=True)
        
    module_head = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(10 if args.dataset == 'cifar10' else 100, 2)
    ).to('cpu')
    setattr(model, 'module_head', module_head)

    masks = torch.load(save_path, map_location=torch.device('cpu'))

    # remove irrelevant weights using masks
    model_params = model.state_dict()
    masked_params = OrderedDict()
    check_flag = 0
    sum_mask_ones = 0
    sum_masks = 0
    for name, weight in model_params.items():
        if f'{name}_mask' in masks:
            mask = masks[f'{name}_mask']
            bin_mask = (mask > 0).float()
            # Calculate how many ones and nums in mask
            num_ones = bin_mask.sum().item()
            total_elem = bin_mask.numel()
            sum_mask_ones += num_ones
            sum_masks += total_elem
            
            masked_weight = weight * bin_mask
            masked_params[name] = masked_weight
            check_flag += 1
        elif f'module_head' in name:
            module_head_param = masks[name]
            masked_params[name] = module_head_param
            check_flag += 1
        else:
            masked_params[name] = weight
    # Convert to million
    sum_mask_ones /= 1e6
    sum_masks /= 1e6
    print(f"Weights of original model:{sum_masks}million")
    print(f"Weights of reengineered module:{sum_mask_ones}million")
    print(f"Weight retain rate:{(sum_mask_ones/sum_masks):.2%}")

    assert check_flag == len(masks)
    model.load_state_dict(masked_params)

    acc = eval_reengineered_model(model, test_loader)  # check the re-engineered model's acc
    # print(f'Reengineered Model   ACC: {acc:.2%}')
    acc_reeng = acc

    cal_flop = CalculateFLOP(model)
    img_size = 32
    total_flop_dense = cal_flop.calculate_flop_fvcore(img_size=img_size, is_sparse=False)
    total_flop_sparse = cal_flop.calculate_flop_fvcore(img_size=img_size, is_sparse=True)
    # flops results
    m_total_flop_dense = total_flop_dense / 1e6
    m_total_flop_sparse = total_flop_sparse / 1e6
    perc_sparse_dense = total_flop_sparse / total_flop_dense
    
    # return results
    return m_total_flop_dense, m_total_flop_sparse, perc_sparse_dense, \
            acc_reeng, sum_mask_ones, sum_masks


def run_calculate_flop_bc(model, dataset, target_class, lr_mask, alpha, callback):

    args = get_args(model=model, dataset=dataset, target_class=target_class, 
                    lr_mask=lr_mask, alpha=alpha)
    print(args)
    config = load_config()
    num_workers = 0
    pin_memory = False

    acc_pre = eval_pretrained_model(args, num_workers, pin_memory)  # check the model's acc
    print(f'Pretrained Model   ACC: {acc_pre:.2%}')

    m_total_flop_dense, m_total_flop_sparse, perc_sparse_dense, \
    acc_reeng, sum_mask_ones, sum_masks= main_func(args, num_workers, pin_memory, config)
    
    print(f'FLOPs  Dense: {m_total_flop_dense:.2f}M')
    print(f'FLOPs Sparse: {m_total_flop_sparse:.2f}M')
    print(f'FLOP% (Sparse / Dense): {perc_sparse_dense:.2%}')
    print(f'Reengineered Model   ACC: {acc_reeng:.2%}\n')
    if callback!="debug":
        callback(m_total_flop_dense=m_total_flop_dense, m_total_flop_sparse=m_total_flop_sparse, \
             perc_sparse_dense=perc_sparse_dense, acc_reeng=acc_reeng, acc_pre=acc_pre,\
                sum_mask_ones=sum_mask_ones, sum_masks=sum_masks,weight_retain_rate=sum_mask_ones/sum_masks)
    
    return m_total_flop_dense, m_total_flop_sparse, \
            perc_sparse_dense, acc_reeng, acc_pre
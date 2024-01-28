import numpy as np
import torch
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import GradSplitter_main.src.models as models_repo
from GradSplitter_main.src.utils.module_info_getter import module_info
import GradSplitter_main.src.utils.module_extractor as extractor


def extract_module(module_conv_info, module_head_para, trained_model):
    tm_name = trained_model.model_name
    # print(f'extract_module trained_model name: {tm_name}')

    if tm_name == 'simcnn':
        from GradSplitter_main.src.models.simcnn import SimCNN
        module = extractor._extract_module_sim_res(module_conv_info, module_head_para, trained_model, SimCNN)
    elif tm_name == 'rescnn':
        from GradSplitter_main.src.models.rescnn import ResCNN
        module = extractor._extract_module_sim_res(module_conv_info, module_head_para, trained_model, ResCNN)
    elif tm_name == 'incecnn':
        from GradSplitter_main.src.models.incecnn import InceCNN
        module = extractor._extract_module_ince(module_conv_info, module_head_para, trained_model, InceCNN)
    elif tm_name.startswith("vgg"):
        from GradSplitter_main.src.models.vgg import VGG
        module = extractor._extract_module_vggs(module_conv_info, module_head_para, trained_model, VGG)
    elif tm_name.startswith("resnet"):
        from GradSplitter_main.src.models.resnet import ResNet
        module = extractor._extract_module_resnets(module_conv_info, module_head_para, trained_model, ResNet)

    else:
        raise ValueError
    return module


def get_target_module_info(modules_info, trained_model, target_class, handle_warning=True):
    # tm_class_name = trained_model.__class__.__name__
    tm_name = trained_model.model_name
    get_module_info = module_info(tm_name)
    module_conv_info, module_head_para = get_module_info(modules_info, target_class, trained_model, handle_warning)
    return module_conv_info, module_head_para


def load_module(module_path, trained_model, target_class):
    modules_info = torch.load(module_path, map_location='cpu')
    module_conv_info, module_head_para = get_target_module_info(modules_info, trained_model, target_class)
    module = extract_module(module_conv_info, module_head_para, trained_model)
    return module


@torch.no_grad()
def module_predict(module, dataset):
    outputs, labels = [], []
    for batch_inputs, batch_labels in dataset:
        batch_inputs = batch_inputs.to(device)
        batch_output = module(batch_inputs)
        outputs.append(batch_output)
        labels.append(batch_labels.to(device))
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    return outputs, labels


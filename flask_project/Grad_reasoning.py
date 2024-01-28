import torch
from GradSplitter_main.src.utils.model_loader import load_model
from GradSplitter_main.src.utils.configure_loader import load_configure
from GradSplitter_main.src.utils.splitter_loader import load_splitter

import torch

def load_model_from_pth(model_name, path):

    base_model = load_model(model_name)
    GradSplitterClass = load_splitter(model_name,None)
    model = GradSplitterClass(base_model,"ones")
    state_dict = torch.load(path)
    model.model.load_state_dict(state_dict)
    return model

def generate_module_from_mask(original_model, mask, class_idx):
    # Clone the original model to create a new GradSplitter
    new_model = type(original_model)(original_model.model, 'ones')
    new_model.load_state_dict(original_model.state_dict())
    
    # Apply the mask
    for name, m in mask.items():
        if f"module_{class_idx}" in name:
            param_name = name.split(f"module_{class_idx}_")[-1]
            
            # Use split to get the actual module and its parameter name
            module_name, param_name = param_name.split('.', 1)
            
            # Get the actual module
            module = getattr(new_model.model, module_name)
            param = getattr(module, param_name)
            
            if isinstance(param, torch.Tensor):  # Only apply operation if param is a Tensor
                param.data.mul_(m)
    
    return new_model


def decompose_model_into_modules(original_model, masks):
    modules = []
    for class_idx in range(10):  # For 10 classes
        module = generate_module_from_mask(original_model, masks, class_idx)
        modules.append(module)
    return modules

def save_modules_to_pth(modules, save_dir):
    for idx, module in enumerate(modules):
        torch.save(module.state_dict(), f"{save_dir}/module_{idx}.pth")

def main(model_name, model_path, mask_path, save_dir):

    original_model = load_model_from_pth(model_name,model_path)
    masks = torch.load(mask_path)
    
    modules = decompose_model_into_modules(original_model, masks)
    
    save_modules_to_pth(modules, save_dir)

if __name__ == "__main__":
    MODEL_NAME  = 'simcnn'
    MODEL_PATH = "GradSplitter_main/data/simcnn_cifar10/trained_models/estimator_1.pth"
    MASK_PATH = "GradSplitter_main/data/simcnn_cifar10/modules/estimator_1/estimator_1.pth"
    SAVE_DIR = 'GradSplitter_main/data/simcnn_cifar10/decomposed_modules/estimator_1/'
    
    main(MODEL_NAME, MODEL_PATH, MASK_PATH, SAVE_DIR)

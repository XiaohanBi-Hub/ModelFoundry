import os
from SeaM_main.src.global_config import global_config as global_config_SeaM

# For fixing invalid data type
def tc_legal(target_class_str):
    if target_class_str is None or target_class_str.strip() == '':
        target_class = -1
    else:
        try:
            target_class = int(target_class_str)
        except ValueError:
            print(f"Error: 'targetClass' value '{target_class_str}' is not a valid integer")
            target_class = -1
    return target_class


# Given name of algorithm, find the directory of it
def dir_convert(algorithm, direct_model_reuse, model_file, dataset_file,
            target_class_str, target_superclass_idx_str,lr_mask,alpha,lr_head=0.1):
    if algorithm == "SEAM":
        # dir for debug in data disk
        if os.path.exists("/data/bixh/ToolDemo_GS/SeaM_main/data"):
            algorithm_path = "/data/bixh/ToolDemo_GS/SeaM_main/data"
        else:
            algorithm_path = f"{global_config_SeaM.data_dir}/flask_project"

        if direct_model_reuse == "Binary Classification":
            file_name = f"lr_head_mask_{lr_head}_{lr_mask}_alpha_{alpha}.pth"
            model_reuse_path = f"/binary_classification/{model_file}_{dataset_file}/tc_{target_class_str}/"
        elif direct_model_reuse == "Multi-Class Classification":
            file_name = f"lr_head_mask_{lr_head}_{lr_mask}_alpha_{alpha}.pth"
            model_reuse_path = f"/multi_class_classification/{model_file}_{dataset_file}/predefined/tsc_{target_superclass_idx_str}/"
        elif direct_model_reuse == "Defect Inheritance":
            file_name = "step_3_seam_ft.pth"
            model_reuse_path = f"/defect_inheritance/seam_ft/resnet18_mit67_dropout_0.0/lr_mask_{lr_mask}_alpha_{alpha}_thres_0.6/"
        return f"{algorithm_path}{model_reuse_path}",file_name
    # =====================================TO BE CONTINUED============================
    elif algorithm == "GradSplitter":
        # algorithm_path = f"{global_config_Grad.data_dir}"
        algorithm_path = "/data/bixh/ToolDemo_GS/GradSplitter_main/data"
        if "cnn" in model_file:
            model_reuse_path = f"/{model_file}_{dataset_file}/modules/estimator_3/"
            file_name = "estimator_3.pth"
        else:
            model_reuse_path = f"/{model_file}_{dataset_file}/modules/estimator_None/"
            file_name = "estimator_None.pth"

        return f"{algorithm_path}{model_reuse_path}",file_name

def get_data(data):
    
    return

# from SeaM_main.src.binary_class.run_calculate_flop import run_calculate_flop
from SeaM_main.src.multi_class.run_model_reengineering import run_model_reengineering_mc
from SeaM_main.src.multi_class.run_calculate_flop import run_calculate_flop_mc
from SeaM_main.src.binary_class.run_calculate_flop import run_calculate_flop_bc
from SeaM_main.src.binary_class.run_model_reengineering import run_model_reengineering_bc
# from SeaM_main.src.binary_class.run_calculate_time_cost import run_calculate_time_cost_bc
from SeaM_main.src.defect_inherit.run_reengineering_finetune import run_reengineering_finetune
from SeaM_main.src.defect_inherit.run_eval_robustness import run_eval_robustness
from SeaM_main.src.defect_inherit.run_standard_finetune import run_standard_finetune

from SeaM_main.src.global_config import global_config as global_config_SeaM
from GradSplitter_main.src.global_configure import global_config as global_config_Grad

from GradSplitter_main.src.script.run_train import run_train_script
from GradSplitter_main.src.script.run_splitter import run_splitter_script
from GradSplitter_main.src.script.run_module_reuse_for_accurate_model import run_ensemble_modules_script
from GradSplitter_main.src.script.run_module_reuse_for_new_task import run_reuse_modules_script
import os
from GradSplitter_main.src.train import run_train
from GradSplitter_main.src.grad_splitter import run_grad_splitter
from GradSplitter_main.src.global_configure import global_config as global_config_Grad
from GradSplitter_main.src.script.run_evaluate_modules import run_evaluate_modules_script


# model_file="resnet20"
model_file = "simcnn"
# model_file = "resnet34"
# model_file="vgg16_bn"
# dataset_file="imagenet"
dataset_file="cifar10"

target_class=0
target_superclass_idx = 0
learning_rate=0.01
# learning_rate=0.1
alpha=1.0
# alpha=2
n_epochs=300

if __name__ == "__main__":
    def callback(**kwargs):
        # socketio.emit('reuse_result', f'Best acc: {acc * 100:.2f}%')
        messages = {
            'acc':'Best Acc:{:.2%}',
            'module_acc':'{}',
            'model_acc':'{}',
            'avg_model_acc':'Average pretrained model Acc:{:.2%}',
        }
        for key, message in messages.items():
            if key in kwargs:
                print('reuse_result', message.format(kwargs[key]))

    # for target_class in range(0,10):
    #     print(f"target_class:{target_class}")

        # run_model_reengineering_bc(model=model_file, dataset=dataset_file, 
        #                         target_class=target_class, lr_mask=learning_rate, 
        #                         alpha=alpha, n_epochs=n_epochs)

        # run_model_reengineering_mc(model=model_file, dataset=dataset_file, 
        #             target_superclass_idx=target_superclass_idx,
        #             lr_mask=learning_rate, alpha=alpha)
    run_evaluate_modules_script(model=model_file,dataset=dataset_file,callback=callback)
    # run_calculate_flop_bc(model=model_file, dataset=dataset_file, 
    #                     target_class=target_class, lr_mask=learning_rate, alpha=alpha, 
    #                     callback="debug") 
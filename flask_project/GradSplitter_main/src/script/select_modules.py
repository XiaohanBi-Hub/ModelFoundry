import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GradSplitter_main.src.utils.configure_loader import load_configure

# After modularization, select the modules.
# First, considering the accuracy, the loss of accuracy should less than 1%.
# Then, considering the number of kernels.
# model = 'simcnn'
# dataset = 'cifar10'

best_epoch_vgg_resnet={
      # "vgg16_bn": 138,
      "vgg16_bn": 48,
      "resnet20": 19,
}

def run_select_modules_script(model,dataset,lr_head = 0.01,
                              lr_modularity = 0.001,alpha = 0.1,
                              batch_size = 64):
      # Alpha for the weighted sum of loss1 and loss2
      estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]
      best_epoch = [133, 111, 133, 111, 124, 99, 144, 114, 78, 86]  # simcnn_cifar10
      configs = load_configure(model, dataset)
      if model in ['simcnn', 'rescnn', 'incecnn']:
            for i, epoch in enumerate(best_epoch):
                  idx = estimator_indices[i]
                  configs.set_estimator_idx(idx)
                  module_save_dir = f'{configs.module_save_dir}/lr_{lr_head}_{lr_modularity}_alpha_{alpha}'

                  cmd = f'cp {module_save_dir}/epoch_{epoch}.pth ' \
                        f'{configs.module_save_dir}/estimator_{idx}.pth'
                  os.system(cmd)
                  print(cmd)
      else:
            # estimator = None
            epoch=best_epoch_vgg_resnet[model]
            configs.set_estimator_idx(None)
            module_save_dir = f'{configs.module_save_dir}/lr_{lr_head}_{lr_modularity}_alpha_{alpha}'
            cmd = f'cp {module_save_dir}/epoch_{epoch}.pth ' \
                        f'{configs.module_save_dir}/estimator_None.pth'
            os.system(cmd)
            print(cmd)

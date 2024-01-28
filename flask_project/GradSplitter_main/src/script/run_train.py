import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)
from GradSplitter_main.src.train import run_train
# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.
estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]

# estimator_indices = [0, 1, 2, 3, 5, 8, 9, 10, 11, 13] # simcnn_svhn
# estimator_indices = [1, 3, 4, 9, 10, 11, 12, 14, 15, 16]  # rescnn_cifar
# estimator_indices = [0, 2, 3, 5, 6, 7, 9, 10, 11, 12]  # rescnn_svhn
# estimator_indices = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12]  # incecnn_cifar
# estimator_indices = [2, 3, 5, 6, 7, 9, 10, 11, 12, 13]  # incecnn_svhn

# train CNN models
# model = 'simcnn'
# dataset = 'cifar10'
# for i, idx in enumerate(estimator_indices):
#     cmd = f'python -u ../train.py ' \
#           f'--model {model} --dataset {dataset} \
#             --execute train_estimator --estimator_idx {idx}'
#     print(cmd)
#     os.system(cmd)

# model = 'simcnn'
# dataset = 'cifar10'
# execute = 'train_estimator'

# if __name__ == '__main__':
def run_train_script(model,dataset,execute,
                     estimator_indices=estimator_indices):
      for i, idx in enumerate(estimator_indices):
            run_train(model,dataset,execute,estimator_idx = idx)

import os
import sys
# sys.path.append("D:/ToolDemo_GS/flask_project")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GradSplitter_main.src.grad_splitter import run_grad_splitter
# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.
estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]

model = 'simcnn'
dataset = 'cifar10'

# for i, estimator_idx in enumerate(estimator_indices):
#     cmd = f'python -u ../grad_splitter.py --model {model} --dataset {dataset} ' \
#           f'--estimator_idx {estimator_idx} > {model}_{dataset}_estimator_{estimator_idx}.log'
#     print(cmd)
#     os.system(cmd)
def run_splitter_script(model,dataset,callback,estimator_indices=estimator_indices,get_epochs="debug"):
      if model in ['simcnn','incecnn','rescnn']:
            for i, estimator_idx in enumerate(estimator_indices):
                  run_grad_splitter(model=model,dataset=dataset,callback=callback,\
                                    estimator_idx=estimator_idx,get_epochs=get_epochs)
      else:
            run_grad_splitter(model=model,dataset=dataset,callback=callback,\
                              get_epochs=get_epochs, estimator_idx=None, is_dirichlet=False)
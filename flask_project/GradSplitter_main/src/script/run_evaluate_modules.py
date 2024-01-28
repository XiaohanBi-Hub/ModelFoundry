import sys
sys.path.append("D:/ToolDemo_GS/flask_project")
from GradSplitter_main.src.experiments.evaluate_modules import run_evaluate_modules
# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.
estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]

# parallel
# model = 'simcnn'
# dataset = 'cifar10'
# for i, estimator_idx in enumerate(estimator_indices):
#     cmd = f'python -u ../experiments/evaluate_modules.py ' \
#           f'--model {model} --dataset {dataset} --estimator_idx {estimator_idx} ' \
#           f'> ./eval_{model}_{dataset}_estimator_{estimator_idx}.log'
#     print(cmd)
#     os.system(cmd)
##################################################################

# model = 'simcnn'
# dataset = 'cifar10'

def run_evaluate_modules_script(model,dataset,estimator_indices=estimator_indices,callback="debug"):
      if model in ['simcnn', 'rescnn', 'incecnn']:
            sum_model_acc = 0
            for i, estimator_idx in enumerate(estimator_indices):
                  model_acc = run_evaluate_modules(model,dataset,estimator_idx=estimator_idx,callback=callback)
                  sum_model_acc += model_acc
            avg_model_acc = sum_model_acc/len(estimator_indices)
            callback(avg_model_acc=avg_model_acc)

      else:
            model_acc = run_evaluate_modules(model,dataset,estimator_idx=None,callback=callback)
            callback(avg_model_acc=model_acc)


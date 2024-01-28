import argparse
import sys
import time
import torch
# sys.path.append('')
# sys.path.append('..')
from GradSplitter_main.src.utils.configure_loader import load_configure
from GradSplitter_main.src.utils.model_loader import load_trained_model
from GradSplitter_main.src.utils.module_tools import load_module, module_predict
from GradSplitter_main.src.utils.dataset_loader import get_dataset_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_ensemble_modules(modules, dataset):
    modules_outputs = []
    data_labels = None
    for each_module in modules:
        outputs, labels = module_predict(each_module, dataset)
        modules_outputs.append(outputs)
        if data_labels is None:
            data_labels = labels
        # else:
            # assert (data_labels == labels).all()
    modules_outputs = torch.cat(modules_outputs, dim=1)
    final_pred = torch.argmax(modules_outputs, dim=1)
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))
    return acc


def main_func(args,estimator_indices):
    configs = load_configure(args.model, args.dataset)
    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    _, test_dataset = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None)

    # load modules
    modules = []
    for target_class, estimator_idx in enumerate(estimator_indices):
        configs = load_configure(args.model, args.dataset)
        configs.set_estimator_idx(estimator_idx)
        trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
        module_path = configs.best_module_path
        each_module = load_module(module_path, trained_model, target_class)
        modules.append(each_module)
        trained_model = None

    # evaluate ensemble modules.
    acc = evaluate_ensemble_modules(modules, test_dataset)
    print(f"Ensemble Modules Test Accuracy: {acc * 100:.2f}%")
    return acc

def get_args(model,dataset,estimator_indices):
    args = argparse.Namespace()
    args.model = model
    args.dataset = dataset
    args.estimator_indices = str(estimator_indices)
    return args

def run_ensemble_modules(model,dataset,estimator_indices):
    args = get_args(model,dataset,estimator_indices)
    print(args)
    estimator_indices = [int(idx) for idx in args.estimator_indices.split(',')]
    acc = main_func(args,estimator_indices)
    return acc

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'])
#     parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
#     parser.add_argument('--estimator_indices', type=str,
#                         help='e.g. 9,6,0,4,1,4,7,7,0,7'
#                              'This means that modules for class_0, class_1, ..., class_9 come from '
#                              'estimator_9, estimator_6, ..., estimator_7, respectively.')
#     args = parser.parse_args()
#     print(args)
#     # configs for module ensemble.  set after evaluate all modules.
#     # for instance, modules for class_0, class_1, ..., class_9 come from estimator_0, estimator_6, ..., estimator_7, respectively.
#     estimator_indices = [int(idx) for idx in args.estimator_indices.split(',')]
#     main_func()

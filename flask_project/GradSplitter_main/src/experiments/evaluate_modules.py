import argparse
import sys
import torch
# sys.path.append('')
# sys.path.append('..')
from GradSplitter_main.src.utils.configure_loader import load_configure
from GradSplitter_main.src.utils.model_loader import load_trained_model
from GradSplitter_main.src.utils.module_tools import load_module, module_predict
from GradSplitter_main.src.utils.dataset_loader import get_dataset_loader
from GradSplitter_main.src.train import test as _test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def evaluate_module_f1(module, dataset, target_class):
    outputs, labels = module_predict(module, dataset)
    predicts = (outputs > 0.5).int().squeeze(-1)
    labels = (labels == target_class).int()

    precision = torch.sum(predicts * labels) / torch.sum(predicts)
    recall = torch.sum(predicts * labels) / torch.sum(labels)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# 后加的，评估整个模型
def eval_model(model,load_dataset,configs,is_dirichlet=True):
    if is_dirichlet:
        print("is_diri train l100")
    dataset_dir = configs.dataset_dir
    _, test_loader = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None, is_dirichlet=is_dirichlet)

    model.eval()
    test_acc = _test(model, test_loader)
    return test_acc

def main_func(args,callback='debug'):
    estimator_idx = args.estimator_idx
    print(f'Estimator {estimator_idx}')
    print('-' * 80)

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)

    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    module_path = configs.best_module_path

    model_acc = eval_model(trained_model,load_dataset,configs)
    print(f'Model acc: {model_acc:.2%}')
    if callback != 'debug':
        callback(model_acc=f'Model acc: {model_acc:.2%}')

    is_dirichlet = True
    if estimator_idx is None:
        is_dirichlet = False

    # evaluate each module
    for i in range(configs.num_classes):
        module = load_module(module_path, trained_model, i)
        module_eval_dataset, _ = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None, is_dirichlet=is_dirichlet)
        result = evaluate_module_f1(module, module_eval_dataset, i)
        # print(f'{result:.4f}')
        print(f'Module{i} acc: {result:.4f}')
        if callback != 'debug':
            callback(module_acc=f'Module{i} acc: {result:.2%}')
    return model_acc

def get_args(model,dataset,estimator_idx):
    args = argparse.Namespace()
    args.model = model
    args.dataset = dataset
    args.estimator_idx = None
    if estimator_idx is not None:
        args.estimator_idx = int(estimator_idx)
    return args

def run_evaluate_modules(model,dataset,estimator_idx,callback='debug'):
    args = get_args(model,dataset,estimator_idx)
    # args = args.parse_args()
    print(args)
    model_acc = main_func(args,callback=callback)
    return model_acc

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'])
    # parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    # parser.add_argument('--estimator_idx', type=int)
    # args = parser.parse_args()
    # print(args)
    # main_func()

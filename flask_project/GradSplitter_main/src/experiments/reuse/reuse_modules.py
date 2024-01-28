import argparse
import numpy as np
import torch

# sys.path.append('..')
# sys.path.append('../..')
from GradSplitter_main.src.utils.configure_loader import load_configure
from GradSplitter_main.src.utils.model_loader import load_trained_model
from GradSplitter_main.src.utils.module_tools import load_module, module_predict
from GradSplitter_main.src.utils.dataset_loader_for_reuse_exp import load_cifar10_svhn_mixed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_estimator_indices(model_name):
    if model_name == 'simcnn':
        cifar = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]
        svhn = [0, 1, 2, 3, 5, 8, 9, 10, 11, 13]
    elif model_name == 'rescnn':
        cifar = [1, 3, 4, 9, 10, 11, 12, 14, 15, 16]
        svhn = [0, 2, 3, 5, 6, 7, 9, 10, 11, 12]
    elif model_name == 'incecnn':
        cifar = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12]
        svhn = [2, 3, 5, 6, 7, 9, 10, 11, 12, 13]
    else:
        raise ValueError
    return cifar, svhn


def load_target_module(model_name, dataset_name, estimator_idx, target_class):
    configs = load_configure(model_name, dataset_name)
    configs.set_estimator_idx(estimator_idx)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    module_path = configs.best_module_path
    module = load_module(module_path, trained_model, target_class)
    return module


def evaluate_reusing(modules, dataset):
    modules_outputs = []
    data_labels = None
    for each_module in modules:
        outputs, labels = module_predict(each_module, dataset)
        modules_outputs.append(outputs)
        if data_labels is None:
            data_labels = labels
        else:
            assert (data_labels == labels).all()

    # modules_outputs_1 = torch.cat(modules_outputs[:5], dim=1)
    # modules_outputs_1 = torch.softmax(modules_outputs_1, dim=1)
    #
    # modules_outputs_2 = torch.cat(modules_outputs[5:], dim=1)
    # modules_outputs_2 = torch.softmax(modules_outputs_2, dim=1)
    #
    # modules_outputs = torch.cat([modules_outputs_1, modules_outputs_2], dim=1)

    modules_outputs = torch.cat(modules_outputs, dim=1)

    final_pred = torch.argmax(modules_outputs, dim=1)
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))

    return acc


def evaluate_module_f1(module, dataset, target_class):
    outputs, labels = module_predict(module, dataset)
    predicts = (outputs > 0.5).int().squeeze(-1)
    labels = (labels == target_class).int()

    precision = torch.sum(predicts * labels) / torch.sum(predicts)
    recall = torch.sum(predicts * labels) / torch.sum(labels)
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1.cpu())


def select_modules(args,class_cifar, class_svhn, val_loader):
    cifar_estimator_indices, svhn_estimator_indices = load_estimator_indices(args.model)

    f1_cifar_log = dict()
    select_cifar_est = []
    for c in class_cifar:
        estimator_f1 = []
        for e_idx in cifar_estimator_indices:
            module = load_target_module(args.model, 'cifar10', e_idx, c)
            # NOTE: for module evaluation, label is different
            f1 = evaluate_module_f1(module, val_loader, class_cifar.index(c))
            estimator_f1.append(f1)
        select_est = cifar_estimator_indices[np.nanargmax((np.array(estimator_f1)))]
        select_cifar_est.append(select_est)
        f1_cifar_log[f'c{c}'] = estimator_f1

    f1_svhn_log = dict()
    select_svhn_est = []
    for c in class_svhn:
        estimator_f1 = []
        for e_idx in svhn_estimator_indices:
            module = load_target_module(args.model, 'svhn', e_idx, c)
            # NOTE: for module evaluation, label is different
            f1 = evaluate_module_f1(module, val_loader, class_svhn.index(c) + len(class_cifar))
            estimator_f1.append(f1)
        select_est = svhn_estimator_indices[np.nanargmax((np.array(estimator_f1)))]
        select_svhn_est.append(select_est)
        f1_svhn_log[f's{c}'] = estimator_f1
    return select_cifar_est, select_svhn_est, f1_cifar_log, f1_svhn_log


def main_func(args):
    configs = load_configure(args.model, 'cifar10')
    dataset_dir_cifar = configs.dataset_dir

    configs = load_configure(args.model, 'svhn')
    dataset_dir_svhn = configs.dataset_dir

    class_cifar = [int(i) for i in args.class_cifar.split(',')]
    class_svhn = [int(i) for i in args.class_svhn.split(',')]

    val_loader, test_loader = load_cifar10_svhn_mixed(dataset_dir_cifar, class_cifar, dataset_dir_svhn, class_svhn,
                                                      is_train=False, shuffle_seed_cifar=None, shuffle_seed_svhn=None,
                                                      is_random=None)

    # select module according to modules evaluation results
    select_cifar_est, select_svhn_est, f1_cifar_log, f1_svhn_log = \
        select_modules(args, class_cifar, class_svhn, val_loader)
    print(f'select modules for cifar: estimator_{select_cifar_est} for class {class_cifar}')
    print(f'select modules for svhn : estimator_{select_svhn_est} for class {class_svhn}')

    # load select modules
    cifar_modules = [load_target_module(args.model, 'cifar10', select_cifar_est[i], c)
                     for i, c in enumerate(class_cifar)]
    svhn_modules = [load_target_module(args.model, 'svhn', select_svhn_est[i], c)
                    for i, c in enumerate(class_svhn)]

    modules = cifar_modules + svhn_modules
    acc = evaluate_reusing(modules, test_loader)
    print(f'ACC: {acc * 100:.2f}')
    return acc

def get_args(model,class_cifar='0',class_svhn='0'):
    args = argparse.Namespace()
    args.model = model
    args.class_cifar = str(class_cifar)
    args.class_svhn = str(class_svhn)
    return args

def run_reuse_modules(model,class_cifar,class_svhn):
    args = get_args(model,class_cifar,class_svhn)
    acc = main_func(args)
    return acc


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'], required=True)
#     parser.add_argument('--class_cifar', type=str, default='0')
#     parser.add_argument('--class_svhn', type=str, default='0')
#     args = parser.parse_args()
#     main()

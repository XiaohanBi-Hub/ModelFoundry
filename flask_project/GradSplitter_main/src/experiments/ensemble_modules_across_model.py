import argparse
import sys
import torch
sys.path.append('')
sys.path.append('..')
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_ensemble_modules(modules, dataset):
    modules_outputs = []
    data_labels = None
    for each_module in modules:
        outputs, labels = module_predict(each_module, dataset)
        modules_outputs.append(outputs)
        if data_labels is None:
            data_labels = labels
        else:
            assert (data_labels == labels).all()
    modules_outputs = torch.cat(modules_outputs, dim=1)
    final_pred = torch.argmax(modules_outputs, dim=1)
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))
    return acc


def main():
    simcnn_configs = load_configure('simcnn', args.dataset)
    rescnn_configs = load_configure('rescnn', args.dataset)
    dataset_dir = simcnn_configs.dataset_dir

    load_dataset = get_dataset_loader(args.dataset)
    _, test_dataset = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None)

    # load modules
    modules = []
    for target_class, idx_pair in enumerate(estimator_indices):
        model_name = ('simcnn', 'rescnn', 'incecnn')[idx_pair[0]]
        estimator_idx = idx_pair[1]
        configs = load_configure(model_name, args.dataset)
        configs.set_estimator_idx(estimator_idx)
        trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
        module_path = configs.best_module_path
        each_module = load_module(module_path, trained_model, target_class)
        modules.append(each_module)

    # evaluate ensemble modules.
    acc = evaluate_ensemble_modules(modules, test_dataset)
    print(f"Ensemble Modules Test Accuracy: {acc * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    parser.add_argument('--estimator_indices', type=str,
                        help='model_idx,estimator_idx|model_idx,estimator_idx...'
                             'model_idx: 0 is SinCNN, 1 is ResCNN, and 2 is InceCNN.'
                             "e.g. '0,0|0,8|1,2|1,4|0,8|1,7|0,5|1,1|1,2|1,4'"
                             'This means that modules for class_0, class_1, ..., class_9 come from '
                             'estimator_0 of simcnn, estimator_8 of simcnn, estimator_2 of rescnn, ..., '
                             'estimator_4 of rescnn respectively.')
    args = parser.parse_args()
    print(args)
    # configs for module ensemble.  set after evaluate all modules.
    # for instance, modules for class_0, class_1, ..., class_9 come from estimator_0, estimator_6, ..., estimator_7, respectively.
    estimator_indices = [(int(idx_pair.split(',')[0]), int(idx_pair.split(',')[1]))
                         for idx_pair in args.estimator_indices.split('|')]
    main()

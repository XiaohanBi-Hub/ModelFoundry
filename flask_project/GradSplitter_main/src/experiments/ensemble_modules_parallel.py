import argparse
import shutil
import sys
import torch
import time
import os
sys.path.append('')
sys.path.append('..')
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPU_LIST = """""".split()


def main():
    for target_class, estimator_idx in enumerate(estimator_indices):
        cmd = f'CUDA_VISIBLE_DEVICES={GPU_LIST[target_class]} python ensemble_modules_parallel_worker.py --model {args.model} ' \
              f'--dataset {args.dataset} --estimator_idx {estimator_idx} --target_class {target_class} &'
        os.system(cmd)

    configs = load_configure(args.model, args.dataset)
    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    _, test_dataset = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None)
    try:
        data_labels = torch.LongTensor(test_dataset.dataset.labels)
    except:
        data_labels = torch.LongTensor(test_dataset.dataset.targets)

    data_labels = data_labels.to(device)
    modules_outputs = []

    for target_class, _ in enumerate(estimator_indices):
        while True:
            if os.path.exists(f'./signals/target_{target_class}.signal'):
                break
            else:
                time.sleep(0.5)
        modules_outputs.append(torch.load(f'./signals/target_{target_class}.pt'))

    s_time = time.time()
    modules_outputs = torch.cat(modules_outputs, dim=1)
    final_pred = torch.argmax(modules_outputs, dim=1)
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))
    print(acc)
    e_time = time.time()
    print(f'main : {e_time - s_time:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    parser.add_argument('--estimator_indices', type=str,
                        help='e.g. 9,6,0,4,1,4,7,7,0,7'
                             'This means that modules for class_0, class_1, ..., class_9 come from '
                             'estimator_9, estimator_6, ..., estimator_7, respectively.')
    args = parser.parse_args()
    print(args)
    # configs for module ensemble.  set after evaluate all modules.
    # for instance, modules for class_0, class_1, ..., class_9 come from estimator_0, estimator_6, ..., estimator_7, respectively.
    estimator_indices = [int(idx) for idx in args.estimator_indices.split(',')]

    if os.path.exists('./signals'):
        shutil.rmtree('./signals')
    os.makedirs('./signals')

    main()

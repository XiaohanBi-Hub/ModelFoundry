import sys
sys.path.append('../..')
from datasets.load_cifar10 import load_cifar10
from datasets.load_cifar100 import load_cifar100
from datasets.load_svhn import _load_svhn_dirichlet
from GradSplitter_main.src.global_configure import global_config as grad_global_config


def load_dataset(dataset_name, is_train, shots=-1, target_class: int=-1, reorganize=False):
    if dataset_name == 'cifar10':
        dataset = load_cifar10(is_train, shots=shots, target_class=target_class, reorganize=reorganize)
    elif dataset_name == 'cifar100':
        dataset = load_cifar100(is_train, shots=shots, target_class=target_class, reorganize=reorganize)
    elif dataset_name == 'svhn':
        dataset = _load_svhn_dirichlet(dataset_dir=f"{grad_global_config.dataset_dir}/svhn",\
                                        is_train=is_train, shuffle_seed=grad_global_config.estimator_idx,\
                                        is_random=True)
    else:
        raise ValueError
    return dataset

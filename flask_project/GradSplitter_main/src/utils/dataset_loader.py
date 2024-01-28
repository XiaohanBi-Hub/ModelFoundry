import copy
import itertools
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def get_dataset_loader(dataset_name, for_modular=False):
    if dataset_name == 'cifar10':
        load_dataset = _load_cifar10
    elif dataset_name == 'svhn':
        load_dataset = _load_svhn
    else:
        raise ValueError
    return load_dataset


def _load_cifar10(dataset_dir, is_train, shuffle_seed, is_random, is_dirichlet=True, 
                            split_train_set='8:2', batch_size=64, num_workers=0, pin_memory=False):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])
        split_ratio = [int(i) / 10 for i in split_train_set.split(':')]
        assert sum(split_ratio) == 1
        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform)

        split_set_indices = []
        
        # load cifar10 dirichlet
        if is_dirichlet:
            print("is_dirichlet")
            alpha = 1
            sampled_indices = _dirichlet_sample(train.targets,
                                                n_classes=10, shuffle_seed=shuffle_seed, min_size=100, alpha=alpha)
            # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
            train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
            val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
            split_set_indices = [train_set, val_set]
        
        #load cifar10 directly
        else:
            total_indices = list(range(len(train)))
            train_set, val_set = total_indices[:int(split_ratio[0] * len(train))], total_indices[int(split_ratio[0] * len(train)):]
            split_set_indices = [train_set, val_set]

        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(train)
            each_set.targets = [each_set.targets[idx] for idx in each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader
    else:
        ratio = 0.2  # 20% test data are used to evaluate modules.
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, transform=transform)
        total_indices = list(range(len(test)))
        module_eval_set, test_set = total_indices[:int(ratio * len(test))], total_indices[int(ratio * len(test)):]
        split_set_indices = [module_eval_set, test_set]
        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(test)
            each_set.targets = [each_set.targets[idx] for idx in each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader


def _load_svhn(dataset_dir, is_train, shuffle_seed, is_random, is_dirichlet=True,  
                        split_train_set='8:2', batch_size=64, num_workers=0, pin_memory=False):
    """
        shuffle_seed: the idx of a base estimator in the ensemble model.
        split_train_set: 8 for train model, 2 for validation of modularization. then 6 for train model, 2 for val model
        """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])
        split_ratio = [int(i) / 10 for i in split_train_set.split(':')]
        assert sum(split_ratio) == 1
        train = torchvision.datasets.SVHN(root=dataset_dir, split='train', transform=transform)
        
        split_set_indices = []
        # load svhn dirichlet
        if is_dirichlet:
            alpha = 0.5
            sampled_indices = _dirichlet_sample(train.labels,
                                                n_classes=10, shuffle_seed=shuffle_seed, min_size=10, alpha=alpha)
            # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
            train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
            val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
            split_set_indices = [train_set, val_set]
        
        #load svhn directly
        else:
            total_indices = list(range(len(train)))
            train_set, val_set = total_indices[:int(split_ratio[0] * len(train))], total_indices[int(split_ratio[0] * len(train)):]
            split_set_indices = [train_set, val_set]

        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(train)
            each_set.labels = each_set.labels[each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader
    else:
        ratio = 0.2  # 20% test data are used to evaluate modules.
        test = torchvision.datasets.SVHN(root=dataset_dir, split='test', transform=transform)
        total_indices = list(range(len(test)))
        module_eval_set, test_set = total_indices[:int(ratio * len(test))], total_indices[int(ratio * len(test)):]
        split_set_indices = [module_eval_set, test_set]
        # split the train set according the split indices.
        split_set_loader = []
        for each_set_indices in split_set_indices:
            each_set = copy.deepcopy(test)
            each_set.labels = each_set.labels[each_set_indices]
            each_set.data = each_set.data[each_set_indices]
            each_set_loader = DataLoader(each_set, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader


def _dirichlet_sample(dataset_labels, n_classes, shuffle_seed, min_size, alpha):
    np.random.seed(shuffle_seed)
    while True:
        proportions = np.random.dirichlet(np.repeat(alpha, n_classes))
        proportions = proportions / np.max(proportions)
        data_idx_per_class = []
        for each_class in range(n_classes):
            target_data_idx = np.where(np.array(dataset_labels) == each_class)[0]
            target_data_idx = target_data_idx.tolist()
            np.random.shuffle(target_data_idx)
            ratio = proportions[each_class]
            data_idx_per_class.append(target_data_idx[:int(ratio * len(target_data_idx))])
        if min([len(each_class_sample) for each_class_sample in data_idx_per_class]) < min_size:
            continue
        else:
            break

    samples_idx = list(itertools.chain(*data_idx_per_class))
    np.random.shuffle(samples_idx)
    return samples_idx

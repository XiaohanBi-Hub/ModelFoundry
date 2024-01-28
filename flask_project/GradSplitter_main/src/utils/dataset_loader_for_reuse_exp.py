import copy
import itertools
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def load_cifar10_svhn_mixed(dataset_dir_cifar, class_cifar, dataset_dir_svhn, class_svhn,
                            is_train, shuffle_seed_cifar, shuffle_seed_svhn,
                            is_random, split_train_set='8:2',
                            batch_size=64, num_workers=0, pin_memory=False):
    if is_train:
        cifar_train, cifar_val = _load_cifar10_train_no_dirichlet(dataset_dir_cifar,
                                                                  is_random, split_train_set,
                                                                  batch_size, num_workers, pin_memory)
        svhn_train, svhn_val = _load_svhn_train_no_dirichlet(dataset_dir_svhn,
                                                             is_random, split_train_set,
                                                             batch_size, num_workers, pin_memory)

        # train_mix
        train_cifar = cifar_train.dataset
        train_cifar_labels, train_cifar_data = _select_data(train_cifar, class_cifar)

        train_svhn = svhn_train.dataset
        train_svhn_labels, train_svhn_data = _select_data(train_svhn, class_svhn)
        train_svhn_labels = [i + len(class_cifar) for i in train_svhn_labels]  # NOTE: here add len(class_cifar)
        train_svhn_data = train_svhn_data.transpose((0, 2, 3, 1))  # unified with CIFAR

        train_mix_data = np.concatenate([train_cifar_data, train_svhn_data], axis=0)
        train_mix_labels = train_cifar_labels + train_svhn_labels

        train_mix = cifar_train.dataset
        train_mix.data = train_mix_data
        train_mix.targets = train_mix_labels
        train_mix_loader = DataLoader(train_mix, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_memory)

        # val_mix
        val_cifar = cifar_val.dataset
        val_cifar_labels, val_cifar_data = _select_data(val_cifar, class_cifar)

        val_svhn = svhn_val.dataset
        val_svhn_labels, val_svhn_data = _select_data(val_svhn, class_svhn)
        val_svhn_labels = [i + len(class_cifar) for i in val_svhn_labels]  # NOTE: here add len(class_cifar)
        val_svhn_data = val_svhn_data.transpose((0, 2, 3, 1))  # unified with CIFAR

        val_mix_data = np.concatenate([val_cifar_data, val_svhn_data], axis=0)
        val_mix_labels = val_cifar_labels + val_svhn_labels

        val_mix = cifar_val.dataset
        val_mix.data = val_mix_data
        val_mix.targets = val_mix_labels
        val_mix_loader = DataLoader(val_mix, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return train_mix_loader, val_mix_loader
    else:
        cifar_val, cifar_test = _load_cifar10_dirichlet(dataset_dir_cifar, is_train, shuffle_seed_cifar,
                                                is_random, split_train_set,
                                                batch_size, num_workers, pin_memory)
        test_cifar = cifar_test.dataset
        test_cifar_labels, test_cifar_data = _select_data(test_cifar, class_cifar)

        val_cifar = cifar_val.dataset
        val_cifar_labels, val_cifar_data = _select_data(val_cifar, class_cifar)

        svhn_val, svhn_test = _load_svhn_dirichlet(dataset_dir_svhn, is_train, shuffle_seed_svhn,
                                            is_random, split_train_set,
                                            batch_size, num_workers, pin_memory)
        test_svhn = svhn_test.dataset
        test_svhn_labels, test_svhn_data = _select_data(test_svhn, class_svhn)
        test_svhn_labels = [i + len(class_cifar) for i in test_svhn_labels]  # NOTE: here add len(class_cifar)
        test_svhn_data = test_svhn_data.transpose((0, 2, 3, 1))  # unified with CIFAR

        val_svhn = svhn_val.dataset
        val_svhn_labels, val_svhn_data = _select_data(val_svhn, class_svhn)
        val_svhn_labels = [i + len(class_cifar) for i in val_svhn_labels]  # NOTE: here add len(class_cifar)
        val_svhn_data = val_svhn_data.transpose((0, 2, 3, 1))  # unified with CIFAR

        test_mix_data = np.concatenate([test_cifar_data, test_svhn_data], axis=0)
        test_mix_labels = test_cifar_labels + test_svhn_labels

        test_mix = cifar_test.dataset
        test_mix.data = test_mix_data
        test_mix.targets = test_mix_labels
        test_mix_loader = DataLoader(test_mix, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        val_mix_data = np.concatenate([val_cifar_data, val_svhn_data], axis=0)
        val_mix_labels = val_cifar_labels + val_svhn_labels

        val_mix = cifar_val.dataset
        val_mix.data = val_mix_data
        val_mix.targets = val_mix_labels
        val_mix_loader = DataLoader(val_mix, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        return val_mix_loader, test_mix_loader


def _select_data(dataset, labels):
    targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
    targets = np.array(targets)
    idx = np.isin(targets, labels)
    target_label = targets[idx].tolist()
    trans_label = [labels.index(i) for i in target_label]
    new_labels = trans_label
    new_data = dataset.data[idx]
    return new_labels, new_data


def _load_cifar10_train_no_dirichlet(dataset_dir, is_random, split_train_set='8:2',
                                     batch_size=64, num_workers=0, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_random:
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize])
    split_ratio = [int(i) / 10 for i in split_train_set.split(':')]
    assert sum(split_ratio) == 1
    train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform)
    sampled_indices = list(range(len(train)))

    # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
    train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
    val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
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


def _load_svhn_train_no_dirichlet(dataset_dir, is_random, split_train_set='8:2',
                                  batch_size=64, num_workers=0, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    if is_random:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize])
    split_ratio = [int(i) / 10 for i in split_train_set.split(':')]
    assert sum(split_ratio) == 1
    train = torchvision.datasets.SVHN(root=dataset_dir, split='train', transform=transform)
    sampled_indices = list(range(len(train)))

    # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
    train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
    val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
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


def _load_cifar10_dirichlet(dataset_dir, is_train, shuffle_seed, is_random, split_train_set='8:2',
                            batch_size=64, num_workers=0, pin_memory=False):
    """
    shuffle_seed: the idx of a base estimator in the ensemble model.
    split_train_set: 8 for train model, 2 for validation of modularization. then 6 for train model, 2 for val model
    """
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

        alpha = 1
        sampled_indices = _dirichlet_sample(train.targets,
                                            n_classes=10, shuffle_seed=shuffle_seed, min_size=100, alpha=alpha)

        # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
        train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
        val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
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
            each_set_loader = DataLoader(each_set, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
            split_set_loader.append(each_set_loader)
        return split_set_loader


def _load_svhn_dirichlet(dataset_dir, is_train, shuffle_seed, is_random, split_train_set='8:2',
                         batch_size=64, num_workers=0, pin_memory=False):
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

        alpha = 0.5
        sampled_indices = _dirichlet_sample(train.labels,
                                            n_classes=10, shuffle_seed=shuffle_seed, min_size=10, alpha=alpha)

        # split the indices of train set into 2 parts, including modularity_train_set and modularity_val_set
        train_set = sampled_indices[: int(split_ratio[0] * len(sampled_indices))]
        val_set = sampled_indices[int(split_ratio[0] * len(sampled_indices)):]
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

import sys
# sys.path.append('..')


def load_configure(model_name, dataset_name):
    model_dataset_name = f'{model_name}_{dataset_name}'
    print(model_dataset_name)
    if model_dataset_name == 'simcnn_cifar10':
        from GradSplitter_main.src.configures.simcnn_cifar10 import Configures
    elif model_dataset_name == 'simcnn_svhn':
        from GradSplitter_main.src.configures.simcnn_svhn import Configures
    elif model_dataset_name == 'rescnn_cifar10':
        from GradSplitter_main.src.configures.rescnn_cifar10 import Configures
    elif model_dataset_name == 'rescnn_svhn':
        from GradSplitter_main.src.configures.rescnn_svhn import Configures
    elif model_dataset_name == 'incecnn_cifar10':
        from GradSplitter_main.src.configures.incecnn_cifar10 import Configures
    elif model_dataset_name == 'incecnn_svhn':
        from GradSplitter_main.src.configures.incecnn_svhn import Configures

    elif model_dataset_name.startswith('vgg') and (model_dataset_name.endswith('_cifar10')):
        from GradSplitter_main.src.configures.vgg_cifar10 import Configures
    elif model_dataset_name.startswith('vgg') and (model_dataset_name.endswith('_svhn')):
        from GradSplitter_main.src.configures.vgg_svhn import Configures
    
    elif model_dataset_name.startswith('resnet') and (model_dataset_name.endswith('_cifar10')):
        from GradSplitter_main.src.configures.resnet_cifar10 import Configures

    else:
        raise ValueError()
    configs = Configures(model_name=model_name)
    return configs

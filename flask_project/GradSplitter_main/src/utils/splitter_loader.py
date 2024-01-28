import sys
sys.path.append('..')


def load_splitter(model_name, dataset_name):
    if dataset_name is None:
        GradSplitter = load_splitter_normal(model_name)
    else:
        raise ValueError()
    return GradSplitter


def load_splitter_normal(model_name):
    if model_name == 'simcnn':
        from GradSplitter_main.src.splitters.simcnn_splitter import GradSplitter
    elif model_name == 'rescnn':
        from GradSplitter_main.src.splitters.rescnn_splitter import GradSplitter
    elif model_name == 'incecnn':
        from GradSplitter_main.src.splitters.incecnn_splitter import GradSplitter
    elif model_name.startswith("vgg"):
        from GradSplitter_main.src.splitters.vgg_splitter import GradSplitter
    elif model_name.startswith("resnet"):
        from GradSplitter_main.src.splitters.resnet_splitter import GradSplitter
    else:
        raise ValueError()
    return GradSplitter


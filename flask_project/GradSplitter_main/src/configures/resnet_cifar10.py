from GradSplitter_main.src.global_configure import GlobalConfigures
from typing import Union, List, Dict, Any, cast

num_conv={
    "resnet20": 21, "resnet32": 33, "resnet44": 45, "resnet56": 57
}

class Configures(GlobalConfigures):
    def __init__(self, model_name):
        super(Configures, self).__init__()
        self.model_name = model_name
        self.dataset_name = 'cifar10'
        self.num_classes = 10
        self.num_conv = num_conv[model_name]

        self.workspace = f'{self.data_dir}/{self.model_name}_{self.dataset_name}'


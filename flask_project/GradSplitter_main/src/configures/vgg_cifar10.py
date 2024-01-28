from GradSplitter_main.src.global_configure import GlobalConfigures
from typing import Union, List, Dict, Any, cast

cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
num_conv={
    "vgg11": 8,
    "vgg13": 10,
    "vgg16": 13,
    "vgg19": 16,
}

class Configures(GlobalConfigures):
    def __init__(self, model_name):
        super(Configures, self).__init__()
        self.model_name = model_name
        self.dataset_name = 'cifar10'
        self.num_classes = 10
        if model_name.endswith("_bn"):
            model_name = model_name.split("_")[0]
            
        self.num_conv = num_conv[model_name]

        self.workspace = f'{self.data_dir}/{self.model_name}_{self.dataset_name}'


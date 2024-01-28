from GradSplitter_main.src.global_configure import GlobalConfigures


class Configures(GlobalConfigures):
    def __init__(self, model_name):
        super(Configures, self).__init__()
        self.model_name = 'simcnn'
        self.dataset_name = 'svhn'
        self.num_classes = 10
        self.num_conv = 13

        self.workspace = f'{self.data_dir}/{self.model_name}_{self.dataset_name}'
        self.dataset_dir = f'{self.data_dir}/dataset/svhn'

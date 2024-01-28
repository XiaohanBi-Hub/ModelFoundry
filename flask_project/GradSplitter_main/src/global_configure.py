import os
# The absolute route of app.py
# Which is "D:\ToolDemo_GS\flask_project" in my PC
BASE_DIR = os.path.dirname(os.path.abspath("app.py"))
# print(BASE_DIR)
# BASE_DIR = f"{BASE_DIR}/flask_project"

class GlobalConfigures:
    def __init__(self):
        import os
        root_dir = f'{BASE_DIR}/GradSplitter_main'  # modify the dir
        assert os.path.exists(root_dir)

        if os.path.exists(f'{root_dir}/data'):
            self.data_dir = f'{root_dir}/data'
        else:
            raise ValueError(f'{root_dir}/data does not exist.')
        self.dataset_dir = f'{self.data_dir}/dataset'
        self.trained_entire_model_name = 'entire_model.pth'
        self.estimator_idx = None

        # Set in directory configures/
        self.workspace = None
        self.best_batch_size = None
        self.best_alpha = None
        self.best_lr = None
        self.best_epoch = None

        # define after setting estimator_idx
        self.trained_model_dir = None
        self.trained_model_path = None
        self.module_save_dir = None
        self.best_module_path = None

    def set_estimator_idx(self, idx):
        self.estimator_idx = idx
        self.trained_model_dir = f'{self.workspace}/trained_models'
        self.trained_model_path = f'{self.trained_model_dir}/estimator_{self.estimator_idx}.pth'
        self.module_save_dir = f'{self.workspace}/modules/estimator_{self.estimator_idx}'
        self.best_module_path = f'{self.module_save_dir}/estimator_{self.estimator_idx}.pth'

global_config = GlobalConfigures()


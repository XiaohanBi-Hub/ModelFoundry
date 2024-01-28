import sys
import os

# print(sys.path)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append('..')
# print(sys.path)

# from global_config import GlobalConfig
try:
    from SeaM_main.src.global_config import GlobalConfig
except ModuleNotFoundError:
    from global_config import GlobalConfig

def load_config():
    config = Config()
    return config


class Config(GlobalConfig):
    def __init__(self):
        super().__init__()
        self.project_data_save_dir = f'{self.data_dir}/binary_classification'

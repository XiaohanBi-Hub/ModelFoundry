import os

# The absolute route of app.py
# Which is "D:\ToolDemo_GS\flask_project" in my PC
BASE_DIR = os.path.dirname(os.path.abspath("app.py"))
# BASE_DIR = f"{BASE_DIR}/flask_project"
# print(BASE_DIR)
class GlobalConfig:
    def __init__(self):
        
        self.BASE_DIR = BASE_DIR
        self.root_dir = f'{BASE_DIR}/SeaM_main'
        self.data_dir = f'{self.root_dir}/data'
        self.dataset_dir = f'{self.data_dir}/dataset'
        
global_config = GlobalConfig()
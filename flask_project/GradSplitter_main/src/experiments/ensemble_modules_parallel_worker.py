import argparse
import sys
import time
import torch
sys.path.append('')
sys.path.append('..')
from torchsummary import summary
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    parser.add_argument('--estimator_idx', type=str)
    parser.add_argument('--target_class', type=str)
    args = parser.parse_args()

    module_idx = int(args.estimator_idx)
    target_class = int(args.target_class)

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(module_idx)

    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    _, test_dataset = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None,
                                   num_workers=1, pin_memory=True)

    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    module_path = configs.best_module_path
    module = load_module(module_path, trained_model, target_class)

    # TEST
    summary(module, input_size=(3, 32, 32), batch_size=64)
    # summary(trained_model, input_size=(3, 32, 32), batch_size=64)
    #

    # trained_model = None

    s_time = time.time()
    with torch.no_grad():
        outputs, labels = [], []
        for batch_inputs, batch_labels in test_dataset:
            batch_inputs = batch_inputs.to(device)
            batch_output = module(batch_inputs)
            outputs.append(batch_output)
            labels.append(batch_labels.to(device))
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)

    e_time = time.time()
    print(f'target_class {target_class}: {e_time - s_time:.1f}s')

    torch.save(outputs, f'./signals/target_{target_class}.pt')

    with open(f'./signals/target_{target_class}.signal', 'w') as f:
        f.write('ok')

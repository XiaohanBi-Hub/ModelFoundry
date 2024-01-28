import argparse
import os.path
import sys
import torch
from torch.utils.data import DataLoader
# sys.path.append('../')
# sys.path.append('../..')
from SeaM_main.src.defect_inherit.models.resnet import resnet18, resnet50
from SeaM_main.src.defect_inherit.utils.dataset_loader import load_dataset
from SeaM_main.src.defect_inherit.finetuner import finetune
from SeaM_main.src.defect_inherit.config import load_config

def standard_finetune(model, train_loader, test_loader, n_epochs, lr, momentum, weight_decay, save_path):
    # finetune all parameters.
    fc_module = model.fc
    ignored_params = list(map(id, fc_module.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optim = torch.optim.SGD(
        [
            {'params': base_params},
            {'params': fc_module.parameters(), 'lr': lr * 10}
        ],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model_ft, best_acc, best_epoch = finetune(model, optim, train_loader, test_loader, n_epochs=n_epochs)
    model_ft_params = model_ft.state_dict()
    torch.save(model_ft_params, save_path)
    print()
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Acc  : {best_acc:.2%}')
    print(f'Finish Fine-tuning.\n\n')

def get_args(model, dataset, dropout=0.0, n_epochs=300, lr=0.005, momentum=0.0, weight_decay=0.0001):
    args = argparse.Namespace()
    args.model = str(model)
    args.dataset = str(dataset)
    args.dropout = float(dropout)
    args.n_epochs = int(n_epochs)
    args.lr = float(lr)
    args.momentum = float(momentum)
    args.weight_decay = float(weight_decay)
    
    return args


def run_standard_finetune(model, dataset, dropout=0.0, n_epochs=300, lr=0.005, momentum=0.0, weight_decay=0.0001):
    args = get_args(model, dataset, dropout, n_epochs, lr, momentum, weight_decay)
    print(args)
    configs = load_config()
    num_workers = 8
    pin_memory = True
    save_path = f'{configs.standard_finetune_dir}/{args.model}_{args.dataset}_dropout_{args.dropout}/model_ft.pth'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    dataset_train = load_dataset(args.dataset, is_train=True)
    dataset_test = load_dataset(args.dataset, is_train=False)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    num_classes = dataset_train.num_classes
    model = eval(args.model)(pretrained=True, dropout=args.dropout, num_classes=num_classes).to('cuda')

    print(f'\n\n## Start Fine-tuning ##\n\n')
    print(model)

    standard_finetune(model, train_loader, test_loader,
                      args.n_epochs, args.lr, args.momentum, args.weight_decay,
                      save_path)


# if __name__ == '__main__':
#    main()

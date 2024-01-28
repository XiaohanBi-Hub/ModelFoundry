import argparse
import os.path
import sys
import torch
import time
import inspect
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SeaM_main.src.multi_class.reengineer import Reengineer
from SeaM_main.src.multi_class.datasets.dataset_loader_mc import load_dataset
from SeaM_main.src.multi_class.config import load_config
from SeaM_main.src.multi_class.models.resnet20 import cifar100_resnet20 as resnet20
# from SeaM_main.src.multi_class.models.resnet50 import resnet50
from SeaM_main.src.multi_class.utils.mask_layer import maskcnn
from SeaM_main.src.multi_class.models.head_models import HeadResNet
from tqdm import tqdm

def get_args(model, dataset, superclass_type='predefined', target_superclass_idx=-1, 
             n_classes=-1, shots= -1, seed=0, n_epochs=300, lr_head=0.1, lr_mask=0.1, 
             alpha=1, early_stop=-1):
    args = argparse.Namespace()
    args.model = model
    args.dataset = dataset
    args.superclass_type = superclass_type
    args.target_superclass_idx = target_superclass_idx
    args.n_classes = n_classes
    args.shots = shots
    args.seed = seed
    args.n_epochs = n_epochs
    args.lr_head = lr_head
    args.lr_mask = lr_mask
    args.alpha = alpha
    args.early_stop = early_stop

    return args

def reengineering(model, train_loader, test_loader, lr_mask, lr_head, 
                  n_epochs, alpha, early_stop, save_path, acc_pre_model, get_epochs):

    reengineer = Reengineer(model, train_loader, test_loader, acc_pre_model)
    reengineered_model = reengineer.alter(lr_mask=lr_mask, lr_head=lr_head,
                               n_epochs=n_epochs, alpha=alpha, get_epochs=get_epochs, early_stop=early_stop)

    masks = reengineered_model.get_masks()
    module_head = reengineered_model.get_module_head()
    masks.update(module_head)
    torch.save(masks, save_path)

    # check
    model_static = model.state_dict()
    reengineered_model_static = reengineered_model.state_dict()
    for k in model_static:
        if 'mask' not in k and 'module_head' not in k:
            model_weight = model_static[k]
            reengineered_model_weight = reengineered_model_static[k]
            assert (model_weight == reengineered_model_weight).all()


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
        batch_outputs = model(batch_inputs)
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)
        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def eval_pretrained_model(args,num_workers,pin_memory):
    model = eval(args.model)(pretrained=True).to('cuda')
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, shots=args.shots,
                 superclass_type=args.superclass_type, target_superclass_idx=args.target_superclass_idx,
                 n_classes=args.n_classes, seed=args.seed, reorganize=False)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    acc = evaluate(model, test_loader)
    return acc


def main_func(args,num_workers,pin_memory,config,get_epochs):
    # if args.model == 'resnet20':
    #     assert args.dataset == 'cifar100'
    # elif args.model == 'resnet50':
    #     assert args.dataset == 'imagenet'
    # else:
    #     raise ValueError
    # print(args)
    # print(args.superclass_type)
    # print(load_dataset.__name__)
    # print(load_dataset.__module__)
    # print(inspect.getsource(load_dataset))
    dataset_train = load_dataset(dataset_name=args.dataset, is_train=True, 
                                 shots=args.shots,superclass_type=args.superclass_type, 
                                 target_superclass_idx=args.target_superclass_idx,
                                 n_classes=args.n_classes, seed=args.seed, reorganize=True)
    dataset_test = load_dataset(dataset_name=args.dataset, is_train=False, 
                                shots=args.shots,superclass_type=args.superclass_type, 
                                target_superclass_idx=args.target_superclass_idx,
                                n_classes=args.n_classes, seed=args.seed, reorganize=True)
    assert dataset_train.classes == dataset_test.classes
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f'INFO: Superclass {args.target_superclass_idx} contains {len(dataset_train.classes)} classes: {dataset_train.classes}.\n')
    if len(dataset_train.classes) == 1:
        sys.exit(0)

    # prepare reengineered model saved path.
    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/{args.superclass_type}/tsc_{args.target_superclass_idx}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.superclass_type == 'predefined':
        save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'
    elif args.superclass_type == 'random':
        # save_path = f'{save_dir}/n_classes_{args.n_classes}.pth'
        raise ValueError('Not support for now.')
    else:
        raise ValueError
    
    if args.dataset == "cifar100":
        if args.model == "resnet20":
            # 名字转变量，模型名, 这个是cifar100的预训练模型
            model = eval(args.model)(pretrained=True).to('cuda')
        else:
            raise ValueError('Cifar100 is only available on ResNet20 for now!')
    elif args.dataset == "imagenet":
        # 这是torchvision里面imagenet预训练模型
        # 调用torchvision.models
        model = eval(args.model)(pretrained=True).to('cuda')
        # model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f'{args.dataset} is not available for now!')
    
    model = maskcnn.replace(model,is_reengineering=True)
    model = HeadResNet(model, is_reengineering=True, num_classes_in_super=len(dataset_train.classes))
    model.to('cuda')

    acc_pre_model = eval_pretrained_model(args,num_workers,pin_memory)
    print(f'Pretrained Model Test Acc: {acc_pre_model:.2%}\n\n')

    s_time = time.time()
    reengineering(model, train_loader, test_loader,
                  args.lr_mask, args.lr_head, args.n_epochs, args.alpha, args.early_stop,
                  save_path, acc_pre_model, get_epochs=get_epochs)
    e_time = time.time()
    print(f'Time Elapse: {(e_time - s_time)/60:.1f} min\n')

    print(f'Pretrained Model Test Acc: {acc_pre_model:.2%}\n\n')


def run_model_reengineering_mc(model, dataset, superclass_type='predefined', 
                            target_superclass_idx=-1, n_classes=-1, shots= -1, 
                            seed=0, n_epochs=300, lr_head=0.1, lr_mask=0.1, 
                            alpha=1, early_stop=-1, get_epochs="debug"):
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(torch.cuda.is_available())
    args = get_args(model, dataset, superclass_type, target_superclass_idx, 
             n_classes, shots, seed, n_epochs, lr_head, lr_mask, alpha, early_stop)
    print(args)
    config = load_config()

    num_workers = 16
    pin_memory = True
    # model_name = args.model
    main_func(args,num_workers,pin_memory,config,get_epochs)
    
# if __name__ == '__main__':
#     args = get_args()
#     print(args)
#     config = load_config()

#     num_workers = 16
#     pin_memory = True

#     model_name = args.model
#     main()
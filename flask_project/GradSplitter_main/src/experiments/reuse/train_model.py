import argparse
import copy
import os
import torch
import sys
# sys.path.append('..')
# sys.path.append('../..')

from tqdm import tqdm
from GradSplitter_main.src.utils.configure_loader import load_configure
from GradSplitter_main.src.utils.dataset_loader_for_reuse_exp import load_cifar10_svhn_mixed
from GradSplitter_main.src.utils.model_loader import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, save_path, lr, n_epochs, early_stop):
    # global lr
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    best_acc, best_epoch = 0.0, 0
    best_model = None
    early_stop_count = 0
    optimization = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs):
        print(f'epoch {epoch}')
        print('-' * 80)

        # train
        epoch_train_loss = []
        epoch_train_acc = []
        model.train()
        for batch_inputs, batch_labels in tqdm(train_loader, ncols=100, desc='train'):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            optimization.zero_grad()
            loss = loss_func(outputs, batch_labels)
            loss.backward()
            optimization.step()
            epoch_train_loss.append(loss.detach())

            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == batch_labels)
            epoch_train_acc.append(torch.div(acc, batch_labels.shape[0]))
        print(f"train_loss: {sum(epoch_train_loss) / len(epoch_train_loss):.2f}")
        print(f"train_acc : {sum(epoch_train_acc) / len(epoch_train_acc) * 100:.2f}%")

        # val
        epoch_val_acc = []
        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in tqdm(val_loader, ncols=100, desc='valid'):
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                outputs = model(batch_inputs)
                pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(pred == batch_labels)
                epoch_val_acc.append(torch.div(acc, batch_labels.shape[0]))
            val_acc = sum(epoch_val_acc) / len(epoch_val_acc)
        print(f"val_acc   : {val_acc * 100:.2f}%")
        print()

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop and early_stop_count == 5:
                print(f'Early Stop.\n')
                break
    print(f"best_epoch: {best_epoch}")
    print(f"best_acc  : {best_acc * 100:.2f}%")
    model.load_state_dict(best_model)
    return model


def test(model, test_loader):
    epoch_acc = []
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == batch_labels)
            epoch_acc.append(torch.div(acc, batch_labels.shape[0]))
    acc = sum(epoch_acc) / len(epoch_acc) * 100
    return acc


def main_func(args,class_cifar,class_svhn,batch_size,split_train_set,\
              save_dir, lr, n_epochs, early_stop):
    str_class_cifar = ''.join([str(i) for i in class_cifar])
    str_class_svhn = ''.join([str(i) for i in class_svhn])
    save_path = f'{save_dir}/c{str_class_cifar}_s{str_class_svhn}.pth'

    configs = load_configure(args.model, 'cifar10')
    dataset_dir_cifar = configs.dataset_dir

    configs = load_configure(args.model, 'svhn')
    dataset_dir_svhn = configs.dataset_dir

    train_loader, val_loader = load_cifar10_svhn_mixed(dataset_dir_cifar, class_cifar, dataset_dir_svhn,
                                                       class_svhn,
                                                       is_train=True, shuffle_seed_cifar=None,
                                                       shuffle_seed_svhn=None,
                                                       is_random=True, split_train_set=split_train_set,
                                                       batch_size=batch_size)
    test_loader = load_cifar10_svhn_mixed(dataset_dir_cifar, class_cifar, dataset_dir_svhn, class_svhn,
                                          is_train=False, shuffle_seed_cifar=None, shuffle_seed_svhn=None,
                                          is_random=None)

    model = load_model(args.model, 2).to(device)
    model = train(model, train_loader, val_loader, save_path, lr, n_epochs, early_stop)
    model.eval()
    acc = test(model, test_loader)
    print(f"\nTest Accuracy: {acc:.2f}%")

def get_args(model,class_cifar='0',class_svhn='0',lr=0.01,batch_size=64,\
             epochs=100,early_stop='store_true',split_train_set='8:2'):
    args = argparse.Namespace()
    args.model = model
    args.class_cifar = str(class_cifar)
    args.class_svhn = str(class_svhn)
    args.lr = float(lr)
    args.batch_size = int(batch_size)
    args.epochs = int(epochs)
    args.early_stop = early_stop
    args.early_stop = str(early_stop)
    args.split_train_set = str(split_train_set)
    return args

def run_train_model(model,class_cifar='0',class_svhn='0',lr=0.01,batch_size=64,\
             epochs=100,early_stop='store_true',split_train_set='8:2'):
    args = get_args(model,class_cifar,class_svhn,lr,batch_size,epochs,early_stop,split_train_set)
    print(args)
    model_name = args.model
    class_cifar = [int(i) for i in args.class_cifar.split(',')]
    class_svhn = [int(i) for i in args.class_svhn.split(',')]

    lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.epochs
    early_stop = args.early_stop
    split_train_set = args.split_train_set

    save_dir = f'./log_trained_model/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main_func(args,class_cifar,class_svhn,batch_size,split_train_set,\
              save_dir, lr, n_epochs, early_stop)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'], required=True)
#     parser.add_argument('--class_cifar', type=str, default='0')
#     parser.add_argument('--class_svhn', type=str, default='0')
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--early_stop', action='store_true')
#     parser.add_argument('--split_train_set', type=str, default='8:2', help='train_model : validation')
#     args = parser.parse_args()
#     print(args)
#     print()

#     model_name = args.model
#     class_cifar = [int(i) for i in args.class_cifar.split(',')]
#     class_svhn = [int(i) for i in args.class_svhn.split(',')]

#     lr = args.lr
#     batch_size = args.batch_size
#     n_epochs = args.epochs
#     early_stop = args.early_stop
#     split_train_set = args.split_train_set

#     save_dir = f'./log_trained_model/{model_name}'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     main_func()

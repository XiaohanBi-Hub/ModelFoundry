import os
import sys
# sys.path.append('..')
sys.path.append("D:/ToolDemo_GS/flask_project")

from itertools import combinations
from GradSplitter_main.src.experiments.reuse.train_model import run_train_model
# if __name__ == '__main__':
#     model = ['simcnn', 'rescnn', 'incecnn'][0]
#     class_cifar_comb = list(combinations(list(range(10)), 1))
#     class_svhn_comb = list(combinations(list(range(10)), 1))

#     for class_cifar in class_cifar_comb:
#         str_class_cifar = ''.join([str(i) for i in class_cifar])
#         class_cifar = ','.join([str(i) for i in class_cifar])
#         for class_svhn in class_svhn_comb:
#             str_class_svhn = ''.join([str(i) for i in class_svhn])
#             class_svhn = ','.join([str(i) for i in class_svhn])

#             cmd = f'python -u ../experiments/reuse/train_model.py' \
#                   f' --model {model} --class_cifar {class_cifar} --class_svhn {class_svhn} --early_stop'
#             print(cmd)
#             os.system(cmd)


if __name__ == '__main__':
    model = ['simcnn', 'rescnn', 'incecnn'][0]
    class_cifar_comb = list(combinations(list(range(10)), 1))
    class_svhn_comb = list(combinations(list(range(10)), 1))

    for class_cifar in class_cifar_comb:
        str_class_cifar = ''.join([str(i) for i in class_cifar])
        class_cifar = ','.join([str(i) for i in class_cifar])
        for class_svhn in class_svhn_comb:
            str_class_svhn = ''.join([str(i) for i in class_svhn])
            class_svhn = ','.join([str(i) for i in class_svhn])

            run_train_model(model,class_cifar,class_svhn)

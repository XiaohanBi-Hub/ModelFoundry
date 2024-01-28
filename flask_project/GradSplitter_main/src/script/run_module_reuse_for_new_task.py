import os
import sys
# sys.path.append("D:/ToolDemo_GS/flask_project")
from itertools import combinations
from GradSplitter_main.src.experiments.reuse.reuse_modules import run_reuse_modules


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

#             cmd = f'python -u ../experiments/reuse/reuse_modules.py ' \
#                   f'--model {model} --class_cifar {class_cifar} --class_svhn {class_svhn}'
#             print(cmd)
#             os.system(cmd)


def run_reuse_modules_script():
    model = ['simcnn', 'rescnn', 'incecnn'][0]
    class_cifar_comb = list(combinations(list(range(10)), 1))
    class_svhn_comb = list(combinations(list(range(10)), 1))

    for class_cifar in class_cifar_comb:
        str_class_cifar = ''.join([str(i) for i in class_cifar])
        class_cifar = ','.join([str(i) for i in class_cifar])
        for class_svhn in class_svhn_comb:
            str_class_svhn = ''.join([str(i) for i in class_svhn])
            class_svhn = ','.join([str(i) for i in class_svhn])
            run_reuse_modules(model,class_cifar,class_svhn)

def run_reuse_modules_script_pair(model,class_cifar,class_svhn,callback="debug"):
    acc = run_reuse_modules(model,class_cifar,class_svhn)
    if callback != "debug":
        callback(acc)

### **The model reuse results of DNNDecomposer regarding composing more accurate model. "TM" denotes target model.**

| Dataset     | Model Name | Acc. (%) Best TM | Acc. (%) Composed Model | Improvement |
| ----------- | ---------- | ---------------- | ----------------------- | ----------- |
| CIFAR-10    | SimCNN     | 81.01            | 86.26                   | 5.25        |
| CIFAR-10    | ResCNN     | 81.88            | 85.95                   | 4.07        |
| CIFAR-10    | InceCNN    | 83.06            | 86.94                   | 3.88        |
| SVHN        | SimCNN     | 87.51            | 93.12                   | 5.61        |
| SVHN        | ResCNN     | 85.07            | 90.55                   | 5.48        |
| SVHN        | InceCNN    | 83.19            | 90.22                   | 7.03        |
| **Average** |            | **-**            | **-**                   | **5.22**    |

### The model decomposing results of DNNDecomposer regarding classification accuracy. "Acc." denotes the average test accuracy.

| Target problem             | Model Name        | Original | Module | Reduction (%) |
| -------------------------- | ----------------- | -------- | ------ | ------------- |
| Binary Classification      | VGG16-CIFAR10     | 15.25    | 0.62   | 95.93         |
|                            | VGG16-CIFAR100    | 15.29    | 1.47   | 90.39         |
|                            | ResNet20-CIFAR10  | 0.27     | 0.03   | 88.89         |
|                            | ResNet20-CIFAR100 | 0.28     | 0.03   | 89.29         |
| Multi-class Classification | ResNet20-CIFAR100 | 0.28     | 0.04   | 85.71         |
|                            | ResNet50-ImageNet | 25.50    | 2.77   | 89.16         |
| **Average**                |                   | **-**    | **-**  | **89.89**     |

### The model weights results of DNNDecomposer.

| Target problem             | Model Name        | Original | Module | Increase |
| -------------------------- | ----------------- | -------- | ------ | -------- |
| Binary Classification      | VGG16-CIFAR10     | 96.50    | 97.12  | 0.62     |
|                            | VGG16-CIFAR100    | 86.82    | 92.93  | 6.11     |
|                            | ResNet20-CIFAR10  | 95.64    | 95.81  | 0.17     |
|                            | ResNet20-CIFAR100 | 83.97    | 90.92  | 6.95     |
| Multi-class Classification | ResNet20-CIFAR100 | 68.29    | 82.76  | 14.47    |
|                            | ResNet50-ImageNet | 78.83    | 85.63  | 6.80     |
| **Average**                |                   | **-**    | **-**  | **5.85** |

Algorithm details can be found at:

SeaM:https://github.com/qibinhang/SeaM

GradSplitter:https://github.com/qibinhang/GradSplitter
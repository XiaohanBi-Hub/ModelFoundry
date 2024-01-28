import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, num_classes=10, conv_configs=None):
    if model_name == 'simcnn':
        from GradSplitter_main.src.models.simcnn import SimCNN
        model = SimCNN(num_classes=num_classes)
    elif model_name == 'rescnn':
        from GradSplitter_main.src.models.rescnn import ResCNN
        model = ResCNN(num_classes=num_classes)
    elif model_name == 'incecnn':
        from GradSplitter_main.src.models.incecnn import InceCNN
        model = InceCNN()

    # model_name = vgg11/13/16/19; vgg11_bn/13_bn/16_bn/19_bn
    # default: vgg16_bn
    elif model_name.startswith("vgg"):
        from GradSplitter_main.src.models.vgg import VGG
        model = VGG(model_name=model_name, conv_configs=conv_configs)
        
    # model_name = resnet20/resnet32/resnet44/resnet56
    # default: resnet20
    elif model_name.startswith("resnet"):
        from GradSplitter_main.src.models.resnet import ResNet
        model = ResNet(model_name=model_name, conv_configs=conv_configs)

    else:
        raise ValueError
    return model


def load_trained_model(model_name, n_classes, trained_model_path):
    model = load_model(model_name, num_classes=n_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

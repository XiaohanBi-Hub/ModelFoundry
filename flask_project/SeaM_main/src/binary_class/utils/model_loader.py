import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, num_classes=10):
    if model_name == 'simcnn':
        from SeaM_main.src.binary_class.models.simcnn import SimCNN
        model = SimCNN(num_classes=num_classes)
    elif model_name == 'rescnn':
        from SeaM_main.src.binary_class.models.rescnn import ResCNN
        model = ResCNN(num_classes=num_classes)
    elif model_name == 'incecnn':
        from SeaM_main.src.binary_class.models.incecnn import InceCNN
        model = InceCNN()
    else:
        raise ValueError
    return model


def load_trained_model(model_name, n_classes, trained_model_path):
    model = load_model(model_name, num_classes=n_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

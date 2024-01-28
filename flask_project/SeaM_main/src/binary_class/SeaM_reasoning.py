import torch
import sys
import os
import copy
from PIL import Image
from torchvision.transforms import transforms
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vgg import cifar10_vgg16_bn as cifar10_vgg16

class CIFAR10Inference:
    def __init__(self, model_path, mask_path):
        self.cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = cifar10_vgg16()

        model_params = torch.load(model_path)
        self.model.load_state_dict(model_params)
        self.model = self.model.to(self.device)
        mask_checkpoint = torch.load(mask_path)
        self.mask_checkpoint = {k: v.to(self.device) for k, v in mask_checkpoint.items()}

    def binarize(self,tensor):
        return (tensor > 0).float()
    
    def mask_model(self,model,mask_checkpoint):
        masked_model = copy.deepcopy(model)
        for name, param in masked_model.named_parameters():
            if name+"_mask" in mask_checkpoint:
                mask = mask_checkpoint[name+"_mask"]
                mask = self.binarize(mask)
                param.data = param.data * mask
        return masked_model

    def predict(self,image_path):
        self.masked_model = self.mask_model(self.model,self.mask_checkpoint)
        self.masked_model.eval()

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        img = Image.open(image_path).convert("RGB")
        transformed_image = transform(img)
        input_data = transformed_image.unsqueeze(0).to(self.device)

        if torch.cuda.is_available():
            self.masked_model = self.masked_model.cuda()
            input_data = input_data.cuda()

        with torch.no_grad():
            outputs = self.masked_model(input_data)
            
        _, predicted = outputs.max(1)
        label_num = predicted.item()
        label = self.cifar10_class[label_num]
        return label,label_num


model_path = "SeaM_main/data/trained_model/cifar10_vgg16_bn-6ee7ea24.pt" 
mask_path = "SeaM_main/data/binary_classification/vgg16_bn_cifar10/tc_3/lr_head_mask_0.1_0.01_alpha_1.0.pth"

cifar10_inference = CIFAR10Inference(model_path,mask_path)
# cifar10_inference.predict('cat.png')


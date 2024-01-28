import torch
import torch.nn as nn


class ResCNN(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super().__init__()
        self.num_classes = num_classes
        self.is_modular = True
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 128),
                            (128, 128), (128, 128),
                            (128, 256), (256, 512),
                            (512, 512), (512, 512),
                            (512, 512), (512, 512),
                            (512, 512), (512, 512)]
            self.is_modular = False
        self.conv_configs = conv_configs
        self.residual_idx = [3, 7, 11]

        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            setattr(self, f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel)
            ))

        self.dropout_12 = nn.Dropout()
        self.fc_12 = nn.Linear(conv_configs[-1][1], num_classes)

        if self.is_modular:
            # self.module_head = nn.Linear(num_classes, 1)
            module_head_dim = 1
            self.module_head = nn.Sequential(
                nn.Linear(self.num_classes, 10),
                nn.ReLU(),
                nn.Linear(10, module_head_dim),
            )

    def forward(self, x):
        out = torch.relu(self.conv_0(x))
        out = torch.relu(self.conv_1(out))
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = torch.relu(self.conv_2(out))
        out = torch.relu(self.conv_3(out)) + res

        out = torch.relu(self.conv_4(out))
        out = torch.max_pool2d(out, kernel_size=2, stride=2)
        out = torch.relu(self.conv_5(out))
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = torch.relu(self.conv_6(out))
        out = torch.relu(self.conv_7(out)) + res

        out = torch.relu(self.conv_8(out))
        out = torch.max_pool2d(out, kernel_size=2, stride=2)
        out = torch.relu(self.conv_9(out))
        out = torch.max_pool2d(out, kernel_size=2, stride=2)

        res = out
        out = torch.relu(self.conv_10(out))
        out = torch.relu(self.conv_11(out)) + res

        out = out.view(out.size(0), -1)
        out = self.dropout_12(out)
        out = self.fc_12(out)

        if self.is_modular:
            out = torch.relu(out)
            out = torch.sigmoid(self.module_head(out))

        return out

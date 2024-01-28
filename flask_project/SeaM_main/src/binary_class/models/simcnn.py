import math
import torch
import torch.nn as nn


class SimCNN(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(SimCNN, self).__init__()
        self.num_classes = num_classes
        self.is_modular = True
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 64),
                            (64, 128), (128, 128),
                            (128, 256), (256, 256), (256, 256),
                            (256, 512), (512, 512), (512, 512),
                            (512, 512), (512, 512), (512, 512)]
            self.is_modular = False
        self.conv_configs = conv_configs

        # the name of conv layer must be 'conv_*'
        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            setattr(self, f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel)
            ))

        self.dropout_13 = nn.Dropout()
        self.fc_13 = nn.Linear(conv_configs[-1][-1], 4096)
        self.dropout_14 = nn.Dropout()
        self.fc_14 = nn.Linear(4096, 4096)
        self.fc_15 = nn.Linear(4096, num_classes)

        if self.is_modular:
            module_head_dim = 1
            self.module_head = nn.Sequential(
                nn.Linear(self.num_classes, 10),
                nn.ReLU(),
                nn.Linear(10, module_head_dim),
            )

        if not self.is_modular:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x):
        y = torch.relu(self.conv_0(x))
        y = torch.relu(self.conv_1(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_2(y))
        y = torch.relu(self.conv_3(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_4(y))
        y = torch.relu(self.conv_5(y))
        y = torch.relu(self.conv_6(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_7(y))
        y = torch.relu(self.conv_8(y))
        y = torch.relu(self.conv_9(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.relu(self.conv_10(y))
        y = torch.relu(self.conv_11(y))
        y = torch.relu(self.conv_12(y))
        y = torch.max_pool2d(y, kernel_size=2, stride=2)

        y = y.view(y.size(0), -1)
        y = self.dropout_13(y)
        y = torch.relu(self.fc_13(y))

        y = self.dropout_14(y)
        y = torch.relu(self.fc_14(y))

        pred = self.fc_15(y)

        if self.is_modular:
            pred = torch.relu(pred)
            pred = torch.sigmoid(self.module_head(pred))

        return pred

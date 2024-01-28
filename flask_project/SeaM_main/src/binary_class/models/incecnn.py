import torch
import torch.nn as nn


class InceCNN(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(InceCNN, self).__init__()
        self.num_classes = num_classes
        self.is_modular = True
        if conv_configs is None:
            conv_configs = [(3, 64), (64, 128), (128, 256),
                            (256, 256), (256, 256), (256, 64),
                            (576, 512), (576, 512), (576, 64),
                            (1088, 512), (1088, 512), (1088, 64)]
            self.is_modular = False
        self.conv_configs = conv_configs
        self.kernel_sizes = [3, 3, 3,
                             1, 3, 5,
                             1, 3, 5,
                             1, 3, 5]
        for i, each_conv_config in enumerate(conv_configs):
            in_channel, out_channel = each_conv_config
            setattr(self, f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=self.kernel_sizes[i], padding=(self.kernel_sizes[i]-1)//2),
                nn.BatchNorm2d(out_channel)
            ))

        self.avg_pool_12 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_12 = nn.Dropout(0.4)
        self.fc_12 = nn.Linear(conv_configs[-3][1] + conv_configs[-2][1] + conv_configs[-1][1],
                               num_classes)

        if self.is_modular:
            # self.module_head = nn.Linear(num_classes, 1)
            module_head_dim = 1
            self.module_head = nn.Sequential(
                nn.Linear(self.num_classes, 10),
                nn.ReLU(),
                nn.Linear(10, module_head_dim),
            )

    def forward(self, x):
        x = torch.relu(self.conv_0(x))
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        out_1 = torch.relu(self.conv_3(x))
        out_2 = torch.relu(self.conv_4(x))
        out_3 = torch.relu(self.conv_5(x))
        x = torch.cat([out_1, out_2, out_3], dim=1)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        out_1 = torch.relu(self.conv_6(x))
        out_2 = torch.relu(self.conv_7(x))
        out_3 = torch.relu(self.conv_8(x))
        x = torch.cat([out_1, out_2, out_3], dim=1)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        out_1 = torch.relu(self.conv_9(x))
        out_2 = torch.relu(self.conv_10(x))
        out_3 = torch.relu(self.conv_11(x))
        x = torch.cat([out_1, out_2, out_3], dim=1)

        x = self.avg_pool_12(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_12(x)
        out = self.fc_12(x)

        if self.is_modular:
            out = torch.relu(out)
            out = torch.sigmoid(self.module_head(out))

        return out


# def test_model():
#     model = InceCNN()
#     print(model)
#     inputs = torch.randn((2, 3, 32, 32))
#     out = model(inputs)
#
#
# test_model()

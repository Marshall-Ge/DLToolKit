from dltoolkit.utils.utils import MODEL_REGISTRY
import torch.nn as nn
import torch.nn.functional as F

@MODEL_REGISTRY.register()
class CovNet_MNIST(nn.Module):
    def __init__(self, config):
        super(CovNet_MNIST, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(int(config.model.param.get('in_channel')), 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, config.data.num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
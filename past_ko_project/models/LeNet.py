import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        self.num_classes=config["data_param"]["num_classes"]
        self.inplanes = config["data_param"]["inplanes"]
        self.size = config["data_param"]["size"]
        self.resize = None
        if self.size == 32:
            self.conv1 = nn.Conv2d(self.inplanes, 6, 5)
        elif self.size == 28:
            self.conv1 = nn.Conv2d(self.inplanes, 6, 5, padding=2, padding_mode='reflect')
        else:
            self.resize = transforms.Resize((32, 32))
            self.conv1 = nn.Conv2d(self.inplanes, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = x.reshape(-1, self.inplanes, self.size, self.size)
        if self.resize:
            x = self.resize(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

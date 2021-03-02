from torch import nn
from torchvision import models
from base import ImageClassificationBase


class Resnet18ImageClassification(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)


class Resnet34ImageClassification(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)


class Resnet50ImageClassification(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)


class Resnet101ImageClassification(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet101(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)


class Resnet152ImageClassification(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet152(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)


import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet34

class ResNetFeatureExtractor(nn.Module):
    """
    Defines Base ResNet-50 feature extractor
    """
    def __init__(self, pretrained=True):
        """
        ---------
        Arguments
        ---------
        pretrained : bool (default=True)
            boolean to control whether to use a pretrained resnet model or not
        """
        super().__init__()
        self.resnet34 = resnet34(pretrained=pretrained)

    def forward(self, x):
        self.block1 = self.resnet34.conv1(x)
        self.block1 = self.resnet34.bn1(self.block1)
        self.block1 = self.resnet34.relu(self.block1)   # [64, H/2, W/2]

        self.block2 = self.resnet34.maxpool(self.block1)
        self.block2 = self.resnet34.layer1(self.block2)  # [64, H/4, W/4]
        self.block3 = self.resnet34.layer2(self.block2)  # [128, H/8, W/8]
        self.block4 = self.resnet34.layer3(self.block3)  # [256, H/16, W/16]
        self.block5 = self.resnet34.layer4(self.block4)  # [512, H/32, W/32]
        return self.block5

class ResNetImageClassififer(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.resnet_34 = ResNetFeatureExtractor(pretrained=pretrained)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_reduction_1 = nn.Conv2d(512, 256, 1, padding="same")
        self.conv_reduction_2 = nn.Conv2d(256, 128, 1, padding="same")
        self.elu = nn.ELU(inplace=True)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        conv_features = self.resnet_34(x)
        conv_features = self.conv_reduction_1(conv_features)
        conv_features = self.elu(conv_features)
        conv_features = self.conv_reduction_2(conv_features)
        conv_features = self.elu(conv_features)

        dense_features = self.avg_pool(conv_features)
        dense_features = torch.flatten(dense_features, 1)

        logits = self.linear(dense_features)
        return logits

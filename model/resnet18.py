import torch
import torch.nn as nn
import torch.nn.functional as F



# class StochasticClassifier(nn.Module):
#     def __init__(self, num_features, num_classes, temp):
#         super().__init__()
#         self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
#         self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))  # each rotation have individual variance here
#         self.temp = temp
#
#     def forward(self, x, stochastic=True, test=False):
#         mu = self.mu
#         sigma = self.sigma
#         sigma = F.softplus(sigma - 4)  # when sigma=0, softplus(sigma-4)=0.0181
#
#         weight = sigma * torch.randn_like(mu) + mu
#         # weight = F.normalize(weight, p=2, dim=1)
#         # x = F.normalize(x, p=2, dim=1)
#         score = F.linear(x, weight)
#         # score = score * self.temp
#         return score


class CosClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
    def forward(self, x):
        weight = self.mu
        score = F.linear(F.normalize(x, p=2, dim=1), F.normalize(weight, p=2, dim=1))
        score = score * 16
        return score


class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))  # each rotation have individual variance here
        self.temp = temp

    def forward(self, x, stochastic=True, test=False):
        mu = self.mu
        sigma = self.sigma
        sigma = F.softplus(sigma - 4)
        weight = sigma * torch.randn_like(mu) + mu
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        score = F.linear(x, weight)
        if test:
            k = 100
            for _ in range(100):
                weight = sigma * torch.randn_like(mu) + mu
                weight = F.normalize(weight, p=2, dim=1)
                x = F.normalize(x, p=2, dim=1)
                score += F.linear(x, weight)
            score = score/(k+1)
        score = score * 16
        return score


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.mc_dropout = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.cosfc = CosClassifier(num_features=512*block.expansion, num_classes=num_classes)
        # self.sfc = StochasticClassifier(num_features=512*block.expansion, num_classes=num_classes, temp=16)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_output=False, stochastic=False, test=False, cos=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        dim = out.size()[-1]
        out = F.avg_pool2d(out, dim)
        out = out.view(out.size(0), -1)
        if stochastic:
            y = self.sfc(out, test=test)
        elif cos:
            y = self.cosfc(out)
        else:
            y = self.linear(out)
        if feature_output:
            return y, out
        else:
            return y

    def forward_threshold(self, x, threshold=1e10):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x_original = x
        x = x.clip(max=threshold)
        last_feature = x.view(x.size(0), -1)
        x = self.fc(last_feature)
        last_feature_original = x_original.view(x_original.size(0), -1)
        x_original = self.fc(last_feature_original)
        return x, x_original


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

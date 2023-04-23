import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, feature_output=False, stochastic=False, test=False, cos=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        dim = out.size()[-1]
        out = F.avg_pool2d(out, dim)
        last_feature = out.view(-1, self.nChannels)
        x = self.fc(last_feature)
        if feature_output:
            return x, last_feature
        else:
            return x

    def mixup_process(self, x_in, x_out, alpha=10.0):
        if x_in.size()[0] != x_out.size()[0]:
            length = min(x_in.size()[0], x_out.size()[0])
            x_in = x_in[:length]
            x_out = x_out[:length]
        lam = np.random.beta(alpha, alpha)
        x_oe = lam * x_in + (1 - lam) * x_out
        return x_oe, lam

    def forward_manifold(self, x, length_size=0):
        layer_mix = np.random.randint(0, 3)
        lam_return = 0
        if layer_mix == 0:
            x_in = x[:length_size]
            x_out = x[length_size:]
            x_oe, lam = self.mixup_process(x_in, x_out)
            lam_return = lam
            x = torch.cat([x_in, x_oe], dim=0)
        x = self.conv1(x)
        x = self.block1(x)
        if layer_mix == 1:
            x_in = x[:length_size]
            x_out = x[length_size:]
            x_oe, lam = self.mixup_process(x_in, x_out)
            lam_return = lam
            x = torch.cat([x_in, x_oe], dim=0)
        x = self.block2(x)
        if layer_mix == 2:
            x_in = x[:length_size]
            x_out = x[length_size:]
            x_oe, lam = self.mixup_process(x_in, x_out)
            lam_return = lam
            x = torch.cat([x_in, x_oe], dim=0)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        dim = x.size()[-1]
        x = F.avg_pool2d(x, dim)
        x = x.view(-1, self.nChannels)
        x = self.fc(x)
        return x, lam_return


    def forward_threshold(self, x, threshold=1e10):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        dim = out.size()[-1]
        out = F.avg_pool2d(out, dim)
        out_original = out
        out = out.clip(max=threshold)
        out = out.view(-1, self.nChannels)
        out_original = out_original.view(-1, self.nChannels)
        return self.fc(out), self.fc(out_original)

    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list


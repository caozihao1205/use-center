from __future__ import absolute_import
# import  numpy as np
'''Resnet for ecg classification using lstm
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch

__all__ = ['resnet_lstm']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_1D(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3x3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1D(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=4, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for ecg model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
            block_ecg = BasicBlock1D
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
            block_ecg = Bottleneck1D
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.batch_size = None
        self.block = block

        self.inplanes = 16
        self.inplanes_ecg = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) #nn.AvgPool2d(8) <-- I made a change here

        self.fc = nn.Linear(64 * block.expansion, 32 * block.expansion)
        self.fc_final = nn.Linear(32 * block.expansion, num_classes)
        self.fca1 = nn.Linear(100, 128)
        self.fca2 = nn.Linear(128, 64 * block.expansion)
        self.time_steps = 10
        self.fcg1 = nn.Linear(50, 128)
        self.fcg2 = nn.Linear(128, 64 * block.expansion)
        self.fc_concat = nn.Linear(64 * block.expansion * 3, 64 * block.expansion)
        self.lstm = nn.LSTM(input_size=64 * block.expansion, hidden_size=64 * block.expansion, num_layers=6,
                            batch_first=True,bidirectional=True)
        self.hidden_cell = None

        self.lstm_ecg = nn.LSTM(input_size=64 * block.expansion, hidden_size=64 * block.expansion, num_layers=2,
                            batch_first=True)

        self.conv1ecg = nn.Conv1d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1ecg = nn.BatchNorm1d(16)
        self.layer1ecg = self._make_layer_1d(block_ecg, 16, 3)
        self.layer2ecg = self._make_layer_1d(block_ecg, 32, 3, stride=2)
        self.layer3ecg = self._make_layer_1d(block_ecg, 64, 3, stride=2)
        self.avgpoolecg = nn.AdaptiveAvgPool1d(output_size=(1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=64 * block.expansion, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_1d(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_ecg != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes_ecg, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes_ecg, planes, stride, downsample))
        self.inplanes_ecg = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_ecg, planes))

        return nn.Sequential(*layers)

    def forward(self,x, w, init_states=None):
        # a = x[1]
        # g = x[2]
        # w = x[0]
        # x = x[1]

        # print("spectogram", x.size())
        # print("gender", g.size())
        # print("age", a.size())
        # print("ecg", w.size())
        self.time_steps = x.size(1)
        self.batch_size = x.size(0)

        # if init_states is None:
        #     self.hidden_cell = (torch.zeros((6*2, self.batch_size, 64 * self.block.expansion)).cuda(),
        #                         torch.zeros((6*2, self.batch_size, 64 * self.block.expansion)).cuda())
        x = x.view(self.batch_size * self.time_steps, x.size(2), x.size(3), x.size(4))
        #
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)   # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # # print(w.size())

        w = w.view(self.batch_size * self.time_steps, w.size(2), w.size(3))

        # w = torch.squeeze(w, dim=1)
        w = self.conv1ecg(w)
        w = self.bn1ecg(w)
        w = self.relu(w)
        w = self.layer1ecg(w)
        w = self.layer2ecg(w)
        w = self.layer3ecg(w)
        w = self.avgpoolecg(w)
        w = torch.squeeze(w)

        # print(w.shape)
        # w = w.view(w.size(0), -1)
        w = w.view(self.batch_size, self.time_steps, w.size(1))
        w, (h0, b0) = self.lstm_ecg(w, self.hidden_cell)
        # # w = w[:, -1, :]
        w = h0[-1, :, :]

        x = x.view(self.batch_size, self.time_steps, x.size(1))
        x, (h1, b1) = self.lstm(x, self.hidden_cell)
        # # # print(h1.size(), h0.size())
        # # # x = x[:, -1, :]
        x = h1[-1, :, :]
        #
        # # a = torch.unsqueeze(a, dim=1)
        # print(w.size())
        # x = self.unet(w.squeeze(1))
        # x = x.permute(0, 2, 1)
        # w = w.squeeze(1)
        # print("before", w.size())
        #
        # x = self.seqnet(w)
        # print("after", w.size())

        #
        # a = self.fca1(a)
        # a = self.fca2(a)
        #
        # g = self.fcg1(g)
        # g = self.fcg2(g)


        # # print("spectogram", x.size())
        # print("gender", g.size())
        # print("age", a.size())
        # print("ecg", w.size())
        # e = torch.cat((x.unsqueeze(0), a.unsqueeze(0), g.unsqueeze(0), w.unsqueeze(0)), dim=0)

        # e = torch.cat((x.unsqueeze(0), a.unsqueeze(0), g.unsqueeze(0)), dim=0)

        e = torch.cat((w.unsqueeze(0),x.unsqueeze(0)), dim=0)
        x = self.transformerEncoder(e)

        # # print("spectogram", x.size())
        x = torch.mean(x, dim=0)

        # x = torch.cat((x, a, g, w), dim=1)

        # x = torch.cat((a, g, w), dim=1)

        # x = self.fc_final(x)

        # print("input", x.size())
        # x = torch.squeeze(w)
        # print("input", x.size())
        # print("spectogram", x.size())
        x = self.fc(x)
        x = self.fc_final(x)


        return x


def resnet_lstm(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

if __name__ == '__main__':
    net=resnet_lstm(depth=20)
    a = torch.rand(3,6,3,10,10)
    b = torch.rand(3,6,1,10)
    output=net(a,b)
    print(net)

    print(output.shape)
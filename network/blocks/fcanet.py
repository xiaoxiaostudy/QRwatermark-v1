import torch.nn as nn
from torchvision.models import ResNet
from .layer import MultiSpectralAttentionLayer


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FcaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        global _mapper_x, _mapper_y
        super(FcaBottleneck, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')

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
        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, ):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7),(1024,7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.downsample = nn.Sequential(
                                        nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                                        nn.BatchNorm2d(planes),
                                        )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fcanet34(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet50(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet101(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def fcanet152(num_classes=1_000, pretrained=False):
    """Constructs a FcaNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(FcaBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

class FcaNet_decoder(nn.Module):


	def __init__(self, in_channels, out_channels, blocks, block_type="FcaBasicBlock", drop_rate=2):
		super(FcaNet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate)
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
     
class FcaNet(nn.Module):


	def __init__(self, in_channels, out_channels, blocks, block_type="FcaBasicBlock"):
		super(FcaNet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels)] if blocks != 0 else []#第一层
		for _ in range(blocks - 1):#后三层
			layer = eval(block_type)(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


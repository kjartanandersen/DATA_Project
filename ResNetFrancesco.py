# ResNet implementation by Francesco Saverio Zuppichini https://github.com/FrancescoSaverioZuppichini/ResNet

import torch.nn as nn
from functools import partial
from collections import OrderedDict
from torchsummary import summary

class Conv2dAuto(nn.Conv2d):
  '''
  Convolutional layer with automatic padding
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding base

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
  '''
  Takes an input with in_channels, applies some blocks of convolutional layers to reduce
  it to out_channels and sum it up to the original input. If their sizes mismatch, then
  the input goes into an identity
  '''
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    self.blocks = nn.Identity()
    self.shortcut = nn.Identity()

  def forward(self, x):
    residual = x
    if self.should_apply_shortcut:
      residual = self.shortcut(x)
    x = self.blocks(x)
    x += residual
    return x

  @property
  def should_apply_shortcut(self):
    return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    '''
    In ResNet each block has a expansion parameter in order to increase the out_channels. Also, the
    identity is defined as a Convolution followed by an Activation layer, this is referred as shortcut.
    '''
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)

        })) if self.should_apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    '''
    Function that stacks convolutional layer and batch norm layer
    '''
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) }))

print(conv_bn(3, 3, nn.Conv2d, kernel_size=3))

class ResNetBasicBlock(ResNetResidualBlock):
    '''
    Basic block for Residual Network, composed of two layers of 3x3 convolutions/batch normalization/ReLU
    '''
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetBottleNeckBlock(ResNetResidualBlock):
    '''
    BottleNeck block that "The three layers are 1x1, 3x3, and 1x1 convolutions, where the 1x1 layers
    are responsible for reducing and then increasing (restoring) dimensions, leaving the 3x3 layer
    a bottleneck with smaller input/output dimensions."
    '''
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
          conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    '''
    A ResNet's layer is composed by n blocks stacked one after the other.
    '''
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])


    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    '''
    The final ResNet Model
    '''

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])


model = resnet101(3, 1000)
summary(model.cuda(), (3, 224, 224))


import torch.nn as nn
from .resnet import ResNet
from ..registry import BACKBONES
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module
class PolNet(nn.Module):
    def __init__(self,
                 in_channels=4,
                 fusion_cfg=[8, 16, 8, 5],
                 res_depth=50):
        super(PolNet, self).__init__()
        self.fusion = None
        if fusion_cfg is not None:
            self.fusion = self._make_layers(fusion_cfg, in_channels)
            self.cnn = ResNet(depth=res_depth,
                              in_channels=fusion_cfg[-1])
        else:
            self.cnn = ResNet(depth=res_depth,
                              in_channels=in_channels)
        self.fusion_out = None
        self.cnn.train(mode=True)
        print(self.fusion)
    def forward(self, x):
        if self.fusion is not None:
            x = self.fusion(x)
            x = self.cnn(x)
        else:
            x = self.cnn(x)
        return x

    '''
    名称：_make_layers
    功能：构造偏振前端网络
    参数：
        cfg:[out_ch_num_1,out_ch_num_2,...out_ch_num_n]
            偏振前端网络的结构，out_ch_num_n代表每层的合成通道数
        src_channels：输入的偏振方向图通道数
    '''
    def _make_layers(self, cfg, in_channel):
        layers = []
        in_channels = in_channel
        if cfg is not None:
            for o in cfg:
                conv2d = nn.Conv2d(in_channels, o, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                layers += [conv2d, nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
                in_channels = o
            return nn.Sequential(*layers)
        else:
            return None

    def init_weights(self, pretrained=None):
        self.cnn.init_weights()
        if self.fusion is not None:
            for m in self.fusion.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

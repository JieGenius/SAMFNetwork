import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from modules.inverse_polar_transformation import InversePolarTransformation
from modules.polar_transformation import PolarTransformation


class MCTF(nn.Module):

    def __init__(self, size, in_polar, out_polar, channel,
                 current_branch_layer, opposite_branch_layer,
                 fusion_branch_layer):
        super().__init__()
        self.polar_transform = PolarTransformation(input_size=size)
        self.inverse_polar_transform = InversePolarTransformation(
            input_size=size)
        self.current_conv = branch_factory(
            current_branch_layer, channel=channel)

        self.opposite_conv = branch_factory(
            opposite_branch_layer, channel=channel)
        self.fusion_conv = branch_factory(fusion_branch_layer, channel=channel)
        self.in_polar = in_polar
        self.out_polar = out_polar

    def forward(self, x, opposite=None):
        if opposite is not None:
            x_opposite = x_opposite_conved = opposite
        else:
            x_opposite = self.inverse_polar_transform(
                x) if self.in_polar else self.polar_transform(x)
            x_opposite_conved = self.opposite_conv(x_opposite)
        x_current_conved = self.current_conv(x)
        if self.in_polar and self.out_polar:
            x_bar = self.polar_transform(x_opposite_conved)
            # x_cat = torch.cat([x_current, x_bar], dim=1)
            x_cat = x_current_conved + x_bar
        elif self.in_polar and (self.out_polar is False):
            x_bar = self.inverse_polar_transform(x_current_conved)
            x_cat = x_bar + x_opposite_conved
        elif (self.in_polar is False) and self.out_polar:
            x_bar = self.polar_transform(x_current_conved)
            x_cat = x_bar + x_opposite_conved
        else:
            x_bar = self.inverse_polar_transform(x_opposite_conved)
            x_cat = x_current_conved + x_bar
        x_fusion = self.fusion_conv(x_cat)
        return x_current_conved, x_opposite_conved, x_fusion


def branch_factory(count, channel=256):
    if count == 1:
        m = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(channel), nn.ReLU())
    elif count == 2:
        m = BasicBlock(channel, channel, 1)
    elif count == 3:
        m = Bottleneck(channel, channel // 4)
    elif count == 4:
        m = nn.Sequential(
            BasicBlock(channel, channel), BasicBlock(channel, channel))
    elif count == 5:
        m = nn.Sequential(
            BasicBlock(channel, channel), Bottleneck(channel, channel // 4))
    elif count == 6:
        m = nn.Sequential(
            Bottleneck(channel, channel // 4),
            Bottleneck(channel, channel // 4))
    else:
        m = nn.Identity()
    return m

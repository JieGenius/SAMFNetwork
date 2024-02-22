import torch
import torch.nn as nn
import torch.nn.functional as F

CV_2PI = 6.283185307179586476925286766559
CV_PI = 3.1415926535897932384626433832795


class PolarTransformation(nn.Module):

    def __init__(self,
                 input_size: int,
                 mode: str = 'bilinear',
                 max_radius=None,
                 border_type='replicate'):
        """将直角坐标转换为极坐标.

        :param input_size:
        :param mode:
        :param: border_type: ["replica","constant",""TRANSPARENT]
        """
        assert mode in ['bilinear', 'nearest']
        super().__init__()
        h, w = input_size, input_size
        cx, cy = h / 2, w / 2
        max_radius = max_radius if max_radius is not None else h / 2
        dh, dw = 2 * max_radius, 2 * max_radius
        k_angle = CV_2PI / dh
        k_mag = max_radius / dw
        rhos = torch.arange(0, dw).view(1, -1) * k_mag
        phis = torch.arange(0, dh).view(-1, 1) * k_angle
        self.input_size = input_size
        # 分别存放x坐标和y坐标，opencv中使用32F
        map_x: torch.Tensor = rhos * torch.cos(phis) + cx
        map_y: torch.Tensor = rhos * torch.sin(phis) + cy
        # 这儿是因为  F.grid_sample 认为 原图中,左上角为（-1，-1）,右下角为(1,1)， 因此对其进行归一化
        map_x = map_x / (self.input_size - 1) * 2 - 1
        map_y = map_y / (self.input_size - 1) * 2 - 1
        self.map: torch.Tensor = torch.stack([map_x, map_y], -1).float()
        self.mode = mode

    def forward(self, x):
        x = x.float()
        if self.map.device != x.device:
            self.map = self.map.to(x.device)
        *b, c, h, w = x.shape
        bt = b[0] if len(b) else 1
        map = torch.stack([self.map] * bt, 0)
        if len(x.shape) == 3:
            return F.grid_sample(
                x.unsqueeze(0),
                map,
                mode=self.mode,
                padding_mode='zeros',
                align_corners=True).squeeze(0)
        else:
            return F.grid_sample(
                x,
                map,
                mode=self.mode,
                padding_mode='zeros',
                align_corners=True)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CV_2PI = 6.283185307179586476925286766559
CV_PI = 3.1415926535897932384626433832795


class InversePolarTransformation(nn.Module):
    def __init__(self, input_size: int, mode: str = "bilinear", border_type="replicate"):
        '''
        将直角坐标转换为极坐标
        :param input_size:
        :param mode:
        :param: border_type: ["replica","constant",""TRANSPARENT]
        '''
        assert mode in ["bilinear", "nearest"]
        super(InversePolarTransformation, self).__init__()
        ANGEL_BORDER = 1
        self.angel_border = ANGEL_BORDER
        self.input_size = input_size
        h, w = input_size, input_size
        cx, cy = h / 2, w / 2
        max_radius = h / 2
        k_angle = np.pi * 2 / h
        k_mag = max_radius / w
        buf_y, buf_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing="ij")
        buf_x: torch.Tensor = buf_x - cx
        buf_y: torch.Tensor = buf_y - cy
        magnitude, angle = cv2.cartToPolar(buf_x.numpy().astype(float),
                                           buf_y.numpy().astype(float),
                                           angleInDegrees=0)
        map_x = torch.from_numpy(magnitude) / k_mag + self.angel_border
        map_y = torch.from_numpy(angle) / k_angle + self.angel_border
        map_x = map_x / (self.input_size - 1 + 2 * self.angel_border) * 2 - 1
        map_y = map_y / (self.input_size - 1 + 2 * self.angel_border) * 2 - 1
        self.map: torch.Tensor = torch.stack([map_x, map_y], -1).float()
        self.mode = mode

    def forward(self, x: torch.Tensor):
        x = x.float()
        if self.map.device != x.device:
            self.map = self.map.to(x.device)
        *b, c, h, w = x.shape
        bt = b[0] if len(b) else 1
        map = torch.stack([self.map] * bt, 0)
        xt = F.pad(x, [self.angel_border] * 4, mode="replicate") if len(x.shape) == 4 else \
            F.pad(x.unsqueeze(0), [self.angel_border] * 4, mode="replicate")
        if len(x.shape) == 3:
            return F.grid_sample(
                xt, map, mode=self.mode, padding_mode="zeros", align_corners=True).squeeze(0)
        else:

            return F.grid_sample(xt, map, mode=self.mode, padding_mode="zeros", align_corners=True)

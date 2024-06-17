import numpy as np
import torch
import math
import torch.nn as nn

from torch.nn.functional import grid_sample
from torchvision.transforms.functional import rotate

import onnx
import onnxsim
import onnxruntime as ort

class Rotate(nn.Module):
    def __init__(self, **kwargs):
        super(Rotate, self).__init__(**kwargs)

    def custom_rotate(self, img, angle, center = None):
        """
        Rotate the image by angle.
        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            img (PIL Image or Tensor): image to be rotated.
            angle (number): rotation angle value in degrees, counter-clockwise.
            center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
                Default is the center of the image.
        Returns:
            Rotated image.
        """

        _, height, width = img.shape
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [0.0, 0.0]
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

        # due to current incherence of rotation angle direction between affine and rotate implementations
        # we need to set -angle.
        angle = (-angle) * (torch.pi / 180)
        shear = torch.FloatTensor([0.0, 0.0, 0.5])
        #shear = shear + torch.zeros_like(shear)

        sx = shear[0] * (torch.pi / 180)
        sy = shear[1] * (torch.pi / 180)

        #sx, sy = torch.tensor(sx), torch.tensor(sy)
        cx, cy = center_f
        tx, ty = 0.0, 0.0

        #angle = torch.tensor(angle).to(torch.float32)
        
        # RSS without scaling
        a = torch.cos(angle - sy) / torch.cos(sy)
        b = -torch.cos(angle - sy) * torch.tan(sx) / torch.cos(sy) - torch.sin(angle)
        c = torch.sin(angle - sy) / torch.cos(sy)
        d = -torch.sin(angle - sy) * torch.tan(sx) / torch.cos(sy) + torch.cos(angle)
        
        matrix = torch.stack([d, -b, shear[0], -c, a, shear[0]])

        #matrix = [x for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy

        _, w, h = img.shape
        theta = matrix.reshape(1, 2, 3)

        d = 0.5
        base_grid = torch.empty(1, h, w, 3, device=theta.device)
        x_grid = torch.linspace(-w * d + 0.5, w * 0.5 + d - 1, steps=w, device=theta.device)
        base_grid[..., 0] = x_grid.clone()
        y_grid = torch.linspace(-h * d + 0.5, h * 0.5 + d - 1, steps=h, device=theta.device).unsqueeze_(-1)
        base_grid[..., 1] = y_grid.clone()
        base_grid[..., 2] = 1

        rescaled_theta = theta.transpose(1, 2)
        rescaled_theta = rescaled_theta / torch.stack([shear[2] * w, shear[2] * h])
        output_grid = base_grid.reshape(1, h * w, 3)
        output_grid = torch.bmm(output_grid, rescaled_theta)
        grid = output_grid.reshape(1, h, w, 2)

        img = img.unsqueeze(0)
        img = grid_sample(img, grid, mode='nearest', padding_mode="zeros", align_corners=False)
        img = img.squeeze(0)

        return img, grid
    '''
    def forward(self,
                prev_bev,
                angle,
                center = [100, 100],
                **kwargs):

        img = rotate(
            img=prev_bev,
            angle=angle.item(),
            center=center,
            **kwargs)

        return img
    '''

model = Rotate()

prev_bev_old = torch.load('prev_bev_old.pt')
prev_bev_old = prev_bev_old.squeeze(1).reshape(50, 50, -1).permute(2, 0, 1).cpu()
angle = torch.tensor(-1.1832172)
rotate_center=[100, 100]

canbus = torch.load('canbus.pt')

prev = rotate(img=prev_bev_old, angle=canbus[0][-1], center=rotate_center)
prev = prev.permute(1, 2, 0).reshape(50 * 50, 1, -1)
print(prev)

prev1, grid = model.custom_rotate(img=prev_bev_old, angle= torch.tensor(canbus[0][-1], dtype=torch.float32), center = rotate_center)
prev1 = prev1.permute(1, 2, 0).reshape(50 * 50, 1, -1)
print(prev1)

import torch.nn.functional as F
mse = F.mse_loss(prev.cuda(), prev1.cuda())
print(mse)

'''
torch.onnx.export(
            model, 
            (prev_bev_old.cpu(), angle, rotate_center),
            "rotate.onnx", 
            export_params=True, 
            input_names = ['img', 'angle', 'center'],
            verbose=True,
            opset_version=16,
        )

model_onnx = onnx.load("/home/ava/rajrup/new_bev/BEVFormer/rotate.onnx")
model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
#assert check, 'assert check failed'
onnx.save(model_onnx, 'rotate-sim.onnx')

prev_bev_old = torch.load('prev_bev_old.pt')
prev_bev_old = prev_bev_old.squeeze(1).reshape(50, 50, -1).permute(2, 0, 1)

canbus = torch.load('canbus.pt')
canbus = canbus[0].astype(np.float32)
prev_bev_rotated = torch.load('prev_bev_rotated.pt')
#print(canbus)

sess = ort.InferenceSession("/home/ava/rajrup/new_bev/BEVFormer/rotate-sim.onnx")
inputs = {'img': prev_bev_old.cpu().numpy(), 'angle': np.array(canbus[-1]).astype(np.float32)}
output = sess.run(None, inputs)
x = torch.Tensor(output[0])
x = x.permute(1, 2, 0).reshape(50 * 50, 1, -1)

#print(output)
print("Original Image")
print(prev_bev_old)
print('______________________________________')
print("Custom Rotate Image")
print(x)
print('______________________________________')
print("Torchvision Rotate Image")
print(prev)

import torch.nn.functional as F
mse = F.mse_loss(x.cuda(), prev.cuda())
print("MSE for custom and torchvision rotate")
print(mse)

torch.onnx.export(
            model, 
            (prev_bev_old.cpu(), canbus[0][-1]),
            "rotate-tv.onnx", 
            export_params=True, 
            input_names = ['img', 'angle'],
            verbose=True,
            opset_version=16,
        )

model_onnx = onnx.load("/home/ava/rajrup/new_bev/BEVFormer/rotate-tv.onnx")
model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
#assert check, 'assert check failed'
onnx.save(model_onnx, 'rotate-tv-sim.onnx')
'''
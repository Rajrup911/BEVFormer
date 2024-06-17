import numpy as np
import torch
import math
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import rotate
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16

def custom_rotate(img, angle, center = None):
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

    center_f = [0.0, 0.0]
    _, height, width = img.shape
    # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
    center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    #angle = -angle
    angle = (-angle) * (math.pi / 180)
    shear = torch.FloatTensor([0.0, 0.0, 0.5])
    #shear = shear + torch.zeros_like(shear)

    sx = shear[0] * (math.pi / 180)
    sy = shear[1] * (math.pi / 180)

    #sx, sy = torch.tensor(sx), torch.tensor(sy)

    cx, cy = center_f
    tx, ty = 0.0, 0.0

    # RSS without scaling
    a = torch.cos(angle - sy) / torch.cos(sy)
    b = -torch.cos(angle - sy) * torch.tan(sx) / torch.cos(sy) - torch.sin(angle)
    c = torch.sin(angle - sy) / torch.cos(sy)
    d = -torch.sin(angle - sy) * torch.tan(sx) / torch.cos(sy) + torch.cos(angle)
    

    matrix = torch.stack([d, -b, shear[0], -c, a, shear[0]])
    #matrix = torch.tensor(matrix)
    #print(type(matrix))
    #exit()
    # matrix = [x for x in matrix]
    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    _, w, h = img.shape
    #dtype = img.dtype if torch.is_floating_point(img) else torch.float32
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

class PerceptionTransformer(BaseModule):
    def __init__(self, **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.embed_dims = 256
        self.num_feature_levels = 4
        self.num_cams = 6
        self.fp16_enabled = False
        #self.onnx_runtime = False
        #self.onnx_export = True

        self.rotate_prev_bev = True
        self.use_shift = True
        self.use_can_bus = True
        self.can_bus_norm = True
        self.use_cams_embeds = True

        self.two_stage_num_proposals = 300
        self.init_layers()
        self.rotate_center = [100, 100]

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        constant_init(self.reference_points, val=0., bias=0.)
        constant_init(self.can_bus_mlp, val=0., bias=0.)

    @auto_fp16(apply_to=('prev_bev'))
    def get_bev_features(
            self,
            canbus,
            prev_bev,
            bev_h=50,
            bev_w=50):    

        """
        obtain bev features.
        """
        bs = 1
        grid_length=[0.512, 0.512]
        bev_h=50
        bev_w=50
        #bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        
        # obtain rotation angle and shift with ego motion
        delta_x = canbus[0].unsqueeze(0).float() # np.array([each['can_bus'][0].cpu().numpy() for each in kwargs['img_metas']])
        
        delta_y = canbus[1].unsqueeze(0).float() # np.array([each['can_bus'][1].cpu().numpy() for each in kwargs['img_metas']])
        
        ego_angle = canbus[-2].unsqueeze(0).float() / torch.pi * 180 #n p.array([each['can_bus'][-2] / torch.pi * 180 for each in kwargs['img_metas']])
        
        #grid_length = torch.tensor(grid_length)
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = (torch.atan(delta_y / (delta_x + 1e-8)) + ((1 - torch.sign(delta_x)) / 2) * torch.sign(delta_y) * torch.pi ) / torch.pi
        
        bev_angle = ego_angle - translation_angle
        # print(bev_h)
        # exit()

        shift_y = translation_length * \
            torch.cos(bev_angle / 180 * torch.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            torch.sin(bev_angle / 180 * torch.pi) / grid_length_x / bev_w

        shift = torch.stack([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev.shape[1] == bev_h * bev_w:
            prev_bev = prev_bev.permute(1, 0, 2)
        if self.rotate_prev_bev:
            for i in range(bs):
                # num_prev_bev = prev_bev.size(1)
                rotation_angle = canbus[-1]
                #print(prev_bev[:, i].reshape(bev_h, bev_w, -1).shape)
                #exit()
                tmp_prev_bev = prev_bev.squeeze(1).reshape(
                    bev_h, bev_w, -1).permute(2, 0, 1)
                tmp_prev_bev, grid = custom_rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, 1, -1)
                prev_bev[:, 0] = tmp_prev_bev[:, 0]
        
        return prev_bev, shift
        
    @auto_fp16(apply_to=('prev_bev'))
    def forward(self,
                canbus,
                prev_bev,
                **kwargs):

        prev_bev = self.get_bev_features(
            canbus,
            prev_bev,
            bev_h=50,
            bev_w=50,
            **kwargs)

        return prev_bev

model = PerceptionTransformer()
prev_bev = torch.randn(2500, 1, 256)
canbus = torch.tensor(np.fromfile("/home/ava/rajrup/BEVFormer/src/projects/configs/can_bus.npy", dtype = np.float32))
#print(model((mlvl_feats, bev_queries, canbus)))
torch.onnx.export(
            model, 
            (canbus, prev_bev),
            "transformerv2.onnx", 
            export_params=True, 
            input_names = ['canbus', 'prev_bev'],
            verbose=True,
            opset_version=16,
        )
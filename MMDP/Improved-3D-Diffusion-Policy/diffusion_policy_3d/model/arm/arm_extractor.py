import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from diffusion_policy_3d.model.arm.timm_img_encoder import TimmImgEncoder
from diffusion_policy_3d.model.vision_3d.pointnet_extractor import build_condition_adapter


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class StateEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU):
        super().__init__()
        self.state_key = 'full_state'
        self.state_shape = observation_space[self.state_key]
        cprint(f"[StateEncoder] state shape: {self.state_shape}", "yellow")
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[StateEncoder] output dim: {output_dim}", "red")
        self.output_dim = output_dim
        
    def output_shape(self):
        return self.output_dim
        
    def forward(self, observations: Dict) -> torch.Tensor:
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)
        return state_feat


class ArmiDP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_encoder: TimmImgEncoder,
                #  img_adaptor: ImgAdapter,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='dp3_encoder',
                 point_downsample=True,
                 use_instruction=False,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.depth_key = 'depth'
        self.image_main_key = 'image_main'
        self.image_wrist_key = 'image_wrist'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels * 2

        self.use_instr = use_instruction
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.depth_shape = observation_space[self.depth_key]
        self.state_shape = observation_space[self.state_key]
        self.image_main_shape = observation_space[self.image_main_key]

        self.num_points = pointcloud_encoder_cfg.num_points # 4096
        
        cprint(f"[ArmiDP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[ArmiDP3Encoder] depth point cloud shape: {self.depth_shape}", "yellow")
        cprint(f"[ArmiDP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[ArmiDP3Encoder] iamge shape: {self.image_main_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        
        
        if pointnet_type == "multi_stage_pointnet":
            from diffusion_policy_3d.model.vision_3d.multi_stage_pointnet import MultiStagePointNetEncoder
            self.pc_extractor = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
            self.depth_extractor = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        self.image_encoder = img_encoder
        self.img_encoder_outshape = self.image_encoder.output_shape()[1] // 2

        self.img_adaptor_main = build_condition_adapter(pointcloud_encoder_cfg.img_adaptor,
                                                    self.img_encoder_outshape,
                                                    pointcloud_encoder_cfg.out_channels)
        self.img_adaptor_wrist = build_condition_adapter(pointcloud_encoder_cfg.img_adaptor,
                                                    self.img_encoder_outshape,
                                                    pointcloud_encoder_cfg.out_channels)
        self.n_output_channels  += pointcloud_encoder_cfg.out_channels * 2

        cprint(f"[ArmDP3Encoder] output dim: {self.n_output_channels}", "red")

        if self.use_instr:
            self.instr_key = 'instruction'
            self.instr_shape = observation_space[self.instr_key]
            self.lang_adaptor = build_condition_adapter(pointcloud_encoder_cfg.lang_adaptor,
                                                    self.instr_shape[0],
                                                    pointcloud_encoder_cfg.out_channels)
            self.n_output_instr_channels = pointcloud_encoder_cfg.out_channels
            cprint(f"[ArmDP3Encoder] instr output dim: {self.n_output_instr_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        depth = observations[self.depth_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        assert len(depth.shape) == 3, cprint(f"point cloud shape: {depth.shape}, length should be 3", "red")

        if self.use_instr:
            instr_embed = observations[self.instr_key]
            assert len(instr_embed.shape) == 2, cprint(f"depth shape: {depth.shape}, length should be 2", "red")
            instr_feat = self.lang_adaptor(instr_embed)   # B * out_channel

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
            depth = self.point_preprocess(depth, self.num_points)
           
        pn_feat = self.pc_extractor(points)    # B * out_channel
        depth_feat = self.depth_extractor(points)    # B * out_channel
         
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64

        image_feat = self.image_encoder({self.image_main_key: observations[self.image_main_key], \
                                         self.image_wrist_key: observations[self.image_wrist_key]})

        image_feat_main = self.img_adaptor_main(image_feat[..., :self.img_encoder_outshape])
        image_feat_wrist = self.img_adaptor_wrist(image_feat[..., self.img_encoder_outshape:])
        
        final_feat = torch.cat([depth_feat, image_feat_main, pn_feat, image_feat_wrist, state_feat], dim=-1)
        
        if self.use_instr:
            return final_feat, instr_feat
        return final_feat


    def output_shape(self):
        return self.n_output_channels

    def output_instr_shape(self):
        return self.n_output_instr_channels
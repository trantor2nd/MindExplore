import torch
import torch.nn as nn
import re

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import diffusion_policy_3d.model.vision_3d.point_process as point_process


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

def build_condition_adapter(projector_type, in_features, out_features):
    projector = None
    if projector_type == 'linear':
        projector = nn.Linear(in_features, out_features)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU(approximate="tanh"))
                modules.append(nn.Linear(out_features, out_features))
            projector = nn.Sequential(*modules)

    if projector is None:
        raise ValueError(f'Unknown projector type: {projector_type}')

    return projector

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
    
class iDP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='dp3_encoder',
                 point_downsample=True,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        self.num_points = pointcloud_encoder_cfg.num_points # 4096
        


        cprint(f"[iDP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[iDP3Encoder] state shape: {self.state_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        
        
        if pointnet_type == "multi_stage_pointnet":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            self.extractor = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
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

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
           
        pn_feat = self.extractor(points)    # B * out_channel
         
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
    

class iDP3MSEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='dp3_encoder',
                 point_downsample=True,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.depth_key = 'depth'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels * 2
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.depth_shape = observation_space[self.depth_key]
        self.state_shape = observation_space[self.state_key]

        self.num_points = pointcloud_encoder_cfg.num_points # 4096
        


        cprint(f"[iDP3MSEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[iDP3MSEncoder] depth shape: {self.depth_shape}", "yellow")
        cprint(f"[iDP3MSEncoder] state shape: {self.state_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        
        
        if pointnet_type == "multi_stage_pointnet":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            self.extractor_pc = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
            self.extractor_depth = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
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

        cprint(f"[iDP3MSEncoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        depth = observations[self.depth_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        assert len(depth.shape) == 3, cprint(f"depth shape: {depth.shape}, length should be 3", "red")

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
            depth = self.point_preprocess(depth, self.num_points)
           
        pn_feat = self.extractor_pc(points)    # B * out_channel
        depth_feat = self.extractor_depth(depth)    # B * out_channel
         
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([depth_feat, pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
    

class iDP3MSInstrEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='dp3_encoder',
                 point_downsample=True,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.depth_key = 'depth'
        self.instr_key = 'instruction'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels * 2
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.depth_shape = observation_space[self.depth_key]
        self.state_shape = observation_space[self.state_key]
        self.instr_shape = observation_space[self.instr_key]

        self.num_points = pointcloud_encoder_cfg.num_points # 4096
        


        cprint(f"[iDP3MSInstrEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[iDP3MSInstrEncoder] depth shape: {self.depth_shape}", "yellow")
        cprint(f"[iDP3MSInstrEncoder] state shape: {self.state_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        
        
        if pointnet_type == "multi_stage_pointnet":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            self.extractor_pc = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
            self.extractor_depth = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        self.lang_adaptor = build_condition_adapter(pointcloud_encoder_cfg.lang_adaptor,
                                                    self.instr_shape[0],
                                                    pointcloud_encoder_cfg.out_channels)

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.n_output_instr_channels = pointcloud_encoder_cfg.out_channels
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[iDP3MSInstr Encoder] obs output dim: {self.n_output_channels}", "red")
        cprint(f"[iDP3MSInstr Encoder] instr output dim: {self.n_output_instr_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        depth = observations[self.depth_key]
        instr_embed = observations[self.instr_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        assert len(depth.shape) == 3, cprint(f"depth shape: {depth.shape}, length should be 3", "red")
        assert len(instr_embed.shape) == 2, cprint(f"depth shape: {depth.shape}, length should be 2", "red")

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
            depth = self.point_preprocess(depth, self.num_points)
           
        pn_feat = self.extractor_pc(points)    # B * out_channel
        depth_feat = self.extractor_depth(depth)    # B * out_channel
        instr_feat = self.lang_adaptor(instr_embed)   # B * out_channel
         
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        obs_feat = torch.cat([depth_feat, pn_feat, state_feat], dim=-1)
        return obs_feat, instr_feat


    def output_shape(self):
        return self.n_output_channels

    def output_instr_shape(self):
        return self.n_output_instr_channels
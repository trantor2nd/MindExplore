import sys
sys.path.append("/home/imagelab/zys/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy")

from typing import Dict
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint
import os
import json
import re

from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('/home/imagelab/zys/checkpoint/distilbert-base-uncased')
instr_model = DistilBertModel.from_pretrained("/home/imagelab/zys/checkpoint/distilbert-base-uncased")

RADIUS_PC = 1.2
RADIUS_DEPTH_1 = 0.5

USE_PC = True
RADIUS_DEPTH = 1.5

USE_DEPTH_1 = False
# RADIUS_DEPTH = 0.8

def depth_to_lidar(depth_image, 
                   fx: float = 388.81756591796875, fy: float = 388.81756591796875, 
                   cx: float = 319.6447448730469, cy: float = 237.4071502685547):
    """
    将深度图像转换为激光雷达点云数据
    :param depth_image: 深度图像（以毫米为单位）
    :param fx, fy: 相机的焦距（单位：像素）
    :param cx, cy: 相机的主点坐标（单位：像素）
    :return: 点云数据（Nx3的numpy数组）
    """
    height, width = depth_image.shape
    points = []

    depth_image = depth_image / 1000.0
    u, v = np.meshgrid(np.arange(width), np.arange(height))  # 创建像素网格
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    depth_mask = depth_image > 0  # 找到有效深度值的位置
    z = depth_image  # z 就是深度

    # 对于有效的像素计算对应的 x, y 坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.vstack((x[depth_mask], y[depth_mask], z[depth_mask])).T
    return points

class MarsmindBaseDataset(BaseDataset):
    def __init__(self,
            data_path, 
            pad_before=0,
            pad_after=0,
            task_name=None,
            num_points=4096,
            use_instruction=False,
            ):
        super().__init__()
        cprint(f'Loading MarsmindBaseDataset from {data_path}/{task_name}', 'green')
        self.task_name = task_name.lower()
        self.num_points = num_points
        self.use_instr = use_instruction

        self.episode_state_data = []
        self.episode_action_data = []
        self.episode_pc_data = []
        self.episode_depth_data = []
        self.episode_img_main_path = []
        self.episode_instr = []

        self.episode_len = []

        for dir_task in os.listdir(os.path.join(data_path)):
            if 'txt' in dir_task:
                continue
            for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
                for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):                    
                    if self.task_name in dir_metaaction.lower():
                        episode_data = self.parse_metaaction(os.path.join(data_path, dir_task, dir_episode, dir_metaaction))
                        self.episode_state_data.append(episode_data['state'])
                        self.episode_action_data.append(episode_data['action'])
                        self.episode_pc_data.append(episode_data['pc'])
                        self.episode_depth_data.append(episode_data['depth'])
                        self.episode_img_main_path.append(episode_data['img_main_path'])
                        self.episode_instr.append(episode_data['instruction'])

                        self.episode_len.append(len(episode_data['state']))
        
        assert len(self.episode_state_data) == len(self.episode_action_data) and \
            len(self.episode_state_data) == len(self.episode_pc_data) and \
            len(self.episode_state_data) == len(self.episode_depth_data) and \
            len(self.episode_state_data) == len(self.episode_instr), 'data length error!'
        
        self.cumulative_len = np.cumsum(self.episode_len)
        self.train_idx = np.arange(-pad_before, pad_after + 1)
        self.obs_len = pad_before + 1

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': np.concatenate(self.episode_action_data, axis=0)}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer['image_main'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_instr:
            normalizer['instruction'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return self.cumulative_len[-1]

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        image_main = sample['img_main'][:,].astype(np.float32)
        depth = sample['depth'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        data = {
            'obs': {
                'agent_pos': agent_pos,
                'point_cloud': point_cloud,
                },
            'action': sample['action'].astype(np.float32)}
        
        data['obs']['image_main'] = image_main
        data['obs']['depth'] = depth

        if self.use_instr:
            instr = sample['instruction'][:,].astype(np.float32)
            data['obs']['instruction'] = instr
           
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_index = np.argmax(self.cumulative_len > idx)
        start_step = idx - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        
        episode_state = self.episode_state_data[episode_index]
        episode_action = self.episode_action_data[episode_index]
        episode_pc = self.episode_pc_data[episode_index]
        episode_depth = self.episode_depth_data[episode_index]
        episode_img_main_path = self.episode_img_main_path[episode_index]
        episode_instr = self.episode_instr[episode_index]

        train_idx = self.train_idx + start_step
        train_idx[train_idx < 0] = 0
        train_idx[train_idx > self.episode_len[episode_index] - 1] = self.episode_len[episode_index] - 1

        sample = dict()
        sample['state'] = np.array(episode_state)[train_idx[:self.obs_len]]
        sample['action'] = np.array(episode_action)[train_idx]
        sample['point_cloud'] = np.array(episode_pc)[train_idx[:self.obs_len]]
        sample['depth'] = np.array(episode_depth)[train_idx[:self.obs_len]]
        if self.use_instr:
            sample['instruction'] = episode_instr

        img_main = []
        for img_idx in train_idx[:self.obs_len]:
            img_main.append(np.array(Image.open(episode_img_main_path[img_idx]).convert("RGB")))
        sample['img_main'] = np.stack(img_main)

        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data


    def parse_metaaction(self, file_path):
        log_files = sorted(os.listdir(os.path.join(file_path, "logs")))
        num_steps = len(log_files)

        EPS = 1e-2
        first_idx = -1
        state = []
        action = []
        pc = []
        depth = []
        img_main_path = []
        
        # some episodes' sample freq is 4Hz and other is 8Hz
        if 'MarsMind_data/Grasp' in file_path or 'MarsMind_data/Place' in file_path or 'MarsMind_data/Search_Move_Container' in file_path\
        or ('MarsMind_data/Sample' in file_path and int(file_path.split('/')[-2][-2:]) > 70):
            sample_step = 2
        else:
            sample_step = 1

        start_idx = 0

        for i in range(start_idx, num_steps, sample_step):
            log_data = self.parse_log_file(os.path.join(file_path, "logs", log_files[i]))
            
            action_qpos = log_data['joint_states_msg']['position']
            action_qvel = log_data['joint_states_msg']['velocity']
            action_base_vel_y = log_data['cmd_vel_msg']['linear_x']
            action_base_delta_ang = log_data['cmd_vel_msg']['angular_z']

            qpos = log_data['joint_states_single_msg']['position']
            qvel = log_data['joint_states_single_msg']['velocity']
            base_vel_y = log_data['bunker_status_msg']['linear_velocity']
            base_delta_ang = log_data['bunker_status_msg']['angular_velocity']


            if action_base_vel_y > EPS or action_base_delta_ang > EPS or base_vel_y > EPS or base_delta_ang > EPS:
                
                state_list = []
                action_list = []
                if USE_DEPTH_1 and not USE_PC:
                    state_list += qpos.tolist()
                    action_list += action_qpos.tolist()
                    state_list += qvel.tolist()
                    action_list += action_qvel[:6].tolist()
                elif USE_PC and not USE_DEPTH_1:
                    state_list += [base_vel_y]+[base_delta_ang]
                    action_list += [action_base_vel_y]+[action_base_delta_ang]
                else:
                    raise ValueError('no implementation.')
                
                state.append(state_list)
                action.append(action_list)

                depth_img = cv2.imread(os.path.join(file_path, "p0", "depth", log_files[i - sample_step].replace('.log', '.png')), \
                                                    cv2.IMREAD_UNCHANGED)
                depth_data = depth_to_lidar(depth_img)
                distances = np.sqrt(depth_data[:, 0]**2 + depth_data[:, 2]**2)
                depth_data = depth_data[distances <= RADIUS_DEPTH]
                depth_data = point_process.uniform_sampling_numpy(depth_data[None, :], self.num_points)[0]
                depth.append(depth_data)
                
                if USE_PC and not USE_DEPTH_1:
                    pc_data = pd.read_csv(os.path.join(file_path, "pc", log_files[i].replace('.log', '.csv')))
                    pc_data = pc_data[['x', 'y', 'z']].to_numpy()
                    distances = np.sqrt(pc_data[:, 0]**2 + pc_data[:, 1]**2)
                    pc_data = pc_data[distances <= RADIUS_PC]
                    # 删除 x 在 (-0.5, 0) 且 y 在 (-0.5, 0.5) 区域内的点
                    mask = ~((pc_data[:, 0] > -0.5) & (pc_data[:, 0] < 0) & (pc_data[:, 1] > -0.5) & (pc_data[:, 1] < 0.5))
                    pc_data = pc_data[mask]
                elif USE_DEPTH_1 and not USE_PC:
                    depth_img = cv2.imread(os.path.join(file_path, "p1", "depth", log_files[i].replace('.log', '.png')), \
                                            cv2.IMREAD_UNCHANGED)
                    pc_data = depth_to_lidar(depth_img)
                    distances = np.sqrt(pc_data[:, 0]**2 + pc_data[:, 2]**2)
                    pc_data = pc_data[distances <= RADIUS_DEPTH_1]
                else:
                    raise ValueError('no implementation.')
                
                # first_pc_data = first_pc_data[first_pc_data[:, 2] >= 0]
                pc_data = point_process.uniform_sampling_numpy(pc_data[None, :], self.num_points)[0]
                pc.append(pc_data)

                img_main_path.append(os.path.join(file_path, "p0", "rgb", log_files[i].replace('.log', '.png')))
                

        if len(state) == 0:
            raise ValueError("Found no qpos that exceeds the threshold.")
        
        with open(os.path.join(file_path, 'instruction.json'), 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instr = instruction_dict['instruction']
        instr = re.sub(r'(?i)\b(being careful|be careful|taking care|and stay)\b.*', '', instr)
        instr = re.sub(r'(?i)\b(while)\b.*', '', instr)
        if instr[-1] == ' ':
            instr = instr[:-1]
        instr = re.sub(r'[,.;]$', '.', instr)
        encoded_input = tokenizer(instr, return_tensors='pt')
        instr_embed = instr_model(**encoded_input).last_hidden_state[0, 0]

        # Return the resulting sample
        return {
            "state": state,
            "action": action,
            "pc": pc,
            "depth": depth,
            "img_main_path": img_main_path,
            "instruction": instr_embed.detach().numpy(),
        }

    def parse_log_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        result = {}

        bunker_status_start = log_content.find("bunker_status_msg:")
        bunker_status_data = log_content[bunker_status_start:].split("cmd_vel_msg:")[0]
        linear_velocity = float(bunker_status_data.split("linear_velocity:")[1].split("\n")[0].strip())
        angular_velocity = float(bunker_status_data.split("angular_velocity:")[1].split("\n")[0].strip())
        result['bunker_status_msg'] = {'linear_velocity': linear_velocity, 'angular_velocity': angular_velocity}

        cmd_vel_start = log_content.find("cmd_vel_msg:")
        cmd_vel_data = log_content[cmd_vel_start:].split("joint_states_single_msg:")[0]
        linear_x = float(cmd_vel_data.split("Linear: x=")[1].split(",")[0].strip())
        angular_z = float(cmd_vel_data.split("Angular: x=0.0, y=0.0, z=")[1].split("\n")[0].strip())
        result['cmd_vel_msg'] = {'linear_x': linear_x, 'angular_z': angular_z}

        joint_states_single_start = log_content.find("joint_states_single_msg:")
        joint_states_single_data = log_content[joint_states_single_start:].split("end_pose_msg:")[0]
        position = joint_states_single_data.split("position:")[1].split("\n")[0].strip()
        velocity = joint_states_single_data.split("velocity:")[1].split("\n")[0].strip()
        result['joint_states_single_msg'] = {
            'position': np.array(eval(position)),
            'velocity': np.array(eval(velocity))
        }

        joint_states_start = log_content.find("joint_states_msg:")
        joint_states_data = log_content[joint_states_start:].split("effort:")[0]
        position = joint_states_data.split("position:")[1].split("\n")[0].strip()
        velocity = joint_states_data.split("velocity:")[1].split("\n")[0].strip()
        result['joint_states_msg'] = {
            'position': np.array(eval(position)),
            'velocity': np.array(eval(velocity))
        }

        return result


if __name__ == '__main__':
    dataset = MarsmindBaseDataset(
        data_path='/home/imagelab/zys/data/MarsMind_data',
        pad_after=15,
        pad_before=1,
        task_name='move',
        num_points=4096
    )
    print(len(dataset))
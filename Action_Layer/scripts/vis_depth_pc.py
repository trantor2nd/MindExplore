import sys
sys.path.append('/home/imagelab/zys/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

import diffusion_policy_3d.model.vision_3d.point_process as point_process

data_path = '/home/imagelab/zys/data/MarsMind_data'
DISTANCE = 1

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

def save_pc_vis(depth_image, name):
    pc_data = depth_to_lidar(depth_image)
    distances = np.sqrt(pc_data[:, 0]**2 + pc_data[:, 2]**2)
    pc_data = pc_data[distances <= DISTANCE]
    # pc_data = pc_data[pc_data[:, 0] >= 0]
    pc_data = point_process.uniform_sampling_numpy(pc_data[None, :], 4096)[0]

    y = pc_data[:, 0]
    z = - pc_data[:, 1]
    x = pc_data[:, 2]

    fig = plt.figure(figsize=(10, 7))

    view_angles = [
        (30, 30),  # 第一个视角 (azim, elev)
        (90, 30),  # 第二个视角
        (180, 0), # 第三个视角
        (270, 30), # 第四个视角
        (0, 90)    # 第五个视角
    ]

    # 绘制每个视角并保存
    for i, (azim, elev) in enumerate(view_angles):
        ax = fig.add_subplot(111, projection='3d')

        # 绘制散点图
        ax.scatter(x, y, z, c='b', s=1)

        # 设置标题和标签
        ax.set_title(f'LiDAR Point Cloud - View {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置视角
        ax.view_init(elev=elev, azim=azim)

        # 保存图像
        plt.savefig(f'/home/imagelab/zys/data/vis_pc/{name}_view_{i+1}.png', dpi=300)

        # 清除当前图像，以便绘制下一个视角
        ax.cla()

for dir_task in os.listdir(os.path.join(data_path)):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            if 'grasp' in dir_metaaction.lower():
                depth_main_path = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'p0', 'depth')))[0]
                depth_wrist_path = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'p1', 'depth')))[0]
                
                depth_main_image = cv2.imread(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'p0', 'depth', depth_main_path), \
                                              cv2.IMREAD_UNCHANGED)
                depth_wrist_image = cv2.imread(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'p1', 'depth', depth_wrist_path), \
                                              cv2.IMREAD_UNCHANGED)

                save_pc_vis(depth_main_image, f'{dir_episode}_{dir_metaaction}_main')
                save_pc_vis(depth_wrist_image, f'{dir_episode}_{dir_metaaction}_wrist')
import sys
sys.path.append('/home/imagelab/zys/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

import diffusion_policy_3d.model.vision_3d.point_process as point_process

data_path = '/home/imagelab/zys/data/MarsMind_data'

for dir_task in os.listdir(os.path.join(data_path)):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            if 'cross' in dir_metaaction.lower():
                pc_path = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'pc')))[0]
                # 读取CSV文件，假设文件中有 'x', 'y', 'z', 'intensity' 四个字段
                df = pd.read_csv(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'pc', pc_path))

                # 随机采样4096个点
                # sampled_df = df.sample(n=4096, random_state=42)

                pc_data = df[['x', 'y', 'z']].to_numpy()
                # distances = np.sqrt(pc_data[:, 0]**2 + pc_data[:, 1]**2)
                # pc_data = pc_data[distances <= 2]
                # pc_data = pc_data[pc_data[:, 0] >= 0]
                
                # 删除 x 在 (-0.5, 0) 且 y 在 (-0.5, 0.5) 区域内的点
                mask = ~((pc_data[:, 0] > -0.5) & (pc_data[:, 0] < 0) & (pc_data[:, 1] > -0.5) & (pc_data[:, 1] < 0.5))
                pc_data = pc_data[mask]

                # pc_data = point_process.uniform_sampling_numpy(pc_data[None, :], 4096)[0]

                # 提取采样后的点云数据
                x = pc_data[:, 0]
                y = pc_data[:, 1]
                z = pc_data[:, 2]

                # 创建一个 3D 图形
                fig = plt.figure(figsize=(10, 7))

                # 定义视角列表
                view_angles = [
                    (30, 30),  # 第一个视角 (azim, elev)
                    (90, 30),  # 第二个视角
                    (180, 30), # 第三个视角
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
                    plt.savefig(f'/home/imagelab/zys/data/vis_pc/{dir_episode}_{dir_metaaction}_view_{i+1}.png', dpi=300)

                    # 清除当前图像，以便绘制下一个视角
                    ax.cla()
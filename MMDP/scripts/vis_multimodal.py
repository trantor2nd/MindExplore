import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

def save_depth_image(depth_image_path, output_path):
    """ 读取并保存伪彩色深度图像 """
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("无法加载深度图像，请检查路径是否正确")

    # 归一化深度数据到0-255
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)

    cv2.imwrite(output_path, depth_colored)
    print(f"深度图像已保存: {output_path}")

def save_point_cloud_2d_projection(point_cloud_csv_path, output_path):
    """ 读取点云数据并保存为 2D 投影图 """
    df = pd.read_csv(point_cloud_csv_path)
    if not {'x', 'y', 'z', 'intensity'}.issubset(df.columns):
        raise ValueError("CSV文件应包含'x', 'y', 'z', 'intensity'列")

    # 选取 x, y 或 x, z 进行 2D 投影
    x, y, intensity = df['x'].values, df['y'].values, df['intensity'].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=intensity, cmap='viridis', s=1)
    plt.colorbar(label="Intensity")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("2D Projection of Lidar Point Cloud")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"点云2D投影图片已保存: {output_path}")


data_path = '/home/imagelab/zys/data/MarsMind_data'

for dir_task in os.listdir(os.path.join(data_path)):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            if 'cross' in dir_metaaction.lower():

                depth_path = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'p0', 'depth')))[0]
                depth_image_path = os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'p0', 'depth', depth_path)
                depth_output_path = f"/home/imagelab/zys/data/vis_pc/{dir_episode}_{dir_metaaction}_depth.png"

                pc_path = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'pc')))[0]
                point_cloud_csv_path = os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'pc', pc_path)
                point_cloud_output_path = f"/home/imagelab/zys/data/vis_pc/{dir_episode}_{dir_metaaction}_pc.png"

                save_depth_image(depth_image_path, depth_output_path)
                save_point_cloud_2d_projection(point_cloud_csv_path, point_cloud_output_path)

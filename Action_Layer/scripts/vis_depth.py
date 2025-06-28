import cv2
import numpy as np
import os

data_path = '/home/imagelab/zys/data/MarsMind_data'
DISTANCE = 2000

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
                
                # depth_main_image[depth_main_image > DISTANCE] = 0
                # depth_wrist_image[depth_wrist_image > DISTANCE] = 0

                if depth_main_image is None or depth_wrist_image is None:
                    print("Failed to load the depth image.")
                    exit()

                depth_main_normalized = cv2.normalize(depth_main_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_wrist_normalized = cv2.normalize(depth_wrist_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                depth_main_colormap = cv2.applyColorMap(depth_main_normalized, cv2.COLORMAP_JET)
                depth_wrist_colormap = cv2.applyColorMap(depth_wrist_normalized, cv2.COLORMAP_JET)

                cv2.imwrite(f"/home/imagelab/zys/data/vis_depth/{dir_episode}_{dir_metaaction}_depth_main.png", depth_main_colormap)
                cv2.imwrite(f"/home/imagelab/zys/data/vis_depth/{dir_episode}_{dir_metaaction}_depth_wrist.png", depth_wrist_colormap)

import os
import numpy as np

data_path = '/home/imagelab/zys/data/MarsMind_data'

def parse_log_file(file_path):
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

# + forward/left
meta_base = {}

for dir_task in os.listdir(data_path):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            if dir_metaaction.lower() == 'photo' or 'README' in dir_metaaction or 'json' in dir_metaaction or 'pt' in dir_metaaction:
                continue
            
            log_files = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, "logs")))

            for log_file in log_files:
                meta_base_name = ''
                log_data = parse_log_file(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, "logs", log_file))
                action_base_vel_y = log_data['cmd_vel_msg']['linear_x']
                action_base_delta_ang = log_data['cmd_vel_msg']['angular_z']

                if abs(action_base_vel_y) != 0.4 and action_base_vel_y != 0.0 and abs(action_base_vel_y) != 0.2:
                    raise ValueError(f'vel_y is {action_base_vel_y}')
                if abs(action_base_delta_ang) != 1 and action_base_delta_ang != 0.0 and abs(action_base_delta_ang) != 0.6:
                    raise ValueError(f'delta_ang is {action_base_delta_ang}')
                
                if action_base_vel_y == 0.4:
                    meta_base_name += 'forward'
                elif action_base_vel_y == -0.4:
                    meta_base_name += 'backward'
                elif action_base_vel_y == 0.2:
                    meta_base_name += 'soft-forward'
                elif action_base_vel_y == -0.2:
                    meta_base_name += 'soft-backward'
            
                if len(meta_base_name) != 0:
                    meta_base_name += '_'

                if action_base_delta_ang == 1:
                    meta_base_name += 'left'
                elif action_base_delta_ang == -1:
                    meta_base_name += 'right'
                elif action_base_delta_ang == 0.6:
                    meta_base_name += 'soft-left'
                elif action_base_delta_ang == -0.6:
                    meta_base_name += 'soft-right'
                
                meta_base[meta_base_name] = meta_base.get(meta_base_name, 0) + 1

print(meta_base)

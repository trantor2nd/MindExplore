import os
import numpy as np
import json

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
velstr2num = {'forward':0.4,
             'backward':-0.4,
             'left':0,
             'right':0,
             'soft-forward':0.2,
             'soft-backward':-0.2,
             'soft-left':0,
             'soft-right':0,
             'stop':0.0}
angstr2num = {'forward':0,
             'backward':0,
             'left':1,
             'right':-1,
             'soft-forward':0,
             'soft-backward':0,
             'soft-left':0.6,
             'soft-right':-0.6,
             'stop':0.0}
num2velstr = {num: name for name, num in velstr2num.items()}
num2angstr = {num: name for name, num in angstr2num.items()}

for dir_task in os.listdir(data_path):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            if 'move' in dir_metaaction.lower() or 'turn' in dir_metaaction.lower() or 'cross' in dir_metaaction.lower():            
                save_json_path = os.path.join(data_path, dir_task, dir_episode, dir_metaaction, "base_instruction.json")
                log_files = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, "logs")), reverse=True)
                action_seq = ''
                cur_act = 'stop'
                cur_act_cnt = 1

                base_instruction = {}

                for log_file in log_files:
                    log_data = parse_log_file(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, "logs", log_file))
                    action_base_vel_y = log_data['cmd_vel_msg']['linear_x']
                    action_base_delta_ang = log_data['cmd_vel_msg']['angular_z']
                    
                    replace_act = False

                    if '_' in cur_act:
                        vel_act, ang_act = cur_act.split('_')
                        if action_base_vel_y != velstr2num[vel_act] or action_base_delta_ang != angstr2num[ang_act]:
                            replace_act = True
                    else:
                        if action_base_vel_y != velstr2num[cur_act] or action_base_delta_ang != angstr2num[cur_act]:
                            replace_act = True

                    if replace_act:
                        vel_act = num2velstr[action_base_vel_y]
                        ang_act = num2angstr[action_base_delta_ang]
                        if not (vel_act == 'stop' and ang_act == 'stop'):
                            action_seq = f'{cur_act}:{cur_act_cnt},' + action_seq
                            if vel_act == 'stop':
                                cur_act = ang_act
                            elif ang_act == 'stop':
                                cur_act = vel_act
                            else:
                                cur_act = vel_act + '_' + ang_act
                            cur_act_cnt = 1                       
                    else:
                        cur_act_cnt += 1

                    write_action_seq = f'{cur_act}:{cur_act_cnt},' + action_seq
                    
                    base_instruction[log_file[:-4]] = write_action_seq
                    
                with open(save_json_path, 'w', encoding='utf-8') as outfile:
                    json.dump(base_instruction, outfile, ensure_ascii=False, indent=4)

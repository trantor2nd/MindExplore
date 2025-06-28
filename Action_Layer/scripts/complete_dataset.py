import os
import shutil

data_path='/home/imagelab/zys/data/MarsMind_data'

def complete_files(pname, mode):
    logs = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'logs')))
    if pname == 'pc':
        p_files = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, pname)))
    else:
        p_files = sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, pname, mode)))
    
    if p_files[0][:-3] == logs[0][:-3] or len(p_files) > len(logs):
        return
    if len(p_files) == len(logs):
        raise ValueError('What?')
    i = 0
    while p_files[0][:-3] != logs[i][:-3]:
        shutil.copy(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, pname, mode, p_files[0]), 
                    os.path.join(data_path, dir_task, dir_episode, dir_metaaction, pname, mode, logs[i].replace('.log', '.'+p_files[0][-3:])))
        print(f'copy {os.path.join(data_path, dir_task, dir_episode, dir_metaaction, pname, mode, p_files[0])} \n to'
            f'{os.path.join(data_path, dir_task, dir_episode, dir_metaaction, pname, mode, logs[i].replace(".log", "."+p_files[0][-3:]))}\n')
        i += 1
    

for dir_task in os.listdir(os.path.join(data_path)):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            if 'md' in dir_metaaction.lower() or 'photo' in dir_metaaction.lower() or '.pt' in dir_metaaction.lower() or 'json' in dir_metaaction.lower():
                continue
            complete_files('p0', 'rgb')
            complete_files('p0', 'depth')
            complete_files('p1', 'rgb')
            complete_files('p1', 'depth')
            complete_files('pc', None)
import os
import json
import re

data_path = '/home/imagelab/zys/data/MarsMind_data'
task_name = 'turn'
task_instr = set()

grasp_list = ['bottle', 'wood', 'branch', 'rock', 'stone']
grasp_cnt = 0
place_list = ['bottle', 'wood', 'branch', 'rock', 'stone']
place_cnt = 0
move_list = ['hill', 'forward', 'boundary', 'container', 'sampling_point']
move_cnt = 0
cross_list = ['hill', 'rock', 'wood']
cross_cnt = 0
turn_list_1 = ['left', 'right']
turn_list_2 = ['90', '360']
turn_cnt = 0

confuse_instr = ['OTO',
                 'Grasp the stone block located at the flat hill. Then lift the stone block. Be careful not to grip too tightly or too loosely.',
                 'Grasp the branch located at the flat hill. Then lift the branch. Be careful not to grip too tightly or too loosely.',
                 'Grasp the wooden block located at the flat hill. Then lift the wooden block. Be careful not to grip too tightly or too loosely.',
                 'Grasp the bottle located at the flat hill. Then lift the bottle. Be careful not to grip too tightly or too loosely. Then turn right in place, go around the flat mound of sand until the cyan container is detected, then move to the cyan container where the bottle should be placed.',
                 'Place the bottle into the pink container and retract arm, being careful not to put it in too fast or too slow.',
                 ]

for dir_task in os.listdir(os.path.join(data_path)):
    if 'txt' in dir_task:
        continue
    for dir_episode in os.listdir(os.path.join(data_path, dir_task)):
        for dir_metaaction in sorted(os.listdir(os.path.join(data_path, dir_task, dir_episode))):
            
            # objects_cnt = 0
            # if 'grasp' in dir_metaaction.lower():
            #     for objects in grasp_list:
            #         if objects in dir_metaaction.lower():
            #             objects_cnt += 1
            #             grasp_cnt += objects_cnt
            #             break
            #     if objects_cnt != 1:
            #         raise ValueError('aa')
            # if 'place' in dir_metaaction.lower():
            #     for objects in place_list:
            #         if objects in dir_metaaction.lower():
            #             objects_cnt += 1
            #             place_cnt += objects_cnt
            #             break
            #     if objects_cnt != 1:
            #         raise ValueError('aa')
            # if 'move' in dir_metaaction.lower():
            #     for objects in move_list:
            #         if objects in dir_metaaction.lower():
            #             objects_cnt += 1
            #             move_cnt += objects_cnt
            #             break
            #     if objects_cnt != 1:
            #         raise ValueError('aa')
            # if 'cross' in dir_metaaction.lower():
            #     for objects in cross_list:
            #         if objects in dir_metaaction.lower():
            #             objects_cnt += 1
            #             cross_cnt += objects_cnt
            #             break
            #     if objects_cnt != 1:
            #         raise ValueError('aa')
            # if 'turn' in dir_metaaction.lower():
            #     for objects in turn_list_1:
            #         if objects in dir_metaaction.lower():
            #             objects_cnt += 10
            #             break
            #     for objects in turn_list_2:
            #         if objects in dir_metaaction.lower():
            #             objects_cnt += 10
            #             break
            #     if objects_cnt == 20:
            #         turn_cnt += 1
            #         objects_cnt = 1
            #     if objects_cnt != 1:
            #         raise ValueError('aa')


            if task_name in dir_metaaction.lower():
                with open(os.path.join(data_path, dir_task, dir_episode, dir_metaaction, 'instruction.json'), 'r') as f_instr:
                    instruction_dict = json.load(f_instr)
                instr = instruction_dict['instruction']
                if instr in confuse_instr:
                    print(os.path.join(data_path, dir_task, dir_episode, dir_metaaction))
                instr = re.sub(r'(?i)\b(being careful|be careful|taking care|and stay)\b.*', '', instr)
                instr = re.sub(r'(?i)\b(while)\b.*', '', instr)
                if instr[-1] == ' ':
                    instr = instr[:-1]
                instr = re.sub(r'[,.;]$', '.', instr)
                task_instr.add(instr)

print(len(task_instr))
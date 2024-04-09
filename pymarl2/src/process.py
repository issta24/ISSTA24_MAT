from ast import literal_eval
import numpy as np
state_won_all = []

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    
    if norms == 0:
        return 0
        
    similarity = dot_product / (norms + 1e-8) # 避免除数为0时发生错误
    
    return similarity

def _compute_distance_behavior_states(y1, y2):
    """
    y1 is a list
    """
    # y is numpy
    y1_length = len(y1)
    y2_length = len(y2)

    coverage_score = abs(y1_length - y2_length)

    common_length = min(y1_length, y2_length)
    y1_common = y1[:common_length]
    y2_common = y2[:common_length]

    # if len(y1) == len(y2):
    #     coverage_score = cosine_similarity(y1_common,y2_common)
    # else:
    #     coverage_score = cosine_similarity(y1_common,y2_common) / abs(len(y1) - len(y2))
    for i in range(common_length):
        y1_e = y1_common[i]
        y2_e = y2_common[i]
        if y1_e == y2_e:
            continue
        else:
            coverage_score += 1

    coverage_score /= float(max(y1_length, y2_length)) 
    # coverage_score /= 4

    return coverage_score

def unfold_list(y_list):
    unfold_y_list = []

    for y in y_list:
        for element in y:
            unfold_y_list.append(element)
    return unfold_y_list



def process_pos_element(state):
    new_state = []
    for time_step_state in state:
        step = []
        for i in range(4):
            if i == 1 or i == 2:
                step.append(round(time_step_state[i]))
            else:
                step.append(time_step_state[i])
        new_state.append(step)
    return unfold_list(new_state)

def isNovel(state1, state2, en_id):
    en_state1 = []
    for time_step_state in state1:
        en_state1.append(time_step_state[en_id])

    en_state2 = []
    for time_step_state in state2:
        en_state2.append(time_step_state[en_id])

    distance = _compute_distance_behavior_states(process_pos_element(en_state1),process_pos_element(en_state2))

    return distance

def main():
    import os,glob
    filenames = glob.glob(os.path.join('/home/dell/mxy/pymarl2_2/pymarl2/results/1c3s5z/mutate_info/2024-2-2-9-0/', "*.txt"))

    for f in filenames:
        state_path = f
        with open(state_path,'r') as f_state:
            lines = f_state.readlines()
            for line in lines:
                my_list = literal_eval(line)
                state_won_all.append(my_list)

    dis_list = []
    for i in range(len(filenames)-1):
        for j in range(i+1,len(filenames)):
            s1 = state_won_all[i]
            s2 = state_won_all[j]
            for en_id in range(9):
                dis_list.append(isNovel(s1,s2,en_id))
    # print(dis_list)
    print(sum(dis_list)/len(dis_list))


    state_won_all_old = []
    
    #
    #old_state_paths = glob.glob(os.path.join('/home/dell/mxy/pymarl2_2/pymarl2/results/1c3s5z/init_seed_info_2/', "enemy_state*.txt"))
    old_state_paths = glob.glob(os.path.join('/home/dell/mxy/pymarl2_2/pymarl2/results/1c3s5z/DRL/2024-1-31-10-11/', "*.txt"))

    for old_state_path in old_state_paths:
        with open(old_state_path,'r') as f_state:
            lines = f_state.readlines()
            for line in lines:
                my_list = literal_eval(line)
                state_won_all_old.append(my_list)

    dis_list = []
    for new_state in state_won_all:
        for old_state in state_won_all_old:
            for en_id in range(9):
                dis_list.append(isNovel(new_state,old_state,en_id))
    print(sum(dis_list)/len(dis_list))


    dis_list = []
    for i in range(len(state_won_all_old)-1):
        for j in range(i+1,len(state_won_all_old)):
            s1 = state_won_all_old[i]
            s2 = state_won_all_old[j]
            for en_id in range(9):
                dis_list.append(isNovel(s1,s2,en_id))
    print(sum(dis_list)/len(dis_list))

main()
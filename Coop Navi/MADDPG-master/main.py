from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
from ast import literal_eval
from graph import TeamGraph
from wwl import compute_wasserstein_distance,compute_wl_propagation_aggregation



def seed_corpus(corpus_total):
    init_seed = {}
    

    #Simple_tag
    init_seed[0] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[1] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[2] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[3] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[4] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[5] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[6] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[7] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[8] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}
    init_seed[9] = {'num_landmarks':1, 'size': 0.2, 'adv':[]}



    with open('init_action_adv_simple_adv.txt','r') as f:
        lines = f.readlines()
        seed_num = 0
        for line in lines:
            my_list = literal_eval(line)
            init_seed[seed_num]['adv'] = my_list
        
            seed_num += 1


    for i in range(10):
        corpus_total.append(init_seed[i])
    
    
    
    return corpus_total

def mutation(seed_origin):
    seed_mutate = {}

    num_landmarks = 2
    p_pos = []
    size = 0.0

    for i in range(num_landmarks):
        p_pos.append(np.random.uniform(-0.9, +0.9, 2))
        size = random.uniform(0,0.5)
    
    adv = seed_origin['adv']
    adv_mutate = []
    r = random.randint(0,len(adv) - 1)
    for i in range(len(adv)):
        if i == r:
            action_adv = [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
            adv_mutate.append(action_adv)
        else:
            adv_mutate.append(adv[i])


    seed_mutate['num_landmarks'] = num_landmarks
    seed_mutate['p_pos'] = p_pos
    seed_mutate['size'] = size
    seed_mutate['adv'] = adv


    return seed_mutate


def update_energy(selected_seed_index, seed_collision,n_collision, collision, is_fail,ind_dis, delta_wl):

    delta = 0

    failed = seed_collision[selected_seed_index][0]
    benign = seed_collision[selected_seed_index][1]

    if is_fail:
        delta_ef = failed / (failed + benign)
    else:
        delta_ef = - 0.1 * benign / (failed + benign)


    O_s_minus = abs(n_collision - collision)

    delta_ev = O_s_minus / (1 - (ind_dis + delta_wl) /2 + 10**(-5))

    delta = 0.5 * delta_ef + 0.5 * np.tanh(delta_ev) - 0.05 * 1

    return delta,delta_ev


def select_seed(seeds_energy,corpus_total):
    select_probabilities = []
    corpus_index = []
    for i in range(len(corpus_total)):
        element_energy = seeds_energy[i]
        if element_energy < 0.0:
            element_energy = 0.0
        select_probabilities.append(element_energy)
        corpus_index.append(i)

    select_probabilities = np.array(select_probabilities)
    select_probabilities /= select_probabilities.sum()
    
    
    select_index = np.random.choice(corpus_index, p=select_probabilities)

    return select_index



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
    for i in range(common_length):
        y1_e = y1_common[i]
        y2_e = y2_common[i]
        
        if y1_e == y2_e:
            continue
        else:
            coverage_score += 1

    coverage_score /= float(max(y1_length, y2_length))
    
    return coverage_score

def process(seq, a_id, evaluate_episode_len):
    seq_a_id = []
    for i in range(evaluate_episode_len):
        for j in range(1,3):
            seq_a_id.append(round(seq[i][a_id][j][0],2))
            seq_a_id.append(round(seq[i][a_id][j][1],2))
    return seq_a_id


def indDiversity(game_state_list, seq, a_id,evaluate_episode_len):
    dis_list = [1.0]

    for i in range(len(game_state_list)):
        dis = _compute_distance_behavior_states(process(game_state_list[i],a_id,evaluate_episode_len),process(seq,a_id,evaluate_episode_len))
        dis_list.append(dis)


    return min(dis_list)


if __name__ == '__main__':
    # get the params
    args = get_args()
    
    # init
    seed_energy = {}
    corpus_total = []
    seed_collision = {}
    state_seq_list = []
    graph_vector_dict = {}
    collision_num = [51,53,59,53,69,57,58,53,57,54]
    corpus_total = seed_corpus(corpus_total)
    for i in range(len(corpus_total)):
        seed_energy[i] = 1.0
        seed_collision[i] = [0,0]
    
    with open('init_state_simple_adv.txt','r') as f:
        lines = f.readlines()
        seed_num = 0
        for line in lines:
            my_list = literal_eval(line)
            state_seq_list.append(my_list)

            vector_list = []
            for i in range(args.evaluate_episode_len):
                if (i+1) % 5 == 0:
                    pos_list = []
                    for j in range(4):
                        pos_list.append([j,my_list[i][j][1][0],my_list[i][j][1][1],my_list[i][j][2][0],my_list[i][j][2][1]])
                    init_team_graph = TeamGraph(4,pos_list)
                    init_team_graph.set_team_graph()
                    init_labels,weights = init_team_graph.re_label_info()
                    vector_list.append(compute_wl_propagation_aggregation(4,init_labels,args.iterations,weights,4)[args.iterations - 1])
            graph_vector_dict[seed_num] = vector_list
            seed_num += 1
    vector_num = len(graph_vector_dict)
    init_wl_dis_list = []
    for i in range(vector_num-1):
        for j in range(i+1,vector_num):
            for k in range(5):
                wl_dis = compute_wasserstein_distance(graph_vector_dict[i][k],graph_vector_dict[j][k],args.iterations,sinkhorn=False,discrete=False)
                init_wl_dis_list.append(wl_dis)
    init_wl_dis = sum(init_wl_dis_list)/len(init_wl_dis_list)


    for i in range(args.evaluate_episodes):

        selected_seed_index = select_seed(seed_energy, corpus_total)
        
        seed = corpus_total[selected_seed_index]
        n_collision = collision_num[selected_seed_index]
        mutate_seed = mutation(seed)
        #print('seed:',mutate_seed)
        
        #mutate_seed = corpus_total[i]
        #mutate_seed = corpus_total[i]
        env, args = make_env(args,mutate_seed)
        runner = Runner(args, env)
        returns, collision, game_state,reward_list = runner.evaluate(mutate_seed)
        max_r = max(reward_list)
        max_r_step = reward_list.index(max_r)
        select_list = [max_r_step-4,max_r_step-2,max_r_step,max_r_step+2,max_r_step+4]

        print('Average returns is', returns)
    

        #评估多样性
        #Win Rate
        if collision > 0:
            seed_collision[selected_seed_index][0] += 1
        else:
            seed_collision[selected_seed_index][1] += 1

        #Individual Diversity
        dis_list = []
        for i in range(4):
            dis = indDiversity(state_seq_list, game_state, i, args.evaluate_episode_len)
            dis_list.append(dis)
        ind_dis = sum(dis_list)/len(dis_list)

        #Team Diversity
        vector_list = []
        for j in range(args.evaluate_episode_len):
            if j in select_list:
                state_timestep = game_state[j]
                pos_list = []
                for k in range(4):
                    pos_list.append([j,state_timestep[k][1][0],state_timestep[k][1][1],state_timestep[k][2][0],state_timestep[k][2][1]])
                team_graph = TeamGraph(4,pos_list)
                team_graph.set_team_graph()
                labels,weights = team_graph.re_label_info()
                vector = compute_wl_propagation_aggregation(4,labels,args.iterations,weights,4)[args.iterations - 1]

                vector_list.append(vector)
        
        wl_dis_list = []
        for i in range(len(graph_vector_dict)):
            for j in range(len(vector_list)):
                wl_dis = compute_wasserstein_distance(vector_list[j],graph_vector_dict[i][j],args.iterations,sinkhorn=False,discrete=False)
                wl_dis_list.append(wl_dis)
        
        wl_dis_avg = sum(wl_dis_list)/len(wl_dis_list)

        delta_wl = wl_dis_avg / init_wl_dis

        dis_all = (ind_dis + delta_wl) / 2

        
        is_fail = False
        if collision:
            is_fail = True
        
        delta,delta_ev = update_energy(selected_seed_index,seed_collision,n_collision, collision,is_fail, ind_dis, delta_wl)

        seed_energy[selected_seed_index] += delta

        if is_fail and dis_all > 0.5:
            seed_energy[len(corpus_total)] = 1 + 0.5 * np.tanh(delta_ev)
            collision_num.append(collision)
            seed_collision[len(corpus_total)] = [0,0]
            corpus_total.append(mutate_seed)
            
            
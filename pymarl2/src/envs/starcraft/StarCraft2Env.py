from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv
from envs.starcraft.smac_maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb
import random
import time
import math
from ast import literal_eval
from envs.starcraft.element import CorpusElement
from envs.starcraft.graph import TeamGraph
from envs.starcraft.node import TeamNode
from envs.starcraft.graph2vec import main_graph
from envs.starcraft.wwl import compute_wl_propagation_aggregation,compute_wasserstein_distance
from envs.starcraft.abs.state_abstracter import StateAbstracter
import numpy as np
import datetime
import os


races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

#state_file = open('results/4-20/state_file_C0.txt','a')
#state_file_diff = open('results/state_file_diff_C0_1.txt','a')
state_sequence_all = []
game_won_dict = {}
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
    for i in range(common_length):
        y1_e = y1_common[i]
        y2_e = y2_common[i]
        
        if y1_e == y2_e:
            continue
        else:
            coverage_score += 1

    coverage_score /= float(max(y1_length, y2_length))


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
def isNovel(state, en_id):
    distance_list = [1.0]

    en_state = []
    for time_step_state in state:
        en_state.append(time_step_state[en_id])

    for old_state in state_won_all:
        old_en_state = []
        for time_step_state in old_state:
            old_en_state.append(time_step_state[en_id])
        distance = _compute_distance_behavior_states(process_pos_element(old_en_state),process_pos_element(en_state))
        distance_list.append(distance)
    return min(distance_list)

def dis_s_s(state,en_id,select_seed):
    en_state = []
    for time_step_state in state:
        en_state.append(time_step_state[en_id])
    
    old_state = state_won_all[select_seed]

    old_en_state = []
    
    for time_step_state in old_state:
        old_en_state.append(time_step_state[en_id])
    distance = _compute_distance_behavior_states(process_pos_element(old_en_state),process_pos_element(en_state))

    return distance



def get_init_dis(state1,state2,en_id):
    en_state1 = []
    for time_step_state in state1:
        en_state1.append(time_step_state[en_id])

    en_state2 = []
    for time_step_state in state2:
        en_state2.append(time_step_state[en_id])

    distance = _compute_distance_behavior_states(process_pos_element(en_state1),process_pos_element(en_state2))

    return distance


class StarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        state_last_action=True,
        state_timestep_number=False,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]
        print('shield:',self.shield_bits_ally)

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.min_x = 0
        self.min_y = 0
        self.map_x = 0
        self.map_y = 0
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        #My data
        self.n_en_state = 0 #记录enemy的状态数量
        self.en_state = []  #记录enemy的状态
        self.en_battle_state = [] #记录enemy32场比赛的状态序列
        self.en_this_battle = []
        self.index = 0
        self.random_flag = [0,0]  #约束的掩码
        #self.corpus_seeds = {}  # save un-failed index
        self.seeds_energy = {}  # save the energy of each seed
        self.num_seed = 10 # number of seeds
        self.corpus_total = [] # save all elements in the fuzzing
        self.corpus_index = [] # save the index of all elements
        self.num_seed_select = {}
        # self.test_nepisode = 10
        self.test_nepisode_index = [0] *2000
        self.actions_json_origin = []
        self.actions_mutate = []
        self.seed_index = -1    #被选择的seed，即将进行mutate
        self.graph_list = []    #所有场次的凸包
        self.area_list = []     #当前场次的凸包
        self.result_path = None
        self.battle_graph_list = []    #当前对战的图
        self.vector_list = []   #所有对战的图的向量表示
        self.reward_list = []
        self.reward_json = {}   #每场对战的reward_list
        self.max_reward_json = {}    #每场对战得到最大reward的step
        self.iterations = 2
        self.init_wl_dis = 0
        self.init_dis = 0
        self.id_pos_dict = {}
        self.id_id_dict = {}    #环境随机id-定义id的映射
        self.mutation_gain = []
        self.select_agent = -1
        self.select_phase = -1
        self.count_table = []
        self.health_minus = []
        self.gain_dict = {} #每个格子的gain
        self.gain_dict_c = {}
        self.gain_dict_s = {}
        self.gain_dict_z = {}
        self.top_10_id = [] #格子前10%
        self.mutate_state_id = [] #记录每局被扰动的状态所占的格子id
        self.mutate_agent_id = [] #记录每局被扰动的智能体id
        self.mutate_agent_state = [] #记录每局被扰动的智能体id-格子id对
        self.num_abs_table_select = {} #记录每个格子的输赢次数
        self.num_abs_table_select_c = {} #记录每个格子的输赢次数
        self.num_abs_table_select_s = {} #记录每个格子的输赢次数
        self.num_abs_table_select_z = {} #记录每个格子的输赢次数
        self.battle_state_id = []   #记录被测智能体出现过的state_id
        self.battle_type_state = [] #记录被测智能体出现过的type-state_id
        self.enemy_state_dict = {}
        self.enemy_state_dict_c = {}
        self.enemy_state_dict_s = {}
        self.enemy_state_dict_z = {}
        self.action_type = {}
        self.battle_agent_state_id_visited = []   #对战结束后更新gain_dict所用，记录被统计过的agent-state
        self.grid_dict_c = {}
        self.grid_dict_s = {}
        self.grid_dict_z = {}
        

        self.state_abstracter = None

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())
    
    def find_position(self,list):
        colossus_index = []
        stalkers_index = []
        zealots_index = []
        en_id_index = []
        for i in range(len(list)):
            if list[i] == 4 or list[i] == 1970:
                colossus_index.append(i)
            elif list[i] == 74 or list[i] == 1971:
                stalkers_index.append(i)
            elif list[i] == 73 or list[i] == 1972:
                zealots_index.append(i)
        
        en_id_index = colossus_index + stalkers_index + zealots_index
        return colossus_index,stalkers_index,zealots_index, en_id_index
    
    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)
        print('map name:',self.map_name)
        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(window_size=self.window_size, want_rgb=False)
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
            realtime=False,
            random_seed=self._seed)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties[self.difficulty])
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.min_x = map_play_area_min.x
        self.min_y = map_play_area_min.y
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                    self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data))
                .reshape(self.map_x, self.map_y)), 1) / 255


        # id_pos_dict = {}
        self.id_pos_dict[0] = [22.859619140625, 15.9970703125]
        self.id_pos_dict[1] = [22.815673828125,15.99609375]
        self.id_pos_dict[2] = [22.815673828125,14.8076171875] 
        self.id_pos_dict[3] = [24.0126953125,15.980224609375] 
        self.id_pos_dict[4] = [23.513427734375,16.9189453125] 
        self.id_pos_dict[5] = [22.359619140625,16.9970703125] 
        self.id_pos_dict[6] = [21.359619140625,16.4970703125] 
        self.id_pos_dict[7] = [22.859619140625,17.9970703125] 
        self.id_pos_dict[8] = [21.859619140625,15.4970703125]

        for i in range(9):
            self.id_id_dict[i] = i
        
        reward_path = 'results/1c3s5z/init_seed_info_2/reward.txt'
        with open(reward_path,'r') as f_r:
            lines = f_r.readlines()
            i = 0
            for line in lines:
                my_list = literal_eval(line)
                self.reward_json[i] = my_list
                max_r = max(my_list)
                self.max_reward_json[i] = my_list.index(max_r) 
                i += 1
        
        for seed in range(10):
            pos_path = 'results/1c3s5z/init_seed_info_2/pos_info_'+str(seed)+'.txt'
            with open(pos_path,'r') as f_p:
                lines = f_p.readlines()
                i = 0
                for line in lines:
                    if i == self.max_reward_json[seed]:
                        pos_info = literal_eval(line)
                        int_team_graph = TeamGraph(len(pos_info),pos_info)
                        int_team_graph.set_team_graph()
                        init_labels,weights = int_team_graph.re_label_info()
                        self.vector_list.append(compute_wl_propagation_aggregation(len(pos_info),init_labels,self.iterations,weights,3)[self.iterations - 1])
                    i += 1
        vector_num = len(self.vector_list)
        init_wl_dis = []
        for i in range(vector_num-1):
            for j in range(i+1,vector_num):
                wl_dis = compute_wasserstein_distance(self.vector_list[i],self.vector_list[j],self.iterations,sinkhorn=False,discrete=False)
                init_wl_dis.append(wl_dis)
        self.init_wl_dis = sum(init_wl_dis)/len(init_wl_dis)
        print('wl_dis:',self.init_wl_dis)


        for seed in range(1,11):
            state_path = 'results/1c3s5z/init_seed_info_2/enemy_state_'+str(seed)+'.txt'
            with open(state_path,'r') as f_state:
                lines = f_state.readlines()
                for line in lines:
                    my_list = literal_eval(line)
                    state_won_all.append(my_list)
        
        init_dis_list = []
        for i in range(len(state_won_all)-1):
            for j in range(i+1,len(state_won_all)):
                s1 = state_won_all[i]
                s2 = state_won_all[j]
                for en_id in range(9):
                    init_dis_list.append(get_init_dis(s1,s2,en_id))
        self.init_dis = sum(init_dis_list)/len(init_dis_list)
        print('ind_dis:',self.init_dis)

        for i in range(self.n_agents):
            self.mutation_gain.append([1.0,1.0,1.0])
            self.count_table.append([0,0,0])


        current_datetime = datetime.datetime.now()
        current_year = current_datetime.year
        current_month = current_datetime.month
        current_day = current_datetime.day
        current_hour = current_datetime.hour
        current_minute = current_datetime.minute

        self.result_path = 'results/1c3s5z/mutate_info/'+str(current_year)+'-'+str(current_month)+'-'+str(current_day)+'-'+str(current_hour)+'-'+str(current_minute)

        os.makedirs(self.result_path, exist_ok=True)

        """create state abstracter"""
        self.state_abstracter = StateAbstracter(raw_state_dim=69,state_dim=5,state_lb=0.0,state_ub=1.0,action_dim=1,reduction=True)
        self.enemy_state_abstracter = StateAbstracter(raw_state_dim=3,state_dim=3,state_lb=0.0,state_ub=1.0,action_dim=1,reduction=False)
        
        state_dimensions = 5**5
        # self.gain_dict = [0.0] * state_dimensions
        # self.num_abs_table_select = [0,0] * state_dimensions
        for i in range(state_dimensions):
            self.gain_dict[i] = 0.0
            self.gain_dict_c[i] = 0.0
            self.gain_dict_s[i] = 0.0
            self.gain_dict_z[i] = 0.0
            self.num_abs_table_select_c[i] = [0,0]
            self.num_abs_table_select_s[i] = [0,0]
            self.num_abs_table_select_z[i] = [0,0]
        
        for i in range(5**3):
            self.enemy_state_dict[i] = 0
        
        self.action_type['dead'] = [0]
        self.action_type['stop_move'] = [1,2,3,4,5]
        attack_action_id = []
        for i in range(6,self.n_actions - 1):
            attack_action_id.append(i)
        self.action_type['attack_heal'] = attack_action_id

    def get_action_type(self,action_id):
        if action_id == 0:
            return 'dead'
        elif (action_id >= 1 and action_id <= 5):
            return 'stop_move'
        else:
            return 'attack_heal'


    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one. """
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1
    
        
    
    def select_seed(self):
        """select a seed from corpus_total"""

        select_probabilities = []
        for i in range(len(self.corpus_total)):
            #element = self.corpus_total[i]
            #element_energy = element.comput_energy(self.episode_limit)
            element_energy = self.seeds_energy[i]
            if element_energy < 0.0:
                element_energy = 0.0
            select_probabilities.append(element_energy)

        select_probabilities = np.array(select_probabilities)
        #print('select_probabilities:',select_probabilities)
        select_probabilities /= select_probabilities.sum()
        select_index = np.random.choice(self.corpus_index, p=select_probabilities)
        #select_index = np.random.choice(list(range(10)))
        #print('select_seed:',select_index)

        
        actions_json = self.corpus_total[select_index].get_actions()

        self.seed_index = select_index

        return actions_json
    
    def mutate_seed(self, actions):
        #Mutate seed according to the reward of selected seed
        seed_id = self.seed_index
        seed_reward = self.reward_json[seed_id]

        seed_reward_process = []
        for s in seed_reward:
            seed_reward_process.append(s)


        #print(seed_reward)

        rm_iterations = len(seed_reward_process) // 2
        rm_num_head = 0
        rm_num_tail = 0
        for i in range(rm_iterations):
            if seed_reward_process[0] == 0.0:
                seed_reward_process = seed_reward_process[1:]
                rm_num_head += 1
            if seed_reward_process[-1] == 0.0:
                seed_reward_process = seed_reward_process[:-1]
                rm_num_tail += 1

        seed_r_min = min(seed_reward_process)
        r_min_index = seed_reward_process.index(seed_r_min)

        r_min_index += rm_num_head  #得到进行mutate的step


        steps = len(actions)

        total_actions = self.get_total_actions()
        
        # mutate_steps = []   #每个agent进行扰动的step
        mutate_actions = []
        for i in range(self.n_agents):
            # random_step = random.randint(0,steps-1)
            # mutate_steps.append(random_step)
            random_action = random.randint(0,total_actions-1)
            mutate_actions.append(random_action)
    

        actions_mutate = {}
        for i in range(steps):
            action_origin = actions[i]
            action_temp = []
            for j in range(9):
                if i == r_min_index:
                    action_temp.append(mutate_actions[j])
                else:
                    action_temp.append(action_origin[j])

            actions_mutate[i] = action_temp
        return actions_mutate

    def get_muate_info(self):
        select_probabilities_ap = []    #action_phase
        for i in range(self.n_agents):
            for j in range(3):
                select_probabilities_ap.append(max(self.mutation_gain[i][j],0.0))
        
        gene_list = list(range(3 * self.n_agents))

        #print('probabilities_ap:',select_probabilities_ap)
        select_probabilities_ap = np.array(select_probabilities_ap)
        select_probabilities_ap /= select_probabilities_ap.sum()
        
        #select one gene
        select_gene = 0
        if self.battles_game < 27:
            select_gene = self.battles_game
        else:
            select_gene = np.random.choice(gene_list, p=select_probabilities_ap)
        
        #Compute the agent_id and phase
        select_agent = int(select_gene) // 3
        select_phase = (int(select_gene) + 1) % 3
        
        if select_phase == 0:
            select_phase = 3
        self.select_agent = select_agent
        self.select_phase = select_phase

        self.count_table[select_agent][select_phase - 1] += 1

        return select_agent,select_phase


    def update_mutation_gain_table(self,game_end_code,ind_feedback,team_feedback):
        a_id = self.select_agent
        phase = self.select_phase - 1
        gain = self.mutation_gain[a_id][phase]
        
        gain_delta = game_end_code + ind_feedback + team_feedback

        self.mutation_gain[a_id][phase] = gain + gain_delta - 0.01
    


    def mutate_seed_table(self,actions):
        #actions == {step:action}
        #Mutate seed according to the mutation-gain table
        steps = len(actions)    #action所包含的时间步

        total_actions = self.get_total_actions()

        mutate_method = random.randint(0,2) #确定扰动的种类

        #print('mutate_method:',mutate_method)

        select_agent,select_phase = self.get_muate_info()  #确定扰动的智能体和扰动阶段

        # actions_phase_1 = {}
        # actions_phase_2 = {}
        # actions_phase_3 = {}

        select_steps = []
        if select_phase == 1:
            select_steps = list(range(0,steps//3))
        if select_phase == 2:
            select_steps = list(range(steps//3,2 * steps//3))
        if select_phase == 3:
            select_steps = list(range(2*steps//3,steps))

        # for i in range(steps):
        #     action_origin = actions[i]

        actions_mutate = {} #扰动后的所有action

        # mutate_method = 0 

        if mutate_method == 0:  #Gene Mutate
            select_mutate_step = random.choice(select_steps)

            random_action = random.randint(0,total_actions-1)

            for i in range(steps):
                action_origin = actions[i]
                action_temp = []
                for j in range(self.n_agents):
                    action_temp.append(action_origin[j])
                if i == select_mutate_step:
                    action_temp[select_agent] = random_action
            
                actions_mutate[i] = action_temp
        elif mutate_method == 1: #Crossover
            a_id_others = []
            for i in range(self.n_agents):
                if i != select_agent:
                    a_id_others.append(i)
            a_id_partner = random.choice(a_id_others)

            select_mutate_step = random.choice(select_steps[:-4])
            crossover_steps = list(range(select_mutate_step,select_mutate_step+5))

            for i in range(steps):
                action_origin = actions[i]
                action_temp = []

                for j in range(self.n_agents):
                    action_temp.append(action_origin[j])
                
                if i in crossover_steps:
                    action_temp[select_agent] = actions[i][a_id_partner]
                
                actions_mutate[i] = action_temp

            

        elif mutate_method == 2: #Shuffle
            select_mutate_step = random.choice(select_steps[:-4])

            shuffle_steps = list(range(select_mutate_step,select_mutate_step+5))
            random.shuffle(shuffle_steps)

            for i in range(steps):
                action_origin = actions[i]
                action_temp = []

                for j in range(self.n_agents):
                    action_temp.append(action_origin[j])

                if i in shuffle_steps:
                    action_temp[select_agent] = actions[shuffle_steps[i-select_mutate_step]][select_agent]
                
                actions_mutate[i] = action_temp

        return actions_mutate





    def add_seed(self,game_end_code):
        #print('-----Add seed-----')
        element = CorpusElement(self.num_seed,1.0,self.seed_index)
        self.corpus_index.append(self.num_seed)
        self.corpus_total.append(element)
        element.set_actions(self.actions_mutate)
        self.num_seed += 1

        energy = element.comput_energy(self.episode_limit,game_end_code)
        self.seeds_energy[element.get_id()] = energy

        self.num_seed_select[element.get_id()] = [0,0]

    def get_top_10(self,gain_dict):   #得到gain为前10%的格子id
        #print('gain_dict:',self.gain_dict)
        items = list(gain_dict.items())
        random.shuffle(items)
        new_dict = dict(items)
        
        sorted_gain = sorted(new_dict.items(), key=lambda x:x[1], reverse=True)
        
        sorted_gain = dict(sorted_gain)
        top_10_id = []
        top_10_dict = {}
        for i in range(len(gain_dict) // 300):
            key = list(sorted_gain.keys())[i]
            value = sorted_gain[key]
            top_10_id.append(key)
            top_10_dict[key] = value
        
        #print(top_10_dict)
        
        return top_10_id

    def mutate_action(self,mutate_al_id,actions_int):
        actions_mutate = []
        total_actions = self.get_total_actions()
        for al_id in range(self.n_agents):
            if al_id in mutate_al_id:
                random_action = random.randint(0,total_actions-1)
                actions_mutate.append(random_action)

                # type = self.get_action_type(int(actions_int[al_id]))
                # random_action = random.choice(self.action_type[type])
                # actions_mutate.append(random_action)

                # avail_actions,avail_actions_list = self.get_avail_agent_actions_2(al_id)
                # action = int(actions_int[al_id])
                
                # avail_actions_list_part = []
                # if action == 0:
                #     actions_mutate.append(0)
                # else:
                #     if (action >= 1 and action <= 5):
                #         for a in avail_actions_list:
                #             if (a >= 1 and a <= 5):
                #                 avail_actions_list_part.append(a)
                #     else:
                #         for a in avail_actions_list:
                #             if a > 5:
                #                 avail_actions_list_part.append(a)
                    
                #     random_action = random.choice(avail_actions_list)
                #     if len(avail_actions_list_part) > 0:
                #         random_action = random.choice(avail_actions_list_part)
                    
                #     actions_mutate.append(random_action)

            else:
                actions_mutate.append(actions_int[al_id])
        return actions_mutate


    def step(self, actions):    #_runner.py中reward, terminated, env_info = self.env.step(actions[0])
        """A single environment step. Returns reward, terminated, info."""
        
        
        '''my code'''
        if self.test_nepisode_index[self.battles_game] == 0:
            self.top_10_id_c = self.get_top_10(self.gain_dict_c)
            self.top_10_id_s = self.get_top_10(self.gain_dict_s)
            self.top_10_id_z = self.get_top_10(self.gain_dict_z)

            # print('top_10_c:',self.top_10_id_c)
            # print('top_10_s:',self.top_10_id_s)
            # print('top_10_z:',self.top_10_id_z)

        # for al_id, al_unit in self.agents.items():
        #     agent_obs = self.get_obs_agent_2(al_id)
        
        
        self.test_nepisode_index[self.battles_game] = 1

        unit_type_list = []
        for e_id, e_unit in self.enemies.items():
            unit_type_list.append(e_unit.unit_type)
        
        colossus_index,stalkers_index,zealots_index, en_id_index = self.find_position(unit_type_list)

        # if self._episode_steps == 0:
        #     id_visitied = []
        #     for e_id, e_unit in self.enemies.items():
        #         e_x = e_unit.pos.x
        #         e_y = e_unit.pos.y
        #         #print('e_x,e_y,e_health:',e_x,e_y,e_unit.health)

        #         pos_dis_list = []

        #         for key in self.id_pos_dict.keys():
        #             init_x = self.id_pos_dict[key][0]
        #             init_y = self.id_pos_dict[key][1]

        #             if key not in id_visitied:
        #                 pos_dis_list.append(math.hypot(e_x - init_x, e_y - init_y))
        #             else:
        #                 pos_dis_list.append(9999)
                
        #         min_pos_dis = min(pos_dis_list)
        #         id_define = pos_dis_list.index(min_pos_dis)
        #         if id_define not in id_visitied:
        #             id_visitied.append(id_define)

        #         self.id_id_dict[e_id] = id_define
        # actions_int_ordinal = []
        

        mutate_al_id = []   #扰动的智能体的id
        for al_id, al_unit in self.agents.items():
            agent_type = al_unit.unit_type
            #agent_state = self.get_agent_state_al_id(al_unit)
            agent_obs_state = self.get_obs_agent_2(al_id)
            #print(agent_state)
            # print('------')
            # print('al_id:',al_id)
            abs_state_id = int(self.state_abstracter.get_state_grid_ids(agent_obs_state)[0])
            
            # print('raw_state:',agent_obs_state[0])
            # print('grid_id:',abs_state_id)
            if agent_type == self.colossus_id:
                if abs_state_id in self.grid_dict_c.keys():
                    self.grid_dict_c[abs_state_id] += 1
                else:
                    self.grid_dict_c[abs_state_id] = 1
            
            elif agent_type == self.stalker_id:
                if abs_state_id in self.grid_dict_s.keys():
                    self.grid_dict_s[abs_state_id] += 1
                else:
                    self.grid_dict_s[abs_state_id] = 1
            
            elif agent_type == self.zealot_id: 
                if abs_state_id in self.grid_dict_z.keys():
                    self.grid_dict_z[abs_state_id] += 1
                else:
                    self.grid_dict_z[abs_state_id] = 1

            if agent_type == self.colossus_id and abs_state_id in self.top_10_id_c:
                self.mutate_state_id.append(abs_state_id)
                mutate_al_id.append(al_id)
                self.mutate_agent_state.append(['colossus',abs_state_id])
            
            if agent_type == self.stalker_id and abs_state_id in self.top_10_id_s:
                self.mutate_state_id.append(abs_state_id)
                mutate_al_id.append(al_id)
                self.mutate_agent_state.append(['stalker',abs_state_id])
            
            if agent_type == self.zealot_id and abs_state_id in self.top_10_id_z:
                self.mutate_state_id.append(abs_state_id)
                mutate_al_id.append(al_id)
                self.mutate_agent_state.append(['zealot',abs_state_id])


        # print('mutate:',mutate_al_id)

        '''end'''
        actions_int_rl = [int(a) for a in actions]
        actions_int = [int(a) for a in actions]
        
        #Ours
        if len(mutate_al_id) > 0:
            actions_mutate = self.mutate_action(mutate_al_id,actions_int)
            actions_int = actions_mutate

        #Random
        # isMutate = random.randint(0,1)
        # if isMutate == 1:
        #     num_mutate_al = random.randint(1,9)
        #     mutate_al_id = random.sample([0,1,2,3,4,5,6,7,8],num_mutate_al)
        #     actions_mutate = self.mutate_action(mutate_al_id,actions_int)
        #     actions_int = actions_mutate


        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))
        
        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action, actions_int_rl[a_id],self._episode_steps)   # Perturbed action and the policy action
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action)
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions) 
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}
        self.reward_list.append(reward)

        for e_id,e_unit in self.enemies.items():
            enemy_type = e_unit.unit_type
            enemy_state = self.get_enemy_state_en_id(e_unit)
            #print('enemy_state:',enemy_state)
            enemy_state_id = int(self.enemy_state_abstracter.get_state_grid_ids(enemy_state)[0])
            self.battle_state_id.append(enemy_state_id)

            if enemy_type == 4:
                self.battle_type_state.append(['colossus', enemy_state_id])
            elif enemy_type == 74:
                self.battle_type_state.append(['stalker', enemy_state_id])
            elif enemy_type == 73:
                self.battle_type_state.append(['zealot', enemy_state_id])
            


        '''my code'''
        # with open('results/1c3s5z/init_seed_info_2/ally_action_'+str(self.battles_game)+'.txt','a') as f:
        #     f.write(str(actions_int)+'\n')
        
        '''end code'''

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            
            #print('game_end_code:',game_end_code)
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1
        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate
        

        '''my code'''
        '''Individual Diveristy'''
        enemy_state,enemy_health_sum,unit_type_list = self.get_enemy_state()
        agent_health_sum = self.get_agent_state()
        
        enemy_state = enemy_state.tolist()


        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        enemy_state_ordinal = []
        for i in range(self.n_enemies): #将enemy_state按照1c3s5z排列
            if i == 0:
                enemy_state_ordinal.append(enemy_state[colossus_index[0]])
            elif i >= 1 and i <= 3:
                enemy_state_ordinal.append(enemy_state[stalkers_index[i-1]])
            elif i >= 4 and i <= 8:
                enemy_state_ordinal.append(enemy_state[zealots_index[i-4]])     

        # for i in range(self.n_enemies): #将enemy_state按照1c3s5z排列
        #     enemy_state_ordinal.append(enemy_state[self.id_id_dict[i]])     
        
        self.en_this_battle.append(enemy_state_ordinal)  #en_this_battle记录一局对战的状态序列
        
        '''Team Graph'''
        pos_list = []
        n_enemies_live = 0
        for i in range(self.n_enemies):
            target_id = en_id_index[i]
            target_unit = self.enemies[target_id]
            if target_unit.health > 0:
                n_enemies_live += 1
                x = target_unit.pos.x
                y = target_unit.pos.y
                pos_list.append([i,x,y,target_unit.health])
        # with open('results/1c3s5z/init_seed_info_2/pos_info_'+str(self.battles_game)+'.txt','a') as f:
        #     f.write(str(pos_list)+'\n')
        team_graph = TeamGraph(n_enemies_live,pos_list)
        team_graph.set_team_graph()
        self.battle_graph_list.append(team_graph)

        
        if game_end_code is not None or self._episode_steps >= self.episode_limit:
            
            # with open('results/1c3s5z/init_seed_info_2/reward.txt','a') as f_r:
            #     f_r.write(str(self.reward_list)+'\n')
            #     self.reward_list = []
            # with open('results/1c3s5z/init_seed_info_2/enemy_state_'+str(self.battles_game)+'.txt','a') as f:
            #         f.write(str(self.en_this_battle))
            
            #Energy Update
            delta_ef = 0.0
            delta_ev = 0.0

            O_s_new = self._episode_steps
            O_s_minus_list = []
            for seq in state_won_all:
                O_s = len(seq)
                minus = abs(O_s - O_s_new)
                O_s_minus_list.append(minus)
            
            O_s_minus = sum(O_s_minus_list) / len(O_s_minus_list)
            
            '''Individual Diversity'''
            dis_list = []
            for i in range(self.n_enemies):   # i equals enemy_id
                dis = isNovel(self.en_this_battle,i)
                dis_list.append(dis)
            
            dis_violation = sum(dis_list)/len(dis_list)

            '''Team Diversity'''
            try:
                max_r = max(self.reward_list)
                max_r_step = self.reward_list.index(max_r)

                graph_select = self.battle_graph_list[max_r_step]
                init_labels,weights = graph_select.re_label_info()
                graph_select_emb = compute_wl_propagation_aggregation(graph_select.n_nodes,init_labels,self.iterations,weights,3)[self.iterations - 1]
                
                wl_dis_list = []
                for emb in self.vector_list:
                    distance = compute_wasserstein_distance(emb,graph_select_emb,self.iterations,sinkhorn=False,discrete=False)
                    wl_dis_list.append(distance)
                wl_dis = sum(wl_dis_list)/len(wl_dis_list)

                wl_dis_violation = (wl_dis - self.init_wl_dis) / self.init_wl_dis
            except:
                wl_dis_violation = 0

            delta_ev = O_s_minus / (1 - (dis_violation + wl_dis_violation) /2 + 10**(-5))

            #print('delta_ev:',delta_ev)
            
            for agent_state in self.mutate_agent_state:
                agent_type = agent_state[0]
                state_id = agent_state[1]
                if agent_state not in self.battle_agent_state_id_visited:
                    if game_end_code == 1:
                        if agent_type == 'colossus' :
                            self.gain_dict_c[state_id] += 0.5 * np.tanh(delta_ev) - 0.05 * 1    #Gain of diversity
                            self.num_abs_table_select_c[state_id][0] += 1
                            delta_ef = self.num_abs_table_select_c[state_id][0] / (self.num_abs_table_select_c[state_id][0]+self.num_abs_table_select_c[state_id][1])
                            self.gain_dict_c[state_id] += 0.5 * delta_ef    #Gain of winning rate
                        elif agent_type == 'stalker':
                            self.gain_dict_s[state_id] += 0.5 * np.tanh(delta_ev) - 0.05 * 1
                            self.num_abs_table_select_s[state_id][0] += 1
                            delta_ef = self.num_abs_table_select_s[state_id][0] / (self.num_abs_table_select_s[state_id][0]+self.num_abs_table_select_s[state_id][1])
                            self.gain_dict_s[state_id] += 0.5 * delta_ef
                        elif agent_type == 'zealot':
                            self.gain_dict_z[state_id] += 0.5 * np.tanh(delta_ev) - 0.05 * 1
                            self.num_abs_table_select_z[state_id][0] += 1
                            delta_ef = self.num_abs_table_select_z[state_id][0] / (self.num_abs_table_select_z[state_id][0]+self.num_abs_table_select_z[state_id][1])
                            self.gain_dict_z[state_id] += 0.5 * delta_ef
                    else: #平局 or lose
                        if agent_type == 'colossus' :
                            self.gain_dict_c[state_id] += 0.5 * np.tanh(delta_ev) - 0.05 * 1
                            self.num_abs_table_select_c[state_id][1] += 1
                            delta_ef = - 0.1 * self.num_abs_table_select_c[state_id][1] / (self.num_abs_table_select_c[state_id][0]+self.num_abs_table_select_c[state_id][1])
                            self.gain_dict_c[state_id] += 0.5 * delta_ef
                        elif agent_type == 'stalker':
                            self.gain_dict_s[state_id] += 0.5 * np.tanh(delta_ev) - 0.05 * 1
                            self.num_abs_table_select_s[state_id][1] += 1
                            delta_ef = - 0.1 * self.num_abs_table_select_s[state_id][1] / (self.num_abs_table_select_s[state_id][0]+self.num_abs_table_select_s[state_id][1])
                            self.gain_dict_s[state_id] += 0.5 * delta_ef
                        elif agent_type == 'zealot':
                            self.gain_dict_z[state_id] += 0.5 * np.tanh(delta_ev) - 0.05 * 1
                            self.num_abs_table_select_z[state_id][1] += 1
                            delta_ef = - 0.1 * self.num_abs_table_select_z[state_id][1] / (self.num_abs_table_select_z[state_id][0]+self.num_abs_table_select_z[state_id][1])
                            self.gain_dict_z[state_id] += 0.5 * delta_ef
                    
                    self.battle_agent_state_id_visited.append(agent_state)
                    # print('delta_ev:',0.5 * np.tanh(delta_ev) - 0.05 * 1)
                    # print('delta_ef:',delta_ef)
            if game_end_code == 1:
                self.vector_list.append(graph_select_emb)
                with open(self.result_path+'/enemy_state_'+str(self.battles_game)+'.txt','a') as f:
                    f.write(str(self.en_this_battle))
                for id in self.battle_state_id:
                    self.enemy_state_dict[id] += 1
                
                for type_state in self.battle_type_state:
                    type = type_state[0]
                    state_id = type_state[1]

                    if type == 'colossus':
                        if state_id in self.enemy_state_dict_c.keys():
                            self.enemy_state_dict_c[state_id] += 1
                        else:
                            self.enemy_state_dict_c[state_id] = 1
                    elif type == 'stalker':
                        if state_id in self.enemy_state_dict_s.keys():
                            self.enemy_state_dict_s[state_id] += 1
                        else:
                            self.enemy_state_dict_s[state_id] = 1
                    elif type == 'zealot':
                        if state_id in self.enemy_state_dict_z.keys():
                            self.enemy_state_dict_z[state_id] += 1
                        else:
                            self.enemy_state_dict_z[state_id] = 1
            
            self.en_this_battle = []
            self.reward_list = []
            self.mutate_state_id = []
            self.mutate_agent_state = []
            self.battle_state_id = []
            self.battle_agent_state_id_visited = []

            # count = 0
            # for key in self.enemy_state_dict.keys():
            #     if self.enemy_state_dict[key] > 0:
            #         count += 1
            # print('self.battles_game:',self.battles_game)
            # print('game_end_code:',game_end_code)
            # print('enemy_state_dict:',count)


            if (self.battles_game+1) % 200 == 0:
                count = 0
                for key in self.enemy_state_dict_c.keys():
                    if self.enemy_state_dict_c[key] > 0:
                        count += 1
                print('enemy_state_dict_c:',count)
                print(self.enemy_state_dict_c)

                count = 0
                for key in self.enemy_state_dict_s.keys():
                    if self.enemy_state_dict_s[key] > 0:
                        count += 1
                print('enemy_state_dict_s:',count)
                print(self.enemy_state_dict_s)

                count = 0
                for key in self.enemy_state_dict_z.keys(): 
                    if self.enemy_state_dict_z[key] > 0:
                        count += 1
                print('enemy_state_dict_z:',count)
                print(self.enemy_state_dict_z)
          
        '''end code'''
        

        return reward, terminated, info
    

    def get_agent_action(self, a_id, action, action_rl,episode_steps):
        """Construct the action for agent a_id."""
        avail_actions,avail_actions_list = self.get_avail_agent_actions_2(a_id)
        if avail_actions[action] == 0:
            action = random.choice(avail_actions_list)
            
        assert avail_actions[action] == 1, \
                "Agent {} cannot perform action {}".format(a_id, action)

        #action = action_rl #DRL
        
        
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (target is None or self.agents[target].health == 0 or
                self.agents[target].health == self.agents[target].health_max):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (al_unit.health != 0 and
                        al_unit.health != al_unit.health_max):
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             al_unit.pos.x, al_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['heal']
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (unit.unit_type == self.marauder_id and
                        e_unit.unit_type == self.medivac_id):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             e_unit.pos.x, e_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['attack']
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack
        print('action_num:',action_num)
        # Check if the action is available
        if (self.heuristic_rest and
            self.get_avail_agent_actions(a_id)[action_num] == 0):

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y): # east or west
                if delta_x > 0: # east
                    target_pos=sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y)
                    action_num = 4
                else: # west
                    target_pos=sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y)
                    action_num = 5
            else: # north or south
                if delta_y > 0: # north
                    target_pos=sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount)
                    action_num = 2
                else: # south
                    target_pos=sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount)
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id = actions['move'],
                target_world_space_pos = target_pos,
                unit_tags = [tag],
                queue_command = False)
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id = action_id,
                target_unit_tag = target_tag,
                unit_tags = [tag],
                queue_command = False)

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative(累计) hit/shield point damage dealt(造成命中/盾点伤害) to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        # flag = False #判断是否攻击力度过大
        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                    
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self):
        """My output."""
        count = 0
        for key in self.enemy_state_dict.keys():
            if self.enemy_state_dict[key] > 0:
                count += 1
        print('enemy_state_dict:',count)
        print(self.enemy_state_dict)

        count = 0
        for key in self.grid_dict_c.keys():
            if self.grid_dict_c[key] > 0:
                count += 1
        print('grid_dict_c:',count)
        print(self.grid_dict_c)

        count = 0
        for key in self.grid_dict_s.keys():
            if self.grid_dict_s[key] > 0:
                count += 1
        print('grid_dict_s:',count)
        print(self.grid_dict_s)

        count = 0
        for key in self.grid_dict_z.keys():
            if self.grid_dict_z[key] > 0:
                count += 1
        print('grid_dict_z:',count)
        print(self.grid_dict_z)

        count = 0
        for key in self.enemy_state_dict_c.keys():
            if self.enemy_state_dict_c[key] > 0:
                count += 1
        print('enemy_state_dict_c:',count)
        print(self.enemy_state_dict_c)

        count = 0
        for key in self.enemy_state_dict_s.keys():
            if self.enemy_state_dict_s[key] > 0:
                count += 1
        print('enemy_state_dict_s:',count)
        print(self.enemy_state_dict_s)

        count = 0
        for key in self.enemy_state_dict_z.keys():
            if self.enemy_state_dict_z[key] > 0:
                count += 1
        print('enemy_state_dict_z:',count)
        print(self.enemy_state_dict_z)


        # state_file = open('results/6-25/state_file_C14_500_A_2.txt','a')
        # for battle in state_sequence_all:
        #     state_file.write(str(battle)+'\n')

        # battle_state_temp = []
        # for state in self.en_battle_state:
        #     if state not in battle_state_temp:
        #         state_file.write(str(state)+'\n')
        #         battle_state_temp.append(state)
        # logging.info("Different state number:")
        # logging.info(len(battle_state_temp))
        print(self.get_stats())
        
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        logging.info("repaly dir:%s" % replay_dir)
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix)
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        #return (0 <= x < 0.5*self.map_x and 0 <= y < 0.5*self.map_y)
        return (0 <= x < self.map_x and 0 <= y < self.map_y)

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)
        # print('type:',unit.unit_type)
        # print('self.colossus_id:',self.colossus_id)
        # print('self.stalker_id:',self.stalker_id)
        # print('self.zealot_id:',self.zealot_id)


        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                    dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (
                        e_x - x
                    ) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (
                        e_y - y
                    ) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                    dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (
                            al_unit.health / al_unit.health_max
                        )  # health
                        # ally_feats[i, ind] = (
                        #     0
                        # )  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))
        return agent_obs
    
    def get_obs_agent_2(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_enemy_feats_size()``, 
           ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()


        enemy_feats_dim = (9, 4)
        ally_feats_dim = (8, 4)
        own_feats_dim = 1

        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)
            

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (
                    dist < sight_range and e_unit.health > 0
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = dist / sight_range  # distance
                    enemy_feats[e_id, 1] = (
                        abs(e_x - x)
                    ) / sight_range  # relative X
                    enemy_feats[e_id, 2] = (
                        abs(e_y - y)
                    ) / sight_range  # relative Y

                    ind = 3
                    if self.obs_all_health:
                        max_shield = self.unit_max_shield(e_unit)
                        enemy_feats[e_id, ind] = (
                            (e_unit.health + e_unit.shield) / (e_unit.health_max + max_shield)
                        )  # health
                
                        ind += 1

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if (
                    dist < sight_range and al_unit.health > 0
                ):  # visible and alive
                    ally_feats[i, 0] = dist / sight_range  # distance
                    ally_feats[i, 1] = abs(al_x - x) / sight_range  # relative X
                    ally_feats[i, 2] = abs(al_y - y) / sight_range  # relative Y

                    ind = 3
                    if self.obs_all_health:
                        max_shield = self.unit_max_shield(al_unit)
                        ally_feats[i, ind] = (
                            (al_unit.health + al_unit.shield) / (al_unit.health_max + max_shield)
                        ) 
                        ind += 1

            # Own features
            ind = 0
            if self.obs_own_health:
                max_shield = self.unit_max_shield(unit)
                own_feats[ind] = (unit.health + unit.shield) / (unit.health_max + max_shield)
                ind += 1


        agent_obs = np.concatenate(
            (
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))
        # return agent_obs
        return np.array([agent_obs])

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if (
                    self.map_type == "MMM"
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = (
                        al_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = (
                        e_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1


        
        
        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps / self.episode_limit)

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state
    
    def get_state_all(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if (
                    self.map_type == "MMM"
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                    x - self.min_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - self.min_y
                ) / self.max_distance_y  # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = (
                        al_unit.shield / max_shield
                    )  # shield

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = (
                        e_unit.shield / max_shield
                    )  # shield
                    ind += 1
        
        
        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps / self.episode_limit)

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))
        print(len(state))
        return state
    
    
    def get_enemy_state_en_id(self,e_unit):
        '''Return the state of enemy en_id'''  
        enemy_state = [0.0] * 3

        if e_unit.health > 0:
            x = e_unit.pos.x
            y = e_unit.pos.y
            max_shield = self.unit_max_shield(e_unit)

            enemy_state[0] = (
                (e_unit.health + e_unit.shield) / (e_unit.health_max + max_shield)
            )  # health
            enemy_state[1] =(x - self.min_x
            ) / self.max_distance_x  # relative X
            enemy_state[2] = (y - self.min_y
            ) / self.max_distance_y  # relative Y
            
            # enemy_state[3] = (
            #     e_unit.shield / max_shield
            # )  # shield
        return np.array([enemy_state])


    def get_agent_state_al_id(self,al_unit):
        '''Return the state of agent al_id'''
        ally_state = [0.0] * 4
        #al_unit = self.get_unit_by_id(agent_id)
        

        if al_unit.health > 0:
            x = al_unit.pos.x
            y = al_unit.pos.y

            max_cd = self.unit_max_cooldown(al_unit)
            ally_state[0] = al_unit.health / al_unit.health_max

            # if (
            #     self.map_type == "MMM"
            #     and al_unit.unit_type == self.medivac_id
            # ):
            #     ally_state[1] = al_unit.energy / max_cd  # energy
            # else:
            #     ally_state[1] = (
            #         al_unit.weapon_cooldown / max_cd
            #     )  # cooldown
            ally_state[1] = (
                x - self.min_x
            ) / self.max_distance_x  # relative X
            ally_state[2] = (
                y - self.min_y
            ) / self.max_distance_y  # relative Y
            
            
            max_shield = self.unit_max_shield(al_unit)
            ally_state[3] = (
                al_unit.shield / max_shield
            )  # shield

        return np.array([ally_state])

    def get_agent_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
 
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        health_sum = 0

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)
                health_sum += al_unit.health

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if (
                    self.map_type == "MMM"
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = (
                        al_unit.shield / max_shield
                    )  # shield
                    ind += 1
    
        return ally_state
    
    def get_enemy_state(self):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        health_sum = 0

        unit_type_list = []
        for e_id, e_unit in self.enemies.items():
            unit_type_list.append(e_unit.unit_type)

        for e_id, e_unit in self.enemies.items():
            # print('e_id:',e_id)
            # print('e_unit.unit_type:',e_unit.unit_type)
            
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                health_sum += e_unit.health

                enemy_state[e_id, 0] = (
                    e_unit.health
                    # e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = x
                enemy_state[e_id, 2] = y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = (
                        e_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1
            # else:
            #     enemy_state[e_id, 0] = 0.0
            #     enemy_state[e_id, 1] = 0.0
            #     enemy_state[e_id, 2] = 0.0

            #     ind = 3
                
            #     enemy_state[e_id, ind] = 0.0
            #     type_id = self.get_unit_type_id(e_unit, False)
            #     enemy_state[e_id, ind + type_id] = 0


        return enemy_state,health_sum,unit_type_list

    def get_obs_enemy_feats_size(self):
        """ Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return self.n_agents - 1, nf_al

    def get_obs_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features.
        """
        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    # def get_visibility_matrix(self):
    #     """Returns a boolean numpy array of dimensions 
    #     (n_agents, n_agents + n_enemies) indicating which units
    #     are visible to each agent.
    #     """
    #     arr = np.zeros(
    #         (self.n_agents, self.n_agents + self.n_enemies), 
    #         dtype=np.bool,
    #     )

    #     for agent_id in range(self.n_agents):
    #         current_agent = self.get_unit_by_id(agent_id)
    #         if current_agent.health > 0:  # it agent not dead
    #             x = current_agent.pos.x
    #             y = current_agent.pos.y
    #             sight_range = self.unit_sight_range(agent_id)

    #             # Enemies
    #             for e_id, e_unit in self.enemies.items():
    #                 e_x = e_unit.pos.x
    #                 e_y = e_unit.pos.y
    #                 dist = self.distance(x, y, e_x, e_y)

    #                 if (dist < sight_range and e_unit.health > 0):
    #                     # visible and alive
    #                     arr[agent_id, self.n_agents + e_id] = 1

    #             # The matrix for allies is filled symmetrically
    #             al_ids = [
    #                 al_id for al_id in range(self.n_agents) 
    #                 if al_id > agent_id
    #             ]
    #             for i, al_id in enumerate(al_ids):
    #                 al_unit = self.get_unit_by_id(al_id)
    #                 al_x = al_unit.pos.x
    #                 al_y = al_unit.pos.y
    #                 dist = self.distance(x, y, al_x, al_y)

    #                 if (dist < sight_range and al_unit.health > 0):  
    #                     # visible and alive
    #                     #arr[agent_id, al_id] = arr[al_id, agent_id] = 1
    #                     arr[agent_id, al_id] = arr[al_id, agent_id] = -1

    #     return arr

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            type_id = unit.unit_type - self._min_unit_type
        else:  # use default SC2 unit types
            if self.map_type == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            elif self.map_type == "colossi_stalkers_zealots":
                # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                if unit.unit_type == 4:
                    type_id = 0
                elif unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_type == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2
            # for communication
            elif self.map_type == "overload_roach":
                # roach
                type_id = 0
            elif self.map_type == "overload_bane":
                # baneling
                type_id = 0
            elif self.map_type == "bZ_hM":
                if unit.unit_type == 107:
                    # hydralisk
                    type_id = 0
                else:
                    # medivacs
                    type_id = 1

        return type_id
    def get_avail_agent_actions_2(self, agent_id):
        avail_actions_list = []
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        #print('agent_id,unit_type',agent_id,unit.unit_type)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1
            avail_actions_list.append(1)
            
            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
                avail_actions_list.append(2)
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
                avail_actions_list.append(3)
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
                avail_actions_list.append(4)
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1
                avail_actions_list.append(5)

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1
                        avail_actions_list.append(t_id + self.n_actions_no_attack)
            return avail_actions,avail_actions_list

        else:
            # only no-op allowed
            avail_actions_list.append(0)
            return [1] + [0] * (self.n_actions - 1),avail_actions_list

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        #print('agent_id,unit_type',agent_id,unit.unit_type)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1
            
            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1
            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)
            

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def init_units(self):
        """Initialise the units."""
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values()
                )
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = (len(self.agents) == self.n_agents)
            all_enemies_created = (len(self.enemies) == self.n_enemies)

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (n_ally_alive == 0 and n_enemy_alive > 0
                or self.only_medivac_left(ally=True)):
            return -1  # lost
        if (n_ally_alive > 0 and n_enemy_alive == 0
                or self.only_medivac_left(ally=False)):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = min_unit_type
            self.stalker_id = min_unit_type + 1
            self.zealot_id = min_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

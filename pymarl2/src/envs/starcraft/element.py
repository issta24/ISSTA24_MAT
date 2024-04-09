import copy

import numpy as np

class CorpusElement(object):
    """Class representing a single element of a corpus."""

    def __init__(self, id, energy,parent_id):
        self.parent_id = None
        self.id = id
        self.energy = energy
        self.actions = None
        self.parent_id = parent_id
        #self.episode_limit = episode_limit
    
    def get_actions(self):
        return self.actions

    def set_actions(self,actions):
        self.actions = actions
    
    def get_id(self):
        return self.id
    
    def comput_energy(self,episode_limit, game_end_code):
        novel = 0  
        isfail = 0
        valid = 0   #失败的程度

        if game_end_code == 1:
            isfail = 0.1
        else:
            isfail = -0.1

        valid = (episode_limit - len(self.actions))/episode_limit

        self.energy = novel + isfail + valid

        return self.energy

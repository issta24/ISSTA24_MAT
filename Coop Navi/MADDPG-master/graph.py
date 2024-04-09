import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math,json

class TeamGraph(object):
    """Class representing a global state graph of the target team."""

    def __init__(self,n_nodes,pos_list):
        self.n_nodes = n_nodes
        #self.team_graph = [[-1] * n_nodes] * n_nodes
        self.team_graph = []
        self.position = pos_list    #position of nodes  [[node_id,x,y,v1,v2]]

    def set_team_graph(self):
        for i in range(self.n_nodes):
            self.team_graph.append([-1]*self.n_nodes)
        for i in range(self.n_nodes - 1):
            for j in range(i+1, self.n_nodes):
                x1 = self.position[i][1]
                y1 = self.position[i][2]
                x2 = self.position[j][1]
                y2 = self.position[j][2]
                self.team_graph[i][j] = math.hypot(x2 - x1, y2 - y1)
                self.team_graph[j][i] = math.hypot(x2 - x1, y2 - y1)
        # for dis in dis_list:
        #     node_1 = dis[0]
        #     node_2 = dis[1]
        #     d = dis[2]
        #     self.team_graph[node_1][node_2] = d
    
    def get_team_graph(self):
        return self.team_graph

    def add_edge(self,dis):
        node_1 = dis[0]
        node_2 = dis[1]
        d = dis[2]
        self.team_graph[node_1][node_2] = d

    def get_connection_info(self):
        num = 0
        dis_sum = 0
        for i in range(0,self.n_nodes):
            for j in range(i+1,self.n_nodes):
                if self.team_graph[i][j] > 0:
                    num += 1
                    dis_sum += self.team_graph[i][j]
        return num,dis_sum/num

        
    def get_degree(self):
        degree_list = [0] * self.n_nodes
        for i in range(0,self.n_nodes):
            num = 0
            for j in range(i+1,self.n_nodes):
                if self.team_graph[i][j] > 0:
                    num += 1
            degree_list[i] = num
        return degree_list
    

    def get_hull(self):
        polygon = []
        for pos in self.position:
            polygon.append((pos[1],pos[2]))

        polygon = np.array(polygon)

        hull = ConvexHull(polygon)

        area = round(hull.area,2)

        return area
    
    def re_label_info(self):
        features = []
        edge_attributes_dict = {}
        for i in range(self.n_nodes):
            features.append([self.position[i][1],self.position[i][2],self.position[i][3],self.position[i][4]])
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    t = (i,j)
                    w = self.team_graph[i][j]
                    edge_attributes_dict[str(t)] = w
        
        return features,edge_attributes_dict


    def get_json_info(self,path,battles_game,graph_id):
        edges = []
        edges_weight = []
        features = {}
        for i in range(self.n_nodes):
            #features[i] = self.n_nodes - 1
            #features[i] = [self.position[i][1],self.position[i][2],self.position[i][3]]
            features[i] = str(self.position[i][3])
        

        for i in range(self.n_nodes - 1):
            for j in range(self.n_nodes - 1):
                if i != j:
                    edges.append([i,j])
                    edges_weight.append([i,j,self.team_graph[i][j]])

        info = {}
        edge_attributes_dict = {}
        for edge in edges_weight:
            t = (edge[0],edge[1])
            w = edge[2]
            edge_attributes_dict[str(t)] = w
        
        info['edges'] = edges
        info['features'] = features
        info['weight'] = edge_attributes_dict

        with open(path +'/'+str(graph_id)+'_graph_'+str(battles_game)+'.json','w') as f_g:
            json.dump(info, f_g)

# t = TeamGraph(5,[[1,1],[2,2]])
# t.set_team_graph([[1,2,3],[2,4,5]])
# print(t.get_connection_info())


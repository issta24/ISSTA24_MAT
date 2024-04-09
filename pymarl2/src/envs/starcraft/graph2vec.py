"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from envs.starcraft.param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])
    weights = data['weight']

    edge_attributes_dict = {}
    for key in weights.keys():
        edge_attributes_dict[eval(key)] = weights[key]


    nx.set_edge_attributes(G=graph,values = edge_attributes_dict,name = 'weight') 

    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    #out = []
    vectors = []
    for f in files:
        identifier = path2name(f)
        vectors.append(list(model.docvecs["g_"+identifier]))
        #out.append([identifier] + list(model.docvecs["g_"+identifier]))
    # column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    # out = pd.DataFrame(out, columns=column_names)
    # out = out.sort_values(["type"])
    # #out.to_csv(output_path, index=None)
    # return out
    return vectors

def main_graph(input_graph_path,battle_id):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    #args = parameter_parser()
    graphs = glob.glob(os.path.join(input_graph_path, "*graph_"+str(battle_id)+".json"))
    #graphs = input_graph
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=4)(delayed(feature_extractor)(g, 2) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=128,
                    window=0,
                    min_count=5,
                    dm=0,
                    sample=0.0001,
                    workers=4,
                    epochs=10,
                    alpha=0.025)

    return(save_embedding(model, graphs, 128))

# if __name__ == "__main__":
#     args = parameter_parser()
#     main(args)

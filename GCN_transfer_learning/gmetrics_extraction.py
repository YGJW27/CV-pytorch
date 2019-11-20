import os
import glob
import pandas as pd
import networkx as nx
import numpy as np


DATA_PATH = "D:/code/DTI_data/network_FN/"
OUTPUT_PATH = "D:/code/DTI_data/output/local_metrics/"

sub_dirs = [x[0] for x in os.walk(DATA_PATH)]
sub_dirs.pop(0)


for sub_dir in sub_dirs:
    file_list = []
    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(DATA_PATH, dir_name, '*')
    file_list.extend(glob.glob(file_glob))
    output_dir = os.path.join(OUTPUT_PATH, dir_name)

    for f in file_list:
        file_name = os.path.basename(f)
        output_dir = os.path.join(OUTPUT_PATH, dir_name)
        output = os.path.join(OUTPUT_PATH, dir_name, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataframe = pd.read_csv(f, sep="\s+", header=None)
        w = dataframe.to_numpy()

        G = nx.from_numpy_array(w)

        degree_dict = nx.degree(G, weight='weight')
        neighbor_degree_dict = nx.average_neighbor_degree(G, weight='weight')
        clustering_coeff_dict = nx.clustering(G, weight=None)
        # betweenness_centrality_dict = nx.betweenness_centrality(G, normalized=True, weight='weight')

        # local_eff_list = np.zeros(len(degree_dict))
        # for node in G.nodes():
        #     local_eff_list[node] = (nx.local_efficiency(G.subgraph(G.neighbors(node))))

        degree_list = np.zeros(len(degree_dict))
        neighbor_degree_list = np.zeros(len(neighbor_degree_dict))
        clustering_coeff_list = np.zeros(len(clustering_coeff_dict))
        # betweenness_centrality_list = np.zeros(len(betweenness_centrality_dict))

        for k, v in degree_dict:
            degree_list[k] = v

        for k, v in neighbor_degree_dict.items():
            neighbor_degree_list[k] = v

        for k, v in clustering_coeff_dict.items():
            clustering_coeff_list[k] = v

        degree_list = np.expand_dims(degree_list, axis=1)
        neighbor_degree_list = np.expand_dims(neighbor_degree_list, axis=1)
        clustering_coeff_list = np.expand_dims(clustering_coeff_list, axis=1)

        feature_array = np.concatenate(
            (degree_list, neighbor_degree_list, clustering_coeff_list),
            axis=1)

        df = pd.DataFrame(feature_array)
        df.to_csv(output, sep="\t", header=False, index=False)

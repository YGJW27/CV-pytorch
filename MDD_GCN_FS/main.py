import networkx as nx
import pandas as pd


def main():
    GroupLevelGraph = "D:/code/DTI_data/network_distance/grouplevel.edge"
    dataframe = pd.read_csv(GroupLevelGraph, sep="\t", header=None)
    w = dataframe.to_numpy()
    G = nx.from_numpy_array(w)

    print(nx.is_connected(G))
    max_cliques = list(nx.algorithms.clique.find_cliques(G))


if __name__ == "__main__":
    main()
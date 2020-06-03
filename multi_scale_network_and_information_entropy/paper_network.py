import networkx as nx
import numpy as np


def nodes_selection():
    G = nx.Graph()
    G.add_nodes_from(range(1, 91))
    G.add_edges_from([
        (3, 7),
        (3, 29),
        (5, 9),
        (5, 37),
        (5, 73),
        (9, 15),
        (9, 37),
        (9, 73),
        (9, 75),
        (13, 15),
        (15, 29),
        (15, 77),
        (9, 32),
        (32, 34),
        (19, 73),
        (20, 32),
        (37, 77),
        (72, 78),
        (75, 77),
        (30, 60),
        (36, 60),
        (36, 68),
        (60, 74),
        (60, 66),
        (67, 70),
        (55, 77),
        (37, 85),
        (37, 89),
        (38, 90),
        (44, 90),
        (50, 52),
        (50, 90),
        (52, 90)])
        # [3 5 7 9 13 15 19 20 29 30 32 34 36 37 38 44 50 52 55 60 66 67 68 70 72 73 74 75 77 78 85 89 90]
    adj_G = nx.to_numpy_array(G)

    assert np.all(adj_G == adj_G.T)
    connected_nodes = np.where(np.sum(adj_G, axis=0) != 0)[0]
    return connected_nodes

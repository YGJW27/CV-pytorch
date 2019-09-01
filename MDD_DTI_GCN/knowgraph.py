import pandas as pd
import numpy as np

OUTPATH = "D:/code/DTI_data/network_distance/know_graph.edge"
g = np.ones((90, 90), dtype=np.float)
df = pd.DataFrame(g)
df.to_csv(OUTPATH, sep=" ", header=False, index=False)

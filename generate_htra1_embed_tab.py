import torch
import pandas as pd
import numpy as np
import pickle

full_tab_for_embed = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/htra1_analysis/htra1_mutants.tsv", sep='\t')

with open("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/htra1_embed_pkl_chunk0", "rb") as f:
    tenor_list_esm2 = pickle.load(f)


np_list = []
for i, ten in enumerate(tenor_list_esm2):
  ten=ten.detach().numpy()
  np_list.append(ten)


full_tab_for_embed["esm_embeddings"] = pd.Series(np_list)

# add representatives
# cluster_tab = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/session_cluster.tsv", sep='\t', header=None)
# cluster_tab.rename(columns={0: "representative", 1: "code"}, inplace=True)
# cluster_tab.drop_duplicates(subset=(["code"]), inplace=True)

# full_tab_with_clusters = pd.merge(full_tab_for_embed, cluster_tab, on="code", how="outer")

with open("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/esm_tab_htra1.pkl", "wb") as f:
    pickle.dump(full_tab_for_embed, f)


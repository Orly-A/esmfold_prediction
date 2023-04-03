from transformers import EsmTokenizer, EsmForSequenceClassification

import esm
import os.path

import torch
import re
import os
import requests
from tqdm.auto import tqdm
import pickle 
import numpy as np 
import sys

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", problem_type="multi_label_classification")


with open("/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/parsed_tab_for_embed.pkl", "rb") as f:
    full_tab_for_embed = pickle.load(f)
full_tab_for_embed.reset_index(inplace=True, drop=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
model = model.to(device)
model = model.eval()

fasta_list = list(zip(full_tab_for_embed["fasta"].index,full_tab_for_embed["fasta"]))
def divide_chunks(fasta_list, n):
    # looping till length l
    for i in range(0, len(fasta_list), n):
        yield fasta_list[i:i + n]
 
 
list_of_chunks = list(divide_chunks(fasta_list, 10))

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

for chunk_num, chunk in enumerate(list_of_chunks):
#    if chunk_num < 1160:
#       continue
    fname = "/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/esm_embeds_folder/embed_pkl_chunk" + str(chunk_num)
    if os.path.isfile(fname):
        continue

    print("working on " + str(chunk_num))
    data = chunk
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

  # Generate per-sequence representations via averaging
  # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
      #  print(i)
      # print(tokens_len)
      # print(batch_lens)

    del results
    del token_representations
    print("saving " + str(chunk_num))
    with open(fname, 'wb') as f:
        pickle.dump(sequence_representations, f)

print("finished all chunks")

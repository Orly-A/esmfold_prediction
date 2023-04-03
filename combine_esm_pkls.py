import pickle
import os
import pandas as pd
from natsort import humansorted

esm_embeds_dir = "/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/esm_embeds_folder"
os.chdir(esm_embeds_dir)
pickle_list = humansorted(os.listdir('.'))
tensor_list = []
for file in pickle_list:
    with open(file, "rb") as f:
        current = pickle.load(f)
#        print(current)
        tensor_list.extend(current)
#        print(tensor_list)
with open ("/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/tenor_list_esm2_final.pkl", "wb") as f:
    pickle.dump(tensor_list, f)
print("finished all the tensors")
print(len(tensor_list))





#embed_pkl_chunk154
#[tensor([ 0.0382, -0.0192, -0.0028,  ..., -0.0897, -0.0386,  0.0871]), tensor([ 0.0416, -0.0191, -0.0019,  ..., -0.0897, -0.0386,  0.0860]), tensor([ 0.0461, -0.0150,  0.0017,  ..., -0.0953, -0.0345,  0.0793]), tensor([ 0.0418, -0.0147, -0.0085,  ..., -0.0913, -0.0316,  0.0836]), tensor([ 0.0372, -0.0158, -0.0141,  ..., -0.0964, -0.0316,  0.0930]), tensor([ 0.0399, -0.0182, -0.0075,  ..., -0.0926, -0.0316,  0.0866]), tensor([ 0.0409, -0.0176, -0.0039,  ..., -0.0944, -0.0327,  0.0866]), tensor([ 0.0387, -0.0251, -0.0062,  ..., -0.0849, -0.0397,  0.0819]), tensor([ 0.0437, -0.0194, -0.0065,  ..., -0.0922, -0.0357,  0.0959]), tensor([ 0.0333, -0.0338, -0.0271,  ..., -0.0987, -0.0227,  0.0984])]
#embed_pkl_chunk155
#[tensor([ 0.0392, -0.0396, -0.0294,  ..., -0.0994, -0.0237,  0.0960]), tensor([ 0.0378, -0.0364, -0.0276,  ..., -0.0975, -0.0232,  0.0927]), tensor([ 0.0367, -0.0340, -0.0338,  ..., -0.0948, -0.0267,  0.0840]), tensor([ 0.0383, -0.0215, -0.0046,  ..., -0.0911, -0.0362,  0.0926]), tensor([ 0.0398, -0.0167, -0.0125,  ..., -0.0919, -0.0373,  0.0873]), tensor([ 0.0401, -0.0138, -0.0044,  ..., -0.0815, -0.0436,  0.0718]), tensor([ 0.0393, -0.0157, -0.0036,  ..., -0.0801, -0.0418,  0.0751]), tensor([ 0.0246, -0.0248, -0.0322,  ..., -0.0793, -0.0458,  0.0719]), tensor([ 0.0369, -0.0240, -0.0307,  ..., -0.0745, -0.0417,  0.0770]), tensor([ 0.0377, -0.0278, -0.0236,  ..., -0.0789, -0.0389,  0.0757])]


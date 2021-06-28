import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
try:
    import cupy as cp
except:
    pass


def calc_auc2(embs: np.array, persons: list):
    N = len(persons)
    N_pairs = int((N*N + N) / 2 - N)

    embs = embs.astype(np.float16)
    if torch.cuda.is_available():
        embs = cp.asarray(embs)
        sims = cp.asnumpy(cp.dot(embs, embs.T))
    else:
        sims = np.matmul(embs, embs.T) # heavy computation time
    matches_arr = [0]*N_pairs
    sims_arr = []

    cnt=0
    for i in range(1, N):
        for j in range(i+1, N):
            if persons[i] == persons[j]: 
                matches_arr[cnt] = 1
            else: 
                break

    sims_arr = []
    for i in range(0, N-1):
        sims_arr += list(sims[i][i+1:])

    if len(matches_arr) <= 100: # debug
        matches_arr.append(1)
        sims_arr.append(0.73)

    return roc_auc_score(matches_arr, sims_arr)

def calc_auc(embs: np.array, persons: list):
    persons = np.array(persons) + 1
    sims = np.matmul(embs, embs.T)
    sim_vec = np.triu(sims, k=1).reshape(-1,)
    ind = np.where(sim_vec!=0)
    sim_vec = sim_vec[ind]

    persons_mat = np.matmul(persons.reshape(-1, 1), persons.reshape(1, -1)) / persons**2
    persons_mat[np.where(persons_mat != 1)] = 0
    persons_vec = persons_mat.reshape(-1,)[ind]
    if len(persons_vec) <= 100: # debug
        persons_vec = np.append(persons_vec, 0)
        sim_vec = np.append(sim_vec, 0.5)

    return roc_auc_score(persons_vec, sim_vec)




    

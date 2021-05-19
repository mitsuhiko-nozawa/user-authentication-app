import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
try:
    import cupy as cp
except:
    pass


def calc_auc(embs: np.array, persons: list):
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

    
    

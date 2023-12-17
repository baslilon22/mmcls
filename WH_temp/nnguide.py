import torch
import faiss
import numpy as np

from copy import deepcopy



def knn_score(feas_train, feas, k=10, min=False):
    feas_train = deepcopy(np.array(feas_train))
    feas = deepcopy(np.array(feas))

    index = faiss.IndexFlatIP(feas_train.shape[-1])
    index.add(feas_train)
    D, I = index.search(feas, k)          # [feas_N, k]

    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))
    return scores


class NNGuideOODDetector:
    def __init__(self, knn_k=10):
        self.knn_k = knn_k
        self.scaled_feas_train = None
    
    def setup(self, train_model_outputs):
        feas_train = train_model_outputs['feas']
        logits_train = train_model_outputs['logits']

        confs_train = torch.logsumexp(logits_train, dim=1)                  # [train_N]
        self.scaled_feas_train = feas_train * confs_train[:, None]          # [train_N, 768], same as 'feas_train'

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']

        confs = torch.logsumexp(logits, dim=1)                              # [test_N]
        guidances = knn_score(self.scaled_feas_train, feas, k=self.knn_k)   # [test_N]
        scores = torch.from_numpy(guidances).to(confs.device)*confs         # [test_N]
        return scores
import time

import numpy as np
import pandas as pd
import torch

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

from sklearn.cluster import KMeans


# This implementation originated from https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/kmeans_sampling.py
@AL_ALGORITHMS.register('kmeans')
class KMeansAL(ActiveBase):
    def __init__(self, args, gpu):
        print("kMeans initialized")
        super().__init__(args, gpu)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]
        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
        diff = len(y) - len(idxs_unlabeled)
        y = y[diff:]
        preds = preds[diff:]
        embs = embs[diff:]

        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embs)

        cluster_idxs = cluster_learner.predict(embs)
        centers = cluster_learner.cluster_centers_[cluster_idxs]



        dis = (embs - centers) ** 2
        dis = dis.sum(axis=1)
        chosen = np.array(
            [np.arange(embs.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])

        query_idx = idxs_unlabeled[chosen]

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[chosen] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_unlabeled), list(chosen_all), y, preds, correct, cluster_learner.labels_, dis, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "cluster", "dis", "query_idx"]).T

        return query_idx, query_df

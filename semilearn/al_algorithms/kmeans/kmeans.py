import time

import numpy as np
import pandas as pd
import torch

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase

from sklearn.cluster import KMeans


# This implementation originated from https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/kmeans_sampling.py
# MIT License
#
# Copyright (c) 2021 Natural Language Processing @UCLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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

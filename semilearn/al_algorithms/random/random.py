import time

import numpy as np
import pandas as pd

from semilearn.core.utils import AL_ALGORITHMS
from semilearn.core.activebase import ActiveBase


@AL_ALGORITHMS.register('random')
class Random(ActiveBase):
    def __init__(self, gpu):
        super().__init__(gpu)
        print("Random started", flush=True)

    def query(self, n, clf, data_loaders):
        idxs_unlabeled = np.arange(len(self.idxs_lb))[~self.idxs_lb]

        idxs_full = np.arange(len(self.idxs_lb))
        #chosen = np.random.choice(idxs_unlabeled, n, replace=False)

        x, y, embs, logits, probs, preds = self.get_x_y_embs_logits_probs_preds(clf, data_loaders["sequential_ulb_loader"])
       # if len(x) != len(idxs_unlabeled):
        diff = len(x) - len(idxs_unlabeled)
        y = y[diff:]
        preds = preds[diff:]


        chosen = np.random.randint(len(idxs_unlabeled),size=n)
        query_idx = idxs_unlabeled[chosen]

        chosen_all = np.zeros(len(idxs_unlabeled), dtype=bool)
        chosen_all[chosen] = True

        correct = preds == y

        query_df = pd.DataFrame(
            [list(idxs_full), list(chosen_all), y, preds, correct, query_idx],
            index=["id", "chosen", "label", "pred", "correct", "query_idx"]).T
        return query_idx, query_df

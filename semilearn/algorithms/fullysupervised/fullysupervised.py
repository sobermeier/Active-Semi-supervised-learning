# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('fullysupervised')
class FullySupervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None, flow_logger=None):
        super().__init__(args, net_builder, tb_log, logger, flow_logger)

    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)['logits']
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict
    
    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")
            
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")

    def train_active(self):
        """
        train function with al_algorithms labeling
        """
        self.model.train()
        self.call_hook("before_run")

        idxs_lb_mask = np.zeros(self.n_pool, dtype=bool)
        idxs_tmp = np.arange(self.n_pool)
        np.random.shuffle(idxs_tmp)
        idxs_lb_mask[idxs_tmp[:self.num_query_epochs[0]]] = True
        self.dataset_dict = self.set_dataset(idxs_lb_mask)
        self.al.update(idxs_lb_mask)
        # Sequential loaders --> idxs are not changed
        # be careful when updating idxs_lb_mask, always has to be mapped back
        al_data_loaders = self.get_data_loader_for_al()

        # reset normal data loaders for semi-supervised training with original training routine
        self.loader_dict = self.set_data_loader()
        qs_counter = 1
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            print(f"Epoch: {epoch} Iter: {self.it} (Query at {self.label_epochs})")
            if epoch > 0 and epoch in self.label_epochs:
                print("al_data_loaders", al_data_loaders.keys(), al_data_loaders)
                query_idxs, query_dict = self.al.query(
                    n=self.num_query_epochs[qs_counter],
                    clf=self.model,
                    data_loaders=al_data_loaders
                )
                self.flow_logger.log_querydict(query_dict, epoch)
                qs_counter += 1
                idxs_lb_mask[query_idxs] = True
                self.dataset_dict = self.set_dataset(idxs_lb_mask)
                self.al.update(idxs_lb_mask)
                al_data_loaders = self.get_data_loader_for_al()
                self.loader_dict = self.set_data_loader()

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:
                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            print(f"... Made it through the loader...")
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")


ALGORITHMS['supervised'] = FullySupervised
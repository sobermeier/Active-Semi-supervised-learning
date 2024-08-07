# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            cur_results = {'acc': eval_dict['eval/top-1-acc'],
                           'F1': eval_dict['eval/F1'],
                           'loss': eval_dict['eval/loss'],
                           'balanced_acc': eval_dict['eval/balanced_acc'],
                           'precision': eval_dict['eval/precision'],
                           'recall': eval_dict['eval/recall'],
                           'labels': len(algorithm.dataset_dict["train_lb"])
                           }
            algorithm.flow_logger.log_results(result=cur_results, step=algorithm.epoch)
            algorithm.flow_logger.log_log_file()

            # update best metrics
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it

    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict

    def after_train_epoch(self, algorithm):

        algorithm.print_fn(f"Epoch {algorithm.epoch} ({algorithm.label_epochs}): AFTER_TRAIN_EPOCH")
        if algorithm.epoch + 1 in algorithm.label_epochs or algorithm.epoch in algorithm.label_epochs:
            algorithm.print_fn(f"\t Eval Test at epoch: {algorithm.epoch}")
            # if not algorithm.args.multiprocessing_distributed or (
            #     algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            #     save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            #     algorithm.save_model('latest_model.pth', save_path)
            #
            results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
            algorithm.print_fn(algorithm.loader_dict)
            if 'eval' in algorithm.loader_dict:
                algorithm.print_fn(f"\t eval is in loader_dict")
                # load the best model and evaluate on test dataset
                # TODO check do i need to load it? should be already there? we always check
                # latest_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'latest_model.pth')
                # algorithm.load_model(latest_model_path)
                test_dict = algorithm.evaluate('eval')
                results_dict[f'eval/acc_{algorithm.epoch}'] = test_dict['eval/top-1-acc']
                # results_dict['eval/epoch'] = test_dict['eval/top-1-acc']
                algorithm.print_fn(f"\n \t {results_dict}")
            algorithm.results_dict = results_dict

# TODO: csv-Logging (MLFlow schließen -> nur csvs, MlFlow Pfad in Config -> bei leer nur csv)
#
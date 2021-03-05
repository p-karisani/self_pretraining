import gc
import os
import copy
import random
import math

from sklearn import mixture
from termcolor import colored
import numpy as np

import torch
import torch.nn.functional as F

from self_pretraining.src.ELib import ELib
from self_pretraining.src.ETweet import ETweet
from self_pretraining.src.ETweet import ELoadType
from self_pretraining.src.ELblConf import ELblConf
from self_pretraining.src.ELbl import ELbl
from self_pretraining.src.EBert import EBert, EBertCLSType
from self_pretraining.src.EBertUtils import EBertConfig, EInputBundle, EBalanceBatchMode
from self_pretraining.src.EVar import EVar


class EPretrainCMD:
    none = 0
    bert_reg = 3

    @staticmethod
    def get(value):
        if value == 'bert':
            return EPretrainCMD.bert
        elif value == 'bert_mine':
            return EPretrainCMD.bert_mine
        elif value == 'bert_reg':
            return EPretrainCMD.bert_reg
        return EPretrainCMD.none


class DistInfo:

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        ELib.PASS()

    def __str__(self):
        return 'mean: ' + str(self.mean) + '\tvar: ' + str(self.var)


class EPretrainProj:
    ######### hyper-parameters
    sample_size = 1000  # initial number of unlabeled documents
    sample_step = 2000  # k (see the paper)
    tmperT = 3  # temperature (see the paper)
    pretrain_epochs = 1
    finetune_epochs = 3
    finetune_log_softmax_weight = 0.3  # lambda (see the paper)
    distr_moment = 0.1  # alpha (see the paper)
    #########

    @staticmethod
    def __sample_from_unlabeled(unlabeled_bundle, count, seed):
        random.seed(seed)
        count = min(count, len(unlabeled_bundle.tws))
        sample_set = set()
        while True:
            tw_ind = random.randint(0, len(unlabeled_bundle.tws) - 1)
            if unlabeled_bundle.tws[tw_ind] not in sample_set:
                sample_set.add(unlabeled_bundle.tws[tw_ind])
                if len(sample_set) >= count:
                    break
        drop_list = [tw for tw in unlabeled_bundle.tws if tw not in sample_set]
        EInputBundle.remove(unlabeled_bundle, drop_list)
        ELib.PASS()

    @staticmethod
    def __stratified_sample_from_bundle(bundle, lc, seed, count):
        bundle_copy = copy.deepcopy(bundle)

        tws = ETweet.random_stratified_sample(bundle_copy.tws, lc, 1 - (count / len(bundle.tws)), seed)
        EInputBundle.remove(bundle_copy, tws)
        return bundle_copy

    @staticmethod
    def __update_label_info(unlabeled_bundle, lbls_list, soft_lbl_list, do_softmax, temperature):
        if len(lbls_list) == 1:
            for tw_ind, _ in enumerate(unlabeled_bundle.tws):
                unlabeled_bundle.input_y[0][tw_ind] = lbls_list[0][tw_ind]
                if do_softmax:
                    unlabeled_bundle.input_y_row[0][tw_ind] = \
                        F.softmax(torch.tensor(soft_lbl_list[0][tw_ind]) / temperature).numpy().tolist()
                else:
                    unlabeled_bundle.input_y_row[0][tw_ind] = soft_lbl_list[0][tw_ind]
        elif len(lbls_list) == 2:
            for tw_ind, _ in enumerate(unlabeled_bundle.tws):
                if do_softmax:
                    y_row = F.softmax(torch.tensor(soft_lbl_list[0][tw_ind]) / temperature).numpy() + \
                            F.softmax(torch.tensor(soft_lbl_list[1][tw_ind]) / temperature).numpy()
                else:
                    y_row = np.array(soft_lbl_list[0][tw_ind]) + np.array(soft_lbl_list[1][tw_ind])
                y_row = (y_row / 2).tolist()
                unlabeled_bundle.input_y_row[0][tw_ind] = y_row
                unlabeled_bundle.input_y[0][tw_ind] = y_row.index(max(y_row))
            ELib.PASS()
        else:
            raise Exception('not implemented function!')
        ELib.PASS()

    @staticmethod
    def __twin_net_transform_labels(pre_distr_info, lbls, logits, temperature, moment):
        ## collect the class probabilities as the data
        data_all, data_b1, data_b2 = list(), list(), list()
        for cur_logit in logits:
            sft = F.softmax(torch.tensor(cur_logit) / temperature).numpy().tolist()
            data_all.append(sft[1])
            if sft[1] < 0.5:
                data_b1.append(sft[1])
            else:
                data_b2.append(sft[1])
        ## fit the current distributions
        data_b1 = np.array(data_b1).reshape(-1, 1)
        current_distr_b1 = mixture.GaussianMixture(n_components=1, covariance_type='full')
        current_distr_b1.fit(data_b1)
        data_b2 = np.array(data_b2).reshape(-1, 1)
        current_distr_b2 = mixture.GaussianMixture(n_components=1, covariance_type='full')
        current_distr_b2.fit(data_b2)
        cur_distr_info = [DistInfo(current_distr_b1.means_[0][0], current_distr_b1.covariances_[0][0][0]),
                          DistInfo(current_distr_b2.means_[0][0], current_distr_b2.covariances_[0][0][0])]
        ## if this is the first iteration, do nothing
        if pre_distr_info is None:
            for d_ind, d_cur in enumerate(data_all):
                lbls[d_ind] = 0 if d_cur < 0.5 else 1
                logits[d_ind] = [1 - d_cur, d_cur]
            return cur_distr_info
        ## update the current distribution
        upd_distr_info = [DistInfo(0, 0), DistInfo(0, 0)]
        upd_distr_info[0].mean = (1 - moment) * cur_distr_info[0].mean + moment * pre_distr_info[0].mean
        upd_distr_info[0].var = (1 - moment) * cur_distr_info[0].var + moment * pre_distr_info[0].var
        upd_distr_info[1].mean = (1 - moment) * cur_distr_info[1].mean + moment * pre_distr_info[1].mean
        upd_distr_info[1].var = (1 - moment) * cur_distr_info[1].var + moment * pre_distr_info[1].var
        ## transform data
        data_all_new = list()
        for d_ind, d_cur in enumerate(data_all):
            if d_cur < 0.5:
                d_cur_prime = (d_cur - cur_distr_info[0].mean) / math.sqrt(cur_distr_info[0].var) # standardize it
                d_new = d_cur_prime * math.sqrt(upd_distr_info[0].var) + upd_distr_info[0].mean # tnasform to new distr
                d_new = max(min(d_new, 1), 0) # project to [0, 1]
            else:
                d_cur_prime = (d_cur - cur_distr_info[1].mean) / math.sqrt(cur_distr_info[1].var)
                d_new = d_cur_prime * math.sqrt(upd_distr_info[1].var) + upd_distr_info[1].mean
                d_new = max(min(d_new, 1), 0)
            data_all_new.append(d_new)
        ## update the labels
        for d_ind, d_cur in enumerate(data_all_new):
            lbls[d_ind] = 0 if d_cur < 0.5 else 1
            logits[d_ind] = [1 - d_cur, d_cur]
        return upd_distr_info

    @staticmethod
    def __run_twin_net(config, lc, query, this_train_bundle, valid_bundle, test_bundle, unlabeled_bundle):
        pretrain_epochs_virtual = 1000000 # to make the learning-rate flat during pre-training
        iterations = math.ceil(1 + (len(unlabeled_bundle.tws) - EPretrainProj.sample_size) / EPretrainProj.sample_step)
        teacher, student = None, None
        distr_info = None
        config.cls_type = EBertCLSType.simple
        config.epoch_count = EPretrainProj.finetune_epochs
        seed = config.seed
        print(colored('initial teacher training ...', 'blue'))
        student = EBert(config)
        student.train([this_train_bundle])
        for cur_itr in range(iterations):
            del teacher
            teacher = student
            print(colored('>>> iteration {}/{}, sample-size {}'.format(
                cur_itr + 1, iterations, EPretrainProj.sample_size), 'red'))
            print(colored('teacher labeling ...', 'blue'))
            ## preparing unlabeled and labeling by teacher
            cur_unlabeled_bundle = copy.deepcopy(unlabeled_bundle)
            EPretrainProj.__sample_from_unlabeled(cur_unlabeled_bundle, EPretrainProj.sample_size,
                                                  seed + 34 * (cur_itr + 1))
            unl_lbl, unl_lgs, _, _ = teacher.test(cur_unlabeled_bundle)
            distr_info = EPretrainProj.__twin_net_transform_labels(distr_info, unl_lbl, unl_lgs,
                                                                   EPretrainProj.tmperT, EPretrainProj.distr_moment)
            print('neg-mean: {:.6f}, neg-std: {:.6f}, pos-mean: {:.6f}, pos-std: {:.6f}'.format(
                distr_info[0].mean, math.sqrt(distr_info[0].var), distr_info[1].mean, math.sqrt(distr_info[1].var)))
            EPretrainProj.__update_label_info(cur_unlabeled_bundle, [unl_lbl], [unl_lgs], False, EPretrainProj.tmperT)
            tr_lbl, tr_lgs, _, _ = teacher.test(this_train_bundle)
            EPretrainProj.__update_label_info(this_train_bundle, [this_train_bundle.input_y[0]], [tr_lgs], True,
                                              EPretrainProj.tmperT)
            ## training student
            del student
            print(colored('student training on unlabeled ...', 'blue'))
            config_st = copy.deepcopy(teacher.config)
            config_st.seed = seed + 566 * (cur_itr + 1)
            config_st.epoch_count = EPretrainProj.pretrain_epochs
            config_st.train_by_log_softmax = True
            config_st.training_log_softmax_weight = 1
            config_st.training_softmax_temperature = EPretrainProj.tmperT
            config_st.balance_batch_mode = EBalanceBatchMode.label_based
            student = EBert(config_st)
            student.train([cur_unlabeled_bundle], setup_learning_tools=True,
                          extra_scheduled_trainset_size=len(this_train_bundle.tws),
                          extra_scheduled_epochs=pretrain_epochs_virtual)
            print(colored('student training on labeled ...', 'blue'))
            config_st.epoch_count = EPretrainProj.finetune_epochs
            config_st.train_by_log_softmax = True
            config_st.training_log_softmax_weight = EPretrainProj.finetune_log_softmax_weight
            config_st.training_softmax_temperature = EPretrainProj.tmperT
            config_st.balance_batch_mode = EBalanceBatchMode.label_based
            student.train([this_train_bundle])
            gc.collect()
            torch.cuda.empty_cache()
            if EPretrainProj.sample_size >= len(unlabeled_bundle.tws):
                break
            seed = seed + 324 * (cur_itr + 1)
            EPretrainProj.sample_size += EPretrainProj.sample_step
        ## testing
        print(colored('testing ...', 'green'))
        _, logit_t, _, _ = teacher.test(test_bundle)
        _, logit_s, _, _ = student.test(test_bundle)
        result_lbl = ELib.majority_logits([logit_t, logit_s])
        perf = ELib.calculate_metrics(test_bundle.input_y[0], result_lbl)
        print('final test results L1> ' + ELib.get_string_metrics(perf))
        del teacher, student
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def run(cmd, per_query, train_path, valid_path_nullable, test_path_nullable,
            unlabeled_path_nullable, model_path, model_path_2,
            lm_model_path, t_lbl_path_1, t_lbl_path_2, output_dir,
            device, device_2, seed, train_sample, unlabeled_sample):
        cmd = EPretrainCMD.get(cmd)
        lc = ELblConf(0, 1,
                      [ELbl(0, EVar.LblNonEventHealth),
                       ELbl(1, EVar.LblEventHealth)]) # mapping the dataset labels to negative and postivie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        queries = [None]
        if per_query:
            queries = ETweet.get_queries(ETweet.load(train_path, ELoadType.none))
        for q_ind, cur_query in enumerate(queries):
            if cur_query is not None:
                print('>>>>>>>> "' + cur_query + '" began')
            if cmd == EPretrainCMD.bert_reg:
                config = EBertConfig.get_config(cmd, EBertCLSType.none, model_path, model_path_2,
                                                lm_model_path, t_lbl_path_1, t_lbl_path_2, output_dir, 5,
                                                device, device_2, seed, cur_query, gradient_checkpointing=False,
                                                check_early_stopping=False)
                train_bundle, valid_bundle, test_bundle, unlabeled_bundle = EInputBundle.get_data(
                    config.label_count, lc, train_path, valid_path_nullable, test_path_nullable,
                    unlabeled_path_nullable, cur_query)
                train_bundle = EPretrainProj.__stratified_sample_from_bundle(train_bundle, lc, config.seed, train_sample)
                EPretrainProj.__sample_from_unlabeled(unlabeled_bundle, unlabeled_sample, config.seed)
                EPretrainProj.__run_twin_net(config, lc, cur_query, train_bundle, valid_bundle, test_bundle,
                                             unlabeled_bundle)
                ELib.PASS()
        ELib.PASS()



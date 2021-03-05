import os
import queue
import sys
import threading

import numpy as np
import copy

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from self_pretraining.src.ELib import ELib
from self_pretraining.src.ETweet import ETweet
from self_pretraining.src.ETweet import ELoadType
from self_pretraining.src.EVar import EVar


class EBalanceBatchMode:
    none = 0
    label_based = 1
    meta_based_discrete = 2
    meta_based_continuous = 3


class EInputListMode:
    sequential = 0
    parallel = 1
    parallel_full = 2


class EBertConfig:

    def __init__(self, cmd, cls_type, bert_config, label_count, model_path, model_path_2,
                 lm_model_path, t_lbl_path_1, t_lbl_path_2, output_dir, device, device_2, dropout_prob,
                 max_seq, batch_size, epoch_count, seed, learn_rate, early_stopping_patience,
                 max_grad_norm, weight_decay, adam_epsilon, warmup_steps, train_by_log_softmax,
                 training_log_softmax_weight, training_softmax_temperature, balance_batch_mode,
                 take_train_checkpoints, train_checkpoint_interval, check_early_stopping):
        self.cmd = cmd
        self.cls_type = cls_type
        self.bert_config = bert_config
        self.label_count = label_count
        self.model_path = model_path
        self.model_path_2 = model_path_2
        self.lm_model_path = lm_model_path
        self.t_lbl_path_1 = t_lbl_path_1
        self.t_lbl_path_2 = t_lbl_path_2
        self.output_dir = output_dir
        self.device = device
        self.device_2 = device_2
        self.dropout_prob = dropout_prob
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.seed = seed
        self.learn_rate = learn_rate
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.train_by_log_softmax = train_by_log_softmax
        self.training_log_softmax_weight = training_log_softmax_weight
        self.training_softmax_temperature = training_softmax_temperature
        self.balance_batch_mode = balance_batch_mode
        self.take_train_checkpoints = take_train_checkpoints
        self.train_checkpoint_interval = train_checkpoint_interval
        self.check_early_stopping = check_early_stopping
        ELib.PASS()

    @staticmethod
    def __get_internal_bert_config(model_path, gradient_checkpointing):
        label_count = 2
        dropout_prob = EVar.Dropout
        task_name = 'cola'
        b_config = BertConfig(num_labels=label_count,
                              hidden_dropout_prob=dropout_prob, finetuning_task=task_name,
                              output_hidden_states=False, output_attentions=False,
                              gradient_checkpointing=gradient_checkpointing)
        b_config.layer_norm_eps = 1e-12
        return b_config

    @staticmethod
    def get_config(cmd, cls_type, model_path, model_path_2, lm_model_path,
                   t_lbl_path_1, t_lbl_path_2, output_dir, epoch_count,
                   device, device_2, seed, query, learn_rate=0.00005,
                   gradient_checkpointing=True, check_early_stopping=True):
        if query is not None:
            if model_path is not None and os.path.exists(os.path.join(model_path, query)):
                model_path = os.path.join(model_path, query)
            if model_path_2 is not None and os.path.exists(os.path.join(model_path_2, query)):
                model_path_2 = os.path.join(model_path_2, query)
            if lm_model_path is not None and os.path.exists(os.path.join(lm_model_path, query)):
                lm_model_path = os.path.join(lm_model_path, query)
            if t_lbl_path_1 is not None and os.path.exists(os.path.join(t_lbl_path_1, query)):
                t_lbl_path_1 = os.path.join(t_lbl_path_1, query)
            if t_lbl_path_2 is not None and os.path.exists(os.path.join(t_lbl_path_2, query)):
                t_lbl_path_2 = os.path.join(t_lbl_path_2, query)
            output_dir = os.path.join(output_dir, query)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        b_config = EBertConfig.__get_internal_bert_config(model_path, gradient_checkpointing)
        config = EBertConfig(cmd=cmd, cls_type=cls_type, bert_config=b_config, label_count=b_config.num_labels,
                             model_path=model_path, model_path_2=model_path_2, lm_model_path=lm_model_path,
                             t_lbl_path_1=t_lbl_path_1, t_lbl_path_2=t_lbl_path_2, output_dir=output_dir,
                             device=device, device_2=device_2, dropout_prob=b_config.hidden_dropout_prob,
                             max_seq=160, batch_size=EVar.BertBatchSize, epoch_count=epoch_count, seed=seed,
                             learn_rate=learn_rate, early_stopping_patience=epoch_count * 10, max_grad_norm=1.0,
                             weight_decay=0.0, adam_epsilon=1e-8, warmup_steps=0, train_by_log_softmax=False,
                             training_log_softmax_weight=1, training_softmax_temperature=1,
                             balance_batch_mode=EBalanceBatchMode.label_based, take_train_checkpoints=False,
                             train_checkpoint_interval=1, check_early_stopping=check_early_stopping)
        return config


class EBertTrainingTools:

    @staticmethod
    def get_parameter_groups(module, learn_rate, weight_decay, adam_epsilon):
        result = list()
        kids = list(module.children())
        for cur_mod in kids:
            if isinstance(cur_mod, nn.ModuleList):
                grad_kids = EBertTrainingTools.get_parameter_groups(cur_mod, learn_rate, weight_decay, adam_epsilon)
                result.extend(grad_kids)
            else:
                result.append(
                    {
                        'params': cur_mod.parameters(),
                        'lr': learn_rate,
                        'weight_decay': weight_decay,
                        'eps': adam_epsilon
                    }
                )
        return result

    @staticmethod
    def get_optimizer_by_modules(module, learn_rate, weight_decay, adam_epsilon):
        param_groups = EBertTrainingTools.get_parameter_groups(module, learn_rate, weight_decay, adam_epsilon)
        optimizer = AdamW(param_groups, lr=learn_rate, weight_decay=weight_decay, eps=adam_epsilon)
        return optimizer

    @staticmethod
    def get_optimizer(parameters, learn_rate, weight_decay, adam_epsilon):
        ## two sets of params
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in classifier.named_parameters()
        #                 if not any(nd in n for nd in no_decay)],
        #      'weight_decay': weight_decay},
        #     {'params': [p for n, p in classifier.named_parameters()
        #                 if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0}]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate, eps=adam_epsilon)

        ## paper optimizer
        optimizer = AdamW(parameters, lr=learn_rate, weight_decay=weight_decay, eps=adam_epsilon)

        ## torch optimizer
        # optimizer = torch.optim.Adam(parameters, lr=learn_rate, eps=adam_epsilon)
        return optimizer

    @staticmethod
    def get_training_steps(epoch_count, batch_size, train_bundle_list, input_mode,
                           extra_trainset_size=0, extra_epochs=0):
        if input_mode == EInputListMode.sequential:
            train_size = 0
            for cur_bund in train_bundle_list:
                train_size += len(cur_bund.input_x)
        elif input_mode == EInputListMode.parallel:
            train_size = len(train_bundle_list[0].input_x)
            for cur_bund in train_bundle_list:
                train_size = min(train_size, len(cur_bund.input_x))
        elif input_mode == EInputListMode.parallel_full:
            train_size = len(train_bundle_list[0].input_x)
            for cur_bund in train_bundle_list:
                train_size = max(train_size, len(cur_bund.input_x))
        training_steps = (train_size // batch_size) * epoch_count
        training_steps += (extra_trainset_size // batch_size) * extra_epochs
        training_steps = int(training_steps)
        return training_steps

    @staticmethod
    def get_scheduler(optimizer, epoch_count, batch_size, warmup_steps, train_bundle_list,
                      input_mode, extra_trainset_size=0, extra_epochs=0):
        training_steps = EBertTrainingTools.get_training_steps(epoch_count, batch_size, train_bundle_list,
                                                               input_mode, extra_trainset_size, extra_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)
        return scheduler


class ETaskState:

    def __init__(self, name, early_stopping_patience=None):
        self.reset()
        self.name = name
        if early_stopping_patience is None:
            self.learning_state = None
        else:
            self.learning_state = ETaskLearningState(early_stopping_patience)

    def update(self, b_size, b_loss, b_lbl_true, b_lbl_pred, b_logits):
        self.size += b_size
        self.loss += b_loss
        self.lbl_true.extend(b_lbl_true)
        self.lbl_pred.extend(b_lbl_pred)
        self.logits.extend(b_logits)

    def update_meta_1(self, meta_1):
        self.meta_1 += meta_1

    def update_meta_2(self, meta_2):
        self.meta_2 += meta_2

    def update_meta_3(self, meta_3):
        self.meta_3 += meta_3

    def reset(self):
        self.f1 = 0
        self.acc = 0
        self.size = 0
        self.loss = 0
        self.meta_1 = 0
        self.meta_2 = 0
        self.meta_3 = 0
        self.lbl_true = list()
        self.lbl_pred = list()
        self.logits = list()


class ETaskLearningState:

    def __init__(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience
        self.loss_list = list()
        self.best_index = -1
        self.best_model = None
        ELib.PASS()

    def should_stop(self, cur_loss, cur_model, device):
        self.loss_list.append(cur_loss)
        if len(self.loss_list) == 1:
            self.best_index = 0
            cur_model.cpu()
            self.best_model = copy.deepcopy(cur_model)
            cur_model.to(device)
            return False
        if self.loss_list[self.best_index] >= self.loss_list[-1]:
            self.best_index = len(self.loss_list) - 1
            cur_model.cpu()
            self.best_model = copy.deepcopy(cur_model)
            cur_model.to(device)
            return False
        if (len(self.loss_list) - (self.best_index + 1)) >= self.early_stopping_patience:
            return True
        return False


class ESyncObj:

    def __init__(self, seed, model_count, synchronized_bundle_indices=None):
        self.seed = seed
        self.model_count = model_count
        self.lock_dataset = threading.Lock()
        self.lock_batch = threading.Lock()
        self.lock_loss_calculation = threading.Lock()
        self.sync_bundle_indices = synchronized_bundle_indices
        self.sync_bundle_batches = dict()
        self.sync_bundle_batches_sizes = list()
        self.sync_counter = model_count # this is needed for the first iteration
        self.sync_list = list()
        self.meta_list = list()
        for ind in range(model_count):
            self.sync_list.append(queue.Queue())
            self.meta_list.append(None)
        ELib.PASS()

    def reset(self):
        self.sync_counter = 0
        self.sync_bundle_batches = dict()
        self.sync_bundle_batches_sizes = list()

    def verify_synced_batch_sizes(self):
        if len(self.sync_bundle_batches_sizes) == 0:
            return True
        b_min = self.sync_bundle_batches_sizes[0][0]
        b_max = self.sync_bundle_batches_sizes[0][1]
        for cur_size in self.sync_bundle_batches_sizes:
            if b_min != cur_size[0] or b_max != cur_size[1]:
                return False
        return True


class EBertDataset(Dataset):

    def __init__(self, input_bundle, tokenizer, max_seq):
        self.input_bundle = input_bundle
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.out_of_bound = 0

    def __find_sublist(self, sub, bigger):
        first, rest = sub[0], sub[1:]
        pos = 0
        try:
            while True:
                pos = bigger.index(first, pos) + 1
                if not rest or bigger[pos:pos + len(rest)] == rest:
                    return pos - 1
        except ValueError:
            return -1

    def __get_query_vec(self, tokens, queries):
        result = [0] * len(tokens)
        for cur_q in queries:
            q_tokens = self.tokenizer.tokenize(cur_q)
            q_ind = self.__find_sublist(q_tokens, tokens)
            if q_ind >= 0:
                result[q_ind] = 1
            ELib.PASS()
        ELib.PASS()
        return result

    def __getitem__(self, index):
        tokens = self.tokenizer.tokenize(self.input_bundle.input_x[index])
        if len(tokens) + 2 > self.max_seq:
            tokens = tokens[:self.max_seq - 2]
            self.out_of_bound += 1
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [self.tokenizer.cls_token_id] + tokens_ids + [self.tokenizer.sep_token_id]
        tokens_len = len(tokens_ids)
        tokens_ids += [0] * (self.max_seq - tokens_len)
        tokens_ids_ts = torch.tensor(tokens_ids, dtype=torch.long)
        query_vec = self.__get_query_vec(tokens, self.input_bundle.queries[index])
        query_vec = [0] + query_vec + [0]
        query_vec += [0] * (self.max_seq - tokens_len)
        query_vec_ts = torch.tensor(query_vec, dtype=torch.long)
        tokens_type_ids = [0] * self.max_seq
        tokens_type_ids_ts = torch.tensor(tokens_type_ids, dtype=torch.long)
        attention_mask = [1] * tokens_len + [0] * (self.max_seq - tokens_len)
        attention_mask_ts = torch.tensor(attention_mask, dtype=torch.long)
        result = {
            'x' : tokens_ids_ts,
            'type' : tokens_type_ids_ts,
            'mask' : attention_mask_ts,
            'query' : query_vec_ts,
            'weight' : self.input_bundle.input_weight[index],
            'meta' : self.input_bundle.input_meta[index],
            'docid' : self.input_bundle.tws[index].Tweetid,
            'len' : tokens_len,
            'task_list': self.input_bundle.task_list}
        for ta_ind, ta_name in enumerate(self.input_bundle.task_list):
            result['y_' + str(ta_ind)] = torch.from_numpy(np.array(self.input_bundle.input_y[ta_ind][index],
                                                                   dtype=np.long))
            result['y_row_' + str(ta_ind)] = torch.from_numpy(np.array(self.input_bundle.input_y_row[ta_ind][index],
                                                                       dtype=np.float32))
        # self.__print_inst(tokens, tokens_len, tokens_ids, query_vec)
        # self.__print_example(index, result)
        return result

    def __len__(self):
        return len(self.input_bundle.input_x)

    def __print_inst(self, tokens, tokens_len, tokens_ids, query_vec):
        tokens = ['['] + tokens + [']'] + ['?'] * (self.max_seq - tokens_len)
        for ind, _ in enumerate(tokens):
            print(str(ind) + ': ' + tokens[ind].ljust(7) +
                  ' [' + str(tokens_ids[ind]).ljust(5) + ']' + '(' + str(query_vec[ind]) + ')')
        print()
        ELib.PASS()

    def __print_example(self, ind, ex):
        cur_input_ids = ex['x'].numpy().tolist()
        cur_attention_mask = ex['mask'].numpy().tolist()
        cur_token_type_ids = ex['type'].numpy().tolist()
        cur_labels = ex['y'].numpy().tolist()
        cur_labels_row = ex['y_row'].numpy().tolist()
        print(ind, ':>')
        print(cur_input_ids)
        print(cur_attention_mask)
        print(cur_token_type_ids)
        print(cur_labels)
        print(cur_labels_row)
        print()


class EInputBundle:

    def __init__(self, task_list, input_x, input_y, input_y_row, queries, input_weight, input_meta, tws):
        self.task_list = task_list
        self.input_x = input_x
        self.input_y = input_y
        self.input_y_row = input_y_row
        self.queries = queries
        self.input_weight = input_weight
        self.input_meta = input_meta
        self.tws = tws
        ELib.PASS()

    @staticmethod
    def get_input_bundle(task_list, tws, lc, filter_query, tokenize_by_etokens, pivot_query, label_count,
                         max_set_length = 0):
        if filter_query is not None:
            tws = ETweet.filter_by_query(tws, filter_query)
        if 0 < max_set_length:
            tws = tws[:max_set_length]
        result_x = list()
        result_y = list()
        [result_y.append([]) for _ in range(len(task_list))]
        result_y_row = list()
        [result_y_row.append([]) for _ in range(len(task_list))]
        queries = list()
        weights = list()
        meta = list()
        for tw_ind, cur_tw in enumerate(tws):
            tokenized = ELib.tokenize_tweet_text(cur_tw, True, tokenize_by_etokens, pivot_query, cur_tw.QueryList)
            result_x.append(tokenized)
            lbl = lc.get_correct_new_label(cur_tw.Label)
            [result_y[t_ind].append(lbl) for t_ind in range(len(task_list))]
            [result_y_row[t_ind].append([0 for _ in range(label_count)]) for t_ind in range(len(task_list))]
            for t_ind in range(len(task_list)):
                result_y_row[t_ind][-1][lbl] = 1
            if pivot_query is not None:
                queries.append([pivot_query])
            elif cur_tw.Query == ETweet.tokenDummyQuery:
                queries.append(cur_tw.QueryList)
            else:
                queries.append([cur_tw.Query])
            weights.append(1.0)
            meta.append(0.0)
        result = EInputBundle(task_list, result_x, result_y, result_y_row, queries, weights, meta, tws)
        return result

    @staticmethod
    def remove(bundle, to_remove_tws):
        id_dict = dict()
        for cur_ind, cur_tw in enumerate(bundle.tws):
            id_dict[cur_tw.Tweetid] = cur_ind
        to_delete = list()
        for cur_tw in to_remove_tws:
            cur_ind = id_dict[cur_tw.Tweetid]
            to_delete.append(cur_ind)
        to_delete.sort(reverse=True)
        for cur_ind in to_delete:
            del bundle.input_x[cur_ind]
            for y_ind in range(len(bundle.input_y)):
                del bundle.input_y[y_ind][cur_ind]
                del bundle.input_y_row[y_ind][cur_ind]
            del bundle.queries[cur_ind]
            del bundle.input_weight[cur_ind]
            del bundle.input_meta[cur_ind]
            del bundle.tws[cur_ind]
        ELib.PASS()

    @staticmethod
    def prune(bundle, to_keep_tws):
        to_remove = ETweet.filter_by_tweets(bundle.tws, to_keep_tws)
        EInputBundle.remove(bundle, to_remove)

    @staticmethod
    def append(tgt_bundle, src_bundle, tws):
        id_dict = dict()
        for cur_ind, cur_tw in enumerate(src_bundle.tws):
            id_dict[cur_tw.Tweetid] = cur_ind
        to_append = list()
        for cur_tw in tws:
            cur_ind = id_dict[cur_tw.Tweetid]
            to_append.append(cur_ind)
        for cur_ind in to_append:
            tgt_bundle.input_x.append(copy.deepcopy(src_bundle.input_x[cur_ind]))
            for y_ind in range(len(tgt_bundle.input_y)):
                tgt_bundle.input_y[y_ind].append(copy.deepcopy(src_bundle.input_y[y_ind][cur_ind]))
                tgt_bundle.input_y_row[y_ind].append(copy.deepcopy(src_bundle.input_y_row[y_ind][cur_ind]))
            tgt_bundle.queries.append(copy.deepcopy(src_bundle.queries[cur_ind]))
            tgt_bundle.input_weight.append(copy.deepcopy(src_bundle.input_weight[cur_ind]))
            tgt_bundle.input_meta.append(copy.deepcopy(src_bundle.input_meta[cur_ind]))
            tgt_bundle.tws.append(copy.deepcopy(src_bundle.tws[cur_ind]))
        ELib.PASS()

    @staticmethod
    def combine_input_bundle(first, second):
        result_x = list()
        result_y = list()
        [result_y.append([]) for _ in range(len(first.task_list))]
        result_y_row = list()
        [result_y_row.append([]) for _ in range(len(first.task_list))]
        queries = list()
        result_weight = list()
        result_meta = list()
        tws = list()
        result = EInputBundle(first.task_list, result_x, result_y, result_y_row, queries,
                              result_weight, result_meta, tws)
        for tw_ind, cur_tw in enumerate(first.tws):
            result.input_x.append(first.input_x[tw_ind])
            for ta_ind in range(len(first.task_list)):
                result.input_y[ta_ind].append(first.input_y[ta_ind][tw_ind])
                if first.input_y_row is not None:
                    result.input_y_row[ta_ind].append(first.input_y_row[ta_ind][tw_ind])
            result.queries.append(first.queries[tw_ind])
            result.input_weight.append(first.input_weight[tw_ind])
            result.input_meta.append(first.input_meta[tw_ind])
            result.tws.append(first.tws[tw_ind])
        for tw_ind, cur_tw in enumerate(second.tws):
            result.input_x.append(second.input_x[tw_ind])
            for ta_ind in range(len(second.task_list)):
                result.input_y[ta_ind].append(second.input_y[ta_ind][tw_ind])
                if second.input_y_row is not None:
                    result.input_y_row[ta_ind].append(second.input_y_row[ta_ind][tw_ind])
            result.queries.append(second.queries[tw_ind])
            result.input_weight.append(second.input_weight[tw_ind])
            result.input_meta.append(second.input_meta[tw_ind])
            result.tws.append(second.tws[tw_ind])
        return result

    @staticmethod
    def get_data(label_count, lc, train_path_nullable, valid_path_nullable, test_path_nullable,
                 unlabeled_path_nullable, filter_query=None, remove_unlabeled_test_tweets=False,
                 tokenize_by_etokens=False, pivot_query=None, max_set_length = 0):
        train_bundle, valid_bundle, test_bundle, unlabeled_bundle, = [None] * 4
        if train_path_nullable is not None:
            tws_train = ETweet.load(train_path_nullable, ELoadType.none, tweet_file=False)
            train_bundle = EInputBundle.get_input_bundle([EVar.DefaultTask], tws_train, lc, filter_query,
                                                         tokenize_by_etokens, pivot_query, label_count,
                                                         max_set_length)
        if valid_path_nullable is not None:
            tws_valid = ETweet.load(valid_path_nullable, ELoadType.none, tweet_file=False)
            valid_bundle = EInputBundle.get_input_bundle([EVar.DefaultTask], tws_valid, lc, filter_query,
                                                         tokenize_by_etokens, pivot_query, label_count,
                                                         max_set_length)
        if test_path_nullable is not None:
            tws_test = ETweet.load(test_path_nullable, ELoadType.none, tweet_file=False)
            if remove_unlabeled_test_tweets:
                print('removing unlabeled test tweets ...')
                ind = 0
                while ind < len(tws_test):
                    if tws_test[ind].Label == 0:
                        del tws_test[ind]
                        ELib.PASS()
                    else:
                        ind += 1
            test_bundle = EInputBundle.get_input_bundle([EVar.DefaultTask], tws_test, lc, filter_query,
                                                        tokenize_by_etokens, pivot_query, label_count,
                                                        max_set_length)
        if unlabeled_path_nullable is not None:
            tws_unlabeled = ETweet.load(unlabeled_path_nullable, ELoadType.none, tweet_file=False)
            unlabeled_bundle = EInputBundle.get_input_bundle([EVar.DefaultTask], tws_unlabeled, lc, filter_query,
                                                             tokenize_by_etokens, pivot_query, label_count,
                                                             max_set_length)
        return train_bundle, valid_bundle, test_bundle, unlabeled_bundle

    @staticmethod
    def get_tweet_query_bundles(label_count, lc, tw_path, max_set_length = 0, remove_lbls = False):
        # skips ETokens!
        tws = ETweet.load(tw_path, ELoadType.none)
        if remove_lbls:
            for cur_tw in tws:
                cur_tw.Label = 0
        tws_q = ETweet.split_by_query(tws)
        result = list()
        for cur_tws in tws_q:
            cur_bund = EInputBundle.get_input_bundle([EVar.DefaultTask], cur_tws, lc, None,
                                                             False, None, label_count, max_set_length)
            result.append(cur_bund)
        return result

    @staticmethod
    def populate_bundle(bundle, lc, size, seed):
        size_needed = size - len(bundle.tws)
        ratio_needed = float(size_needed) / len(bundle.tws)
        tws = ETweet.random_stratified_sample(bundle.tws, lc, ratio_needed, seed, True)
        if len(tws) + len(bundle.tws) > size:
            tws = tws[:len(tws) - 1]
        EInputBundle.append(bundle, bundle, tws)
        ELib.PASS()


class ETokenAligner:

    @staticmethod
    def __reconstruct_bert_tokens(bert_tokens):
        result = list()
        for ind in range(len(bert_tokens)):
            if bert_tokens[ind].startswith("##"):
                result[-1][0] = result[-1][0] + bert_tokens[ind][2:]
            else:
                result.append([bert_tokens[ind], ind])
        return result

    @staticmethod
    def __align_tokens_in_tweet(pivot, pivot_pos, bert_tokens_rec, bert_tokens_rec_ind):
        span = 1
        while True:
            phrase = ''.join([entry[0] for entry in
                              bert_tokens_rec[bert_tokens_rec_ind: bert_tokens_rec_ind + span]])
            if pivot == phrase:
                return span, phrase
            if pivot_pos == 'U' and phrase == 'www':
                return span, 'www'
            if phrase == '[UNK]':
                return 1, pivot
            span += 1
            if len(bert_tokens_rec) < bert_tokens_rec_ind + span:
                return None
        ELib.PASS()

    @staticmethod
    def __align_tokens(bert_tokens_rec, tw_tokens):
        result = list()
        debug_tweet = ''
        debug_tweet_main = ''
        bert_tokens_rec_ind = 0
        for cur_tw_tok_ind, cur_tw_tok in enumerate(tw_tokens):
            try:
                span, debug_phrase = ETokenAligner.__align_tokens_in_tweet(
                    cur_tw_tok.Text, cur_tw_tok.POS, bert_tokens_rec, bert_tokens_rec_ind)
            except:
                sys.exit('out of span index in Tweet')
            result.append(bert_tokens_rec[bert_tokens_rec_ind][1])
            bert_tokens_rec_ind += span
            debug_tweet += debug_phrase + ' '
            debug_tweet_main += (cur_tw_tok.Text if cur_tw_tok.POS != 'U' else 'www') + ' '
        if debug_tweet != debug_tweet_main:
            print('tokenization mismatch : ' + debug_tweet_main + '   |||   ' + debug_tweet)
        return result

    @staticmethod
    def align(bert_tokens, tw_tokens):
        bert_tokens_rec = ETokenAligner.__reconstruct_bert_tokens(bert_tokens)
        result = ETokenAligner.__align_tokens(bert_tokens_rec, tw_tokens)
        return result





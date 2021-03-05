import copy
import os
import shutil
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertForSequenceClassification

from self_pretraining.src.ELib import ELib
from self_pretraining.src.EVar import EVar


class EClassifier(nn.Module):

    def __init__(self):
        super(EClassifier, self).__init__()
        self.logs = dict()
        self.train_state = 0
        self.epoch_index = -1
        self.train_step = -1
        self.hooked_modules = dict()
        self.hook_interval = None
        self.hook_activated = None
        self.hooksForward = list()
        self.hooksBackward = list()
        ELib.PASS()

    def setup_logs(self, dir_path, curve_names, add_hooks=False, hook_interval=10):
        # one summary_writer will be created for each name in curve_names
        # also if add_hooks=True one summary_writer will be also created
        # for each module in self.hooked_modules
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        for cur_name in curve_names:
            self.logs[cur_name] = SummaryWriter(os.path.join(dir_path, cur_name))
        if add_hooks:
            self.hook_interval = hook_interval
            self.hook_activated = True
            for name, module in self.hooked_modules.items():
                self.logs[name] = SummaryWriter(os.path.join(dir_path, name))
                self.hooksForward.append(module.register_forward_hook(self.__hook_forward))
                self.hooksBackward.append(module.register_backward_hook(self.__hook_backward))
        ELib.PASS()

    def __get_module_name(self, module):
        for name, cur_module in self.hooked_modules.items():
            if cur_module is module:
                return name
        return None

    def __hook_forward(self, module, input, output):
        if self.hook_activated and self.train_step % self.hook_interval == 0:
            name = self.__get_module_name(module)
            self.logs[name].add_histogram('activations', output.to('cpu').detach().numpy(), self.train_step)
            self.logs[name].add_histogram('weights', module.weight.data.to('cpu').detach().numpy(), self.train_step)
            self.logs[name].add_histogram('bias', module.bias.data.to('cpu').detach().numpy(), self.train_step)
        ELib.PASS()

    def __hook_backward(self, module, grad_input, grad_output):
        if self.hook_activated and self.train_step % self.hook_interval == 0:
            name = self.__get_module_name(module)
            self.logs[name].add_histogram('out-gradients', grad_output[0].to('cpu').detach().numpy(), self.train_step)
            self.logs[name].add_histogram('bias-gradients', grad_input[0].to('cpu').detach().numpy(), self.train_step)
        ELib.PASS()

    def close_logs(self):
        for cur_name, cur_log in self.logs.items():
            cur_log.close()
        for cur_h in self.hooksForward:
            cur_h.remove()
        for cur_h in self.hooksBackward:
            cur_h.remove()
        ELib.PASS()


class EBertClassifier(EClassifier):

    @staticmethod
    def create(class_type, training_object, config, **kwargs):
        result = class_type(config, **kwargs)
        try:
            print('loading modified pre-trained model..', end='. ', flush=True)
            state_dict = os.path.join(config.model_path, 'pytorch_model.bin')
            result.load_state_dict(torch.load(state_dict))
            print('loaded from ' + config.model_path, flush=True)
        except Exception as e:
            print('failed', flush=True)
            EBertClassifier.load_pretrained_bert_modules(result.__dict__['_modules'], config)
        # self.bert_classifier = BertForSequenceClassification.from_pretrained(
        #     self.config.model_path, config=self.config.bert_config)
        result._add_bert_hooks()
        result.training_object = training_object
        return result

    @staticmethod
    def load_pretrained_bert_modules(modules, config):
        mod_list = modules
        if type(modules) is not OrderedDict:
            mod_list = OrderedDict([('reconfig', modules)])
        loaded = set()
        for cur_module in mod_list.items():
            if type(cur_module[1]) is BertModel or isinstance(cur_module[1], EBertModelWrapper):
                print('{}: '.format(cur_module[0]), end='', flush=True)
                EBertClassifier.__load_pretrained_bert_module(cur_module[1], config, loaded)
            elif type(cur_module[1]) is nn.ModuleList:
                for c_ind, cur_child_module in enumerate(cur_module[1]):
                    if type(cur_child_module) is BertModel or isinstance(cur_child_module, EBertModelWrapper):
                        print('{}[{}]: '.format(cur_module[0], c_ind), end='', flush=True)
                        EBertClassifier.__load_pretrained_bert_module(cur_child_module, config, loaded)
        ELib.PASS()

    @staticmethod
    def __load_pretrained_bert_module(module, config, loaded):
        if type(module) is BertModel:
            if module not in loaded:
                module.load_state_dict(EBertClassifier.__load_pretrained_bert_layer(config).state_dict())
                loaded.add(module)
            else:
                print('already loaded')
            ELib.PASS()
        elif isinstance(module, EBertModelWrapper):
            if module not in loaded:
                module.bert_layer.load_state_dict(EBertClassifier.__load_pretrained_bert_layer(config).state_dict())
                loaded.add(module)
            else:
                print('already loaded')
            ELib.PASS()
        else:
            print(colored('unknown bert module to load', 'red'))

    @staticmethod
    def __load_pretrained_bert_layer(config):
        print('loading default model..', end='. ', flush=True)
        result = BertModel.from_pretrained(config.model_path, config=config.bert_config)
        print('loaded from ' + config.model_path, flush=True)
        return result

    def __init__(self):
        super(EBertClassifier, self).__init__()
        self.output_vecs = None
        self.output_vecs_detail = None
        self.training_object = None
        ELib.PASS()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.bert_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _add_bert_hooks(self):
        if len(self.hooked_modules) > 0:
            raise Exception('You have added hooks to EBertClassifier! \n'
                            'BERT was loaded from a file and hooks might have been overwritten! \n'
                            'override "_add_bert_hooks()" and add all of the hooks there!')

    def freeze_modules(self, m_list):
        for cur_m in m_list:
            self.training_object.remove_module_from_optimizer(cur_m)
        ELib.PASS()

    def unfreeze_modules(self, m_list):
        for cur_m in m_list:
            self.training_object.add_module_to_optimizer(cur_m)
        ELib.PASS()

    def set_modules_learning_rate(self, m_list, lr):
        for cur_m in m_list:
            self.training_object.set_module_learning_rate(cur_m, lr)
        ELib.PASS()


class EBertModelWrapper(EBertClassifier):

    def __init__(self, bert_config):
        super(EBertModelWrapper, self).__init__()
        self.bert_layer = BertModel(bert_config)

    def __format__(self, format_spec):
        ELib.PASS()


class EBertClassifierSimple(EBertClassifier):

    def __init__(self, config):
        super(EBertClassifierSimple, self).__init__()
        self.config = config
        self.bert_layer = BertModel(self.config.bert_config)
        self.last_dropout_layer = nn.Dropout(self.config.dropout_prob)
        self.last_layer = nn.Linear(self.config.bert_config.hidden_size, self.config.label_count)
        self.apply(self._init_weights)
        ELib.PASS()

    def forward(self, input_batch, apply_softmax):
        b_output = self.bert_layer(input_batch['x'],
                                 attention_mask=input_batch['mask'],
                                 token_type_ids=input_batch['type'])
        last_hidden_states = b_output[0]
        output_pooled = b_output[1]

        self.output_vecs = np.copy(output_pooled.detach().to('cpu').numpy()).tolist()
        self.output_vecs_detail = np.copy(last_hidden_states.detach().to('cpu').numpy()).tolist()
        for cur_seq_ind, cur_seq_len in enumerate(input_batch['len']):
            self.output_vecs_detail[cur_seq_ind] = self.output_vecs_detail[cur_seq_ind][:cur_seq_len]
            ELib.PASS()

        output_pooled = self.last_dropout_layer(output_pooled)
        logits = self.last_layer(output_pooled)
        if apply_softmax:
            logits = F.softmax(logits, dim = 1)
        return [(EVar.DefaultTask, logits)] # or [(input_batch['task_list'][0][0], logits)]


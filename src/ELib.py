import os
import re
from statistics import mode
from nltk.tokenize import TweetTokenizer
from django.core.validators import URLValidator
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import unidecode
import traceback
from termcolor import colored

import torch
import torch.nn.functional as F
from sklearn import metrics
import warnings

class ELib:
    _dels = " ['\"“”‘’]+|[.?!,…]+|[:;]+(?:--+|―|—|~|–|=)@^*&%$#{}<>/\\~"
    tokenizer = TweetTokenizer()
    url_checker = URLValidator()
    __rnd_print_counter = 0

    @staticmethod
    def __out(force, text, end):
        printAll = True
        if printAll or force:
            print(text, end=end)

    @staticmethod
    def outLine(line):
        ELib.__out(False, line, "\n")

    @staticmethod
    def outLineForce(line):
        ELib.__out(True, line, "\n")

    @staticmethod
    def out(text):
        ELib.__out(False, text, "")

    @staticmethod
    def outForce(text):
        ELib.__out(True, text, "")

    @staticmethod
    def normalizeTime(text):
        if not str.isdigit(text[0]):
            # tokens = text.split(" ")
            # result = tokens[3] + " - " + tokens[2] + " " + tokens[1] + " " + tokens[5]
            # return result
            return text
        else:
            return text

    @staticmethod
    def get_time(only_time=False):
        if only_time:
            return datetime.now().strftime('%H:%M:%S')
        return datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    @staticmethod
    def  is_delimiter(str):
        for dl in ELib._dels:
            if str == dl:
                return True
        return False

    @staticmethod
    def PASS():
        pass

    @staticmethod
    def progress_made(current_index, max_value_exclusive, number_of_intervals):
        current_number = current_index + 1
        result_1 = (current_number == 1)
        result_2 = int((number_of_intervals * current_number) / max_value_exclusive) !=\
                   int((number_of_intervals * (current_number - 1)) / max_value_exclusive)
        result = result_1 or result_2
        return result

    @staticmethod
    def progress_percent(current_index, max_value_exclusive):
        current_number = current_index + 1
        result = ((10.0 * current_number) / max_value_exclusive) * 10
        return str(int(result)) + '%'

    @staticmethod
    def __is_number(param):
        try:
            float(param)
        except ValueError:
            return False
        return True

    @staticmethod
    def tokenize_tweet_text(tw, normalize, tokenize_by_etokens=False,
                            pivot_query=None, query_list=None):
        if tokenize_by_etokens:
            tokens = list()
            for to_ind, tok in enumerate(tw.ETokens):
                temp = unidecode.unidecode(tok.Text).lower().replace(' ', '-')
                if temp != '':
                    tok.Text = temp
                    tokens.append(temp)
                else:
                    tok.Text = '.'
                    tokens.append('.')
        else:
            tokens = ELib.tokenizer.tokenize(tw.Text)
        for to_ind, cur_tok in enumerate(tokens):
            if pivot_query is not None:
                if cur_tok.lower() in query_list:
                    tokens[to_ind] = pivot_query
            if normalize:
                if re.match(ELib.url_checker.regex, cur_tok) is not None:
                    tokens[to_ind] = 'www'
                # elif cur_tok.startswith('@'):
                #     tokens[to_ind] = '@user'
                # elif ELib.__is_number(cur_tok):
                #     tokens[to_ind] = 'number'
        result = (' '.join(tokens)).lower()
        return result

    @staticmethod
    def draw_activation_curves(curves, names, start_x=0):
        # draw_curve([
        #           [[ndarray], [ndarray], [ndarray]],
        #           [[ndarray], [ndarray], [ndarray]]
        #           ],
        #           ['c1',
        #           'c2'])
        # scenario: show the output activations of multiple layers across multiple epochs

        fig = plt.figure()
        for s_ind, cur_series in enumerate(curves):
            x = list()
            means = list()
            vars = list()
            for t_ind, cur_t in enumerate(cur_series):
                x.append(start_x + t_ind)
                means.append(np.mean(cur_t))
                vars.append(np.std(cur_t))
            plt.errorbar(x=x, y=means, yerr=vars, label=names[s_ind])
            assert 3==2+1
        plt.legend(fontsize=20, loc='lower right')
        plt.show()
        assert 3==2+1

    @staticmethod
    def draw_gradient_curves(curves, names):
        # draw_curve([
        #           [ndarray],
        #           [ndarray]
        #           ],
        #           ['c1',
        #           'c2'])
        # scenario 1: show the gradient distributions of one specific layer across multiple epochs
        # scenario 2: show the gradient distributions of multiple layers in one specific epoch

        fig = plt.figure()
        for s_ind, cur_series in enumerate(curves):
            y, binEdges = np.histogram(cur_series, bins=100, normed=True)
            bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            plt.plot(bincenters, y, '-', label=names[s_ind])
            assert 3==2+1
        plt.legend(fontsize=20, loc='upper right')
        plt.show()
        assert 3==2+1

    @staticmethod
    def average_lists_elementwise(param_list):
        param_list_np = np.array(param_list)
        result = np.sum(param_list_np, axis=0)
        result = result / len(param_list)
        return result.tolist()

    @staticmethod
    def print_randoms():
        ELib.__rnd_print_counter += 1
        print('randoms [' + str(ELib.__rnd_print_counter) + '] > ',
              '| python:' + str(random.random()),
              '| numpy:' + str(np.random.random()),
              '| torch:' + str(torch.rand(1, 1).item()))

    @staticmethod
    def averaged_tempered_softmax(np_arr, temperature, weights=None, do_softmax=True):
        if not isinstance(np_arr, np.ndarray):
            np_arr = np.array(np_arr)
        ## calculate softmax over tables
        dump = list()
        if do_softmax:
            for cur_pred in np_arr:
                cur_pred_ts = torch.tensor(cur_pred)
                if len(cur_pred.shape) == 1:
                    sft = F.softmax(cur_pred_ts / temperature, dim=0).numpy().tolist()
                else:
                    sft = F.softmax(cur_pred_ts / temperature, dim=1).numpy()
                dump.append(sft)
        else:
            dump = np_arr
        if len(np_arr.shape) == 2:
            return dump
        ## prepare weights
        if weights is None:
            weights = [1.0 for _ in np_arr]
        sum_weight = sum(weights)
        ## take average
        result = np.array(dump[0]) * weights[0]
        for ind in range(1, len(dump)):
            result = result + dump[ind] * weights[ind]
        result = result / sum_weight
        result = result.tolist()
        return result

    @staticmethod
    def calculate_metrics(truths, preds):
        # sklearn turns on warnings after we import it, call this to make sure it is silent
        warnings.filterwarnings('ignore')
        F1 = metrics.f1_score(truths, preds)
        Pre = metrics.precision_score(truths, preds)
        Rec = metrics.recall_score(truths, preds)
        Acc = metrics.accuracy_score(truths, preds)
        return [F1, Pre, Rec, Acc]

    @staticmethod
    def calculate_f1(truths, preds):
        # sklearn turns on warnings after we import it, call this to make sute it is silent
        warnings.filterwarnings('ignore')
        if 0 <= min(truths) and max(truths) <= 1 and 0 <= min(preds) and max(preds) <= 1:
            F1 = metrics.f1_score(truths, preds)
        else:
            F1 = metrics.f1_score(truths, preds, average='weighted')
        return F1

    @staticmethod
    def get_string_metrics(metrics):
        return 'F1: {:.3f} Pre: {:.3f} Rec: {:.3f} Acc: {:.3f}'.format(metrics[0], metrics[1], metrics[2], metrics[3])

    @staticmethod
    def majority_voting(lbl_list):
        result = list()
        for entry_ind, _ in enumerate(lbl_list[0]):
            cur_lbls = [run[entry_ind] for run in lbl_list]
            try:
                cur_l = mode(cur_lbls)
            except:
                lbl_count = list()
                for cur_lll in list(set(cur_lbls)):
                    lbl_count.append([cur_lll, sum(run.count(cur_lll) for run in lbl_list)])
                lbl_count.sort(key=lambda comp: comp[1])
                cur_l = lbl_count[0][0]
            result.append(cur_l)
        return result

    @staticmethod
    def majority_logits(logit_list, weights=None):
        probs = ELib.averaged_tempered_softmax(logit_list, 1, weights=weights)
        result = list()
        for cur_prob in probs:
            lbl = cur_prob.index(max(cur_prob))
            result.append(lbl)
        return result

    @staticmethod
    def CrossEntropyLossWithSoftLabels(pred_tensor, true_tensor, temperature=1.0, mean=True):
        pred_softmaxed = F.log_softmax(pred_tensor / temperature, dim=1)
        loss = -(true_tensor * pred_softmaxed).sum(dim=1)
        if mean:
            loss = loss.mean()
        return loss

    @staticmethod
    def proxy_train(cls, *args, **kwargs):
        try:
            cls.train(*args, **kwargs)
        except Exception as e:
            print(colored('Exception occurred in the training thread, model_id:{}>'.format(cls.model_id),
                          'green'))
            print(colored(traceback.format_exc(), 'red'))
            print(colored('__________________________________________________________', 'red'))
            os._exit(1)

    @staticmethod
    def print_gpu_content(device):
        if torch.cuda.is_available():
            summary = torch.cuda.memory_summary(device)
            print(colored('----------------------------------------------', 'red'))
            print(summary)
            print(colored('----------------------------------------------', 'red'))
        ELib.PASS()

    @staticmethod
    def get_formatted_float_list(arg_list):
        if type(arg_list[0]) is not list:
            result = '[' + ','.join(['{:.3f}'.format(entry) for entry in arg_list]) + ']'
        else:
            result = '[\n'
            for cur_row in arg_list:
                result += ' [' + ','.join(['{:.3f}'.format(entry) for entry in cur_row]) + ']\n'
            result += ']'
        return result

    @staticmethod
    def save_histogram(title, series_list, label_list, bins, output_dir, file_name):
        fig, ax = plt.subplots()
        ax.hist(series_list, label=label_list, bins=bins)
        ax.legend(fontsize=10, loc='best')
        plt.title(title, fontsize=20)
        plt.savefig(output_dir + file_name) # save should be before show() !
        # plt.show()
        ELib.PASS()

    @staticmethod
    def one_hot(lbl_list, cls_count):
        result = np.eye(cls_count)[lbl_list]
        return result.astype(np.int)

    @staticmethod
    def logit_to_label(logit_list):
        result = np.argmax(logit_list, axis=1).tolist()
        return result

    @staticmethod
    def logit_to_prob(logit_list):
        result = F.softmax(torch.tensor(logit_list), dim=1).numpy().tolist()
        return result


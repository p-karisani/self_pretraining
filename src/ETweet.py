import sys
from math import ceil

from termcolor import colored

from self_pretraining.src.ELib import ELib
from self_pretraining.src.EFeat1Gram import EFeat1Gram
import io
import random


class ELoadType:
    none = 0
    pos_tagger = 1
    stored_tags = 2
    stored_tags_all_DontUseThis = 3
    stored_injected_tags = 4
    stored_injected_tags_all_DontUseThis = 5


class ETweetRepo:

    def __init__(self):
        self.q_vec = None
        self.tw_vec = None
        self.first_occur_q = None
        ELib.PASS()


class ETweet:

    useridAlign = 25
    tokenDummyQuery = 'DUMMY'

    def __initValues(self):
        self.Repo = ETweetRepo()
        self.AAAText = ""
        self.Tweetid = ""
        self.Label = 0
        self.Userid = ""
        self.Time = ""
        self.ReplyCount = 0
        self.LikeCount = 0
        self.RetweetCount = 0
        self.Query = ""
        self.QueryList = None
        self.Text = ""
        # self.Text.endswith("#sarcasm")

        self.NewLabel = 0
        self.ETokens = []

    def __init__(self, line=None):
        self.__initValues()
        if line is not None:
            tokens = line.split("\t")
            self.__setFields(tokens, 0)

    def __removeSarcasm(self, text):
        if text.endswith("#sarcasm"):
            return text[:-1 * len("#sarcasm")]
        else:
            return text

    def __setFields(self, tokens, startIndex):
        ind = startIndex
        self.Tweetid = tokens[ind]
        ind += 1
        self.Label = int(tokens[ind])
        ind += 1
        self.Userid = tokens[ind][:ETweet.useridAlign].strip()
        self.Time = tokens[ind][ETweet.useridAlign:].strip()
        ind += 1
        self.ReplyCount = int(tokens[ind])
        ind += 1
        self.LikeCount = int(tokens[ind])
        ind += 1
        self.RetweetCount = int(tokens[ind])
        ind += 1
        self.Query = tokens[ind]
        ETweet.configureQuery(self)
        ind += 1
        text = tokens[ind]
        self.AAAText = text
        self.Text = self.__removeSarcasm(text)
        ind += 1
        return ind

    def __str__(self):
        result = ""
        result += self.Tweetid
        result += "\t"
        result += str(self.Label)
        result += "\t"
        result += self.Userid.ljust(ETweet.useridAlign, " ")
        tm = ELib.normalizeTime(self.Time)
        result += tm
        result += "\t"
        result += str(self.ReplyCount)
        result += "\t"
        result += str(self.LikeCount)
        result += "\t"
        result += str(self.RetweetCount)
        result += "\t"
        if self.Query != ETweet.tokenDummyQuery:
            result += self.Query
        else:
            if len(self.QueryList) > 0:
                for q in self.QueryList:
                    result += '|' + q
            else:
                result += '|'
        result += "\t"
        result += self.Text
        return result.strip()

    def to_token_based_string(self):
        result = ''
        for tok in self.ETokens:
            result += tok.Text \
                      + ('[H]' if tok.IsHuman else '') \
                      + ('[P]' if tok.IsPositiveHuman else '') \
                      + ('[S]' if tok.IsSynthesized else '') \
                      + ' '
        return result

    @staticmethod
    def configureQuery(tw):
        if tw.Query[0] == '|':
            tw.QueryList = tw.Query.split('|')
            tw.QueryList = [x for x in tw.QueryList if x]
            tw.Query = ETweet.tokenDummyQuery
            pass
        else:
            tw.QueryList = [tw.Query]
            pass
        pass

    @staticmethod
    def __text_to_tweet_object(tokens):
        if len(tokens) != 4:
            raise ValueError('cannot parse the input file, I expect: doc-id <TAB> label <TAB> domain <TAB> text')
        tw = ETweet()
        tw.Tweetid = tokens[0]
        tw.Label = int(tokens[1])
        tw.Query = tokens[2]
        ETweet.configureQuery(tw)
        tw.Text = tokens[3]
        tw.AAAText = tokens[3]
        return tw

    @staticmethod
    def load(filePath, load_type, tweet_file=True):
        try:
            if type(load_type) == bool:
                print('change param to LoadType')
                exit(1)
            result = []
            file = open(filePath, "r", encoding="utf-8")
            lines = file.readlines()
            file.close()
            if load_type == ELoadType.pos_tagger:
                pass
            else:
                if tweet_file:
                    for ind, line in enumerate(lines):
                        tw = ETweet(line)
                        result.append(tw)
                        if (ind + 1) % 1000000 == 0:
                            print((ind + 1), ' reading lines')
                    if (ind + 1) > 1000000 and ELoadType.none != load_type:
                        print('reading tokens')
                    tags = None
                    if load_type == ELoadType.stored_tags or \
                            load_type == ELoadType.stored_tags_all_DontUseThis:
                        tags = EFeat1Gram.read_dep_tags(filePath + "-tags")
                    elif load_type == ELoadType.stored_injected_tags or \
                            load_type == ELoadType.stored_injected_tags_all_DontUseThis:
                        tags = EFeat1Gram.read_dep_tags(filePath + "-tags-synthesized")
                    for ind, tw in enumerate(result):
                        if load_type == ELoadType.stored_tags_all_DontUseThis or \
                                load_type == ELoadType.stored_injected_tags_all_DontUseThis:
                            tw.ETokens = EFeat1Gram.convert_all_tags_to_tokens(tags[ind])
                        elif load_type == ELoadType.stored_tags or \
                                load_type == ELoadType.stored_injected_tags:
                            tw.ETokens = EFeat1Gram.convert_tags_to_tokens(tags[ind])
                        if (ind + 1) % 500000 == 0 and ELoadType.none != load_type:
                            print((ind + 1), ' constructing tokens')
                else:
                    for ind, line in enumerate(lines):
                        tokens = line.strip().split('\t')
                        tw = ETweet.__text_to_tweet_object(tokens)
                        result.append(tw)
                    ELib.PASS()
        except Exception as err:
            print(colored('Error in loading: "{}"\n\n{}'.format(filePath, str(err)), 'red'))
            sys.exit(1)
        return result

    @staticmethod
    def save(tws, filePath):
        with io.open(filePath, 'w', encoding='utf-8') as ptr:
            for cur_tw in tws:
                ptr.write(str(cur_tw) + '\n')

    @staticmethod
    def save_tweets_as_text_file(start_id, tws, file_path):
        result = ''
        for cur_tw in tws:
            if cur_tw.Query != ETweet.tokenDummyQuery:
                query = cur_tw.Query
            else:
                query = ''
                if len(cur_tw.QueryList) > 0:
                    for q in cur_tw.QueryList:
                        query += '|' + q
                else:
                    query += '|'
            result += '{}\t{}\t{}\t{}\n'.format(str(start_id).zfill(7), str(cur_tw.Label), query, cur_tw.Text.strip())
            start_id += 1
        with io.open(file_path, 'w', encoding='utf-8') as ptr:
                ptr.write(result)
        ELib.PASS()

    @staticmethod
    def split_by_query(tws):
        result_dict = dict()
        for cur_tw in tws:
            if cur_tw.Query not in result_dict:
                result_dict[cur_tw.Query] = list()
            result_dict[cur_tw.Query].append(cur_tw)
        dict_items = list(result_dict.items())
        dict_items.sort(key=lambda entry: entry[0])
        result_list = [entry[1] for entry in dict_items]
        return result_list

    @staticmethod
    def split_by_first_query_in_tweet(tws):
        result_dict = dict()
        for cur_tw in tws:
            for cur_tok in cur_tw.ETokens:
                q_ind = 0
                for cur_q in cur_tw.QueryList:
                    if cur_tok.Text.find(cur_q) >= 0:
                        break
                    q_ind += 1
                if q_ind < len(cur_tw.QueryList):
                    if cur_q not in result_dict:
                        result_dict[cur_q] = list()
                    result_dict[cur_q].append(cur_tw)
                    cur_tw.Repo.first_occur_q = cur_q
                    break
        dict_items = list(result_dict.items())
        dict_items.sort(key=lambda entry: -1 * len(entry[1]))
        return dict_items

    @staticmethod
    def filter_tweets_by_label(tws, lbl):
        result = list()
        for cur_tw in tws:
            if cur_tw.Label == lbl:
                result.append(cur_tw)
        return result

    @staticmethod
    def filter_tweets_by_correct_label(tws, lc, correct_lbl):
        result = list()
        for cur_tw in tws:
            if lc.get_correct_new_label(cur_tw.Label) == correct_lbl:
                result.append(cur_tw)
        return result

    @staticmethod
    def get_queries(tws):
        result_dict = set()
        for cur_tw in tws:
            if cur_tw.Query not in result_dict:
                result_dict.add(cur_tw.Query)
        result = list(result_dict)
        result.sort()
        return result

    @staticmethod
    def filter_by_query(tws, query):
        result = list()
        for cur_tw in tws:
            if cur_tw.Query == query:
                result.append(cur_tw)
        return result

    @staticmethod
    def random_stratified_sample(tws, lc, ratio, seed, with_replacement=False):
        random.seed(seed)
        neg = ETweet.filter_tweets_by_correct_label(tws, lc, lc.negative_new_label.new_label)
        if with_replacement:
            result_neg = random.choices(neg, k=ceil(len(neg) * ratio))
        else:
            result_neg = random.sample(neg, k=ceil(len(neg) * ratio))
        pos = ETweet.filter_tweets_by_correct_label(tws, lc, lc.positive_new_label.new_label)
        if with_replacement:
            result_pos = random.choices(pos, k=ceil(len(pos) * ratio))
        else:
            result_pos = random.sample(pos, k=ceil(len(pos) * ratio))
        result = list()
        result.extend(result_neg)
        result.extend(result_pos)
        random.shuffle(result)
        return result

    @staticmethod
    def filter_by_tweets(tws_all, tws_exclude):
        exclude_set = set()
        for cur_tw in tws_exclude:
            exclude_set.add(cur_tw.Tweetid)
        result = list()
        for cur_tw in tws_all:
            if cur_tw.Tweetid not in exclude_set:
                result.append(cur_tw)
        return result



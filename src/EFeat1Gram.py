from self_pretraining.src.EToken import EToken
from self_pretraining.src.ELib import ELib

class EFeat1Gram:

    @staticmethod
    def read_dep_tags(file_path):
        file = open(file_path, "r")
        lines = file.readlines()
        file.close()
        result = []
        cur = None
        for ind in range(len(lines)):
            cur_line = lines[ind].strip()
            if cur == None or (cur_line == "" and ind + 1 < len(lines)):
                cur = []
                result.append(cur)
            if cur_line != "":
                cur.append(cur_line)
        return result

    @staticmethod
    def convert_lines_to_tokens(dep_tags):
        c_result = []
        result_dict = dict()
        for ind in range(len(dep_tags)):
            tok = EToken()
            c_result.append(tok)
        for ind, dep_line in enumerate(dep_tags):
            cols = dep_line.split("\t")
            et = c_result[ind]
            et.Order = int(cols[0])
            et.Text = cols[1].lower()
            et.POS = cols[4]
            if len(cols) > 8:
                et.IsSynthesized = True if cols[8] == 'new' else False
            if len(cols) > 9:
                et.IsHuman = True if cols[9] == 'H' else False
            if len(cols) > 10:
                et.IsPositiveHuman = True if cols[10] == 'PH' else False
            et.Weight= 1
            et.RootValue = int(cols[6])
            result_dict[et.Order] = et
        for et in c_result:
            if et.RootValue != -1 and et.RootValue != 0 and et.RootValue in result_dict:
                et.Root = result_dict[et.RootValue]
        return c_result

    @staticmethod
    def convert_all_tags_to_tokens(dep_tags):
        c_result = EFeat1Gram.convert_lines_to_tokens(dep_tags)
        EFeat1Gram.build_tweet_trees(c_result)
        return c_result

    @staticmethod
    def convert_tags_to_tokens(dep_tags):
        c_result = EFeat1Gram.convert_lines_to_tokens(dep_tags)
        result = []
        for tok in c_result:
            if not ELib.is_delimiter(tok.Text):
                result.append(tok)
                if tok.Text[0] == '#':
                    tok.Text = tok.Text[1:]
        EFeat1Gram.build_tweet_trees(result)
        return result

    @staticmethod
    def build_tweet_trees(toks):
        for ind_pat, pat in enumerate(toks):
            if pat.RootValue != -1:
                for ind_cht, cht in enumerate(toks):
                    if pat.Order == cht.RootValue:
                        pat.Children.append(cht)


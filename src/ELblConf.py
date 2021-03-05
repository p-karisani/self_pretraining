from self_pretraining.src.ELib import ELib

class ELblConf:

    def __init__(self, negative_new_label, positive_new_label, labels):
        self.negative_new_label = None
        self.positive_new_label = None
        self.labels = labels
        for cur_lbl in labels:
            if negative_new_label == cur_lbl.new_label:
                self.negative_new_label = cur_lbl
                break
        for cur_lbl in labels:
            if positive_new_label == cur_lbl.new_label:
                self.positive_new_label= cur_lbl
                break

    def get_correct_new_label(self, lbl):
        for cur_lbl in self.labels:
            if cur_lbl.is_source(lbl):
                return cur_lbl.new_label
        ELib.out("Unknown label to map!")
        return -10

    def get_sample_label_from_new_label(self, new_label):
        for cur_lbl in self.labels:
            if new_label == cur_lbl.new_label:
                return cur_lbl.source_lbls[0]
        ELib.out("Unknown NewLabel to map!")
        return -10

    def get_new_label_count(self):
        return len(self.labels)


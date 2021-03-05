
class ELbl:

    def __init__(self, new_label, source_lbls):
        self.new_label = new_label
        self.source_lbls = source_lbls

    def is_source(self, lbl):
        for cur_lbl in self.source_lbls:
            if cur_lbl == lbl:
                return True
        return False




class EToken:

    def __initValues(self):
        self.Order = 0
        self.Text = ""
        self.POS = ""
        self.Weight = 0.0
        self.Root = None
        self.RootValue = 0
        self.Children = []
        self.IsSynthesized = False
        self.IsHuman = False
        self.IsPositiveHuman = False
        self.IsVecLess = False

    def __init__(self):
        self.__initValues()

    def __str__(self):
        result = ""
        if self.Root is not None:
            result = self.Text + " [" + self.POS + ", " + \
                     str(len(self.Children)) + ", " + self.Root.Text + "]"
        else:
            result = self.Text + " [" + self.POS + ", " + \
                     str(len(self.Children)) + "]"
        return result

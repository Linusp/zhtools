class Token():

    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end

    def __str__(self):
        return self.word

    def __repr__(self):
        return f'<Token {self.word}({self.start}:{self.end})>'

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False

        return self.__dict__ == other.__dict__

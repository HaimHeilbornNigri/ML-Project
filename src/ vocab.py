from preprocessing import tokenize

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build(self, texts):
        for text in texts:
            for word in tokenize(text):
                self.add_word(word)

    def encode(self, text):
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in tokenize(text)
        ]

    def __len__(self):
        return len(self.word2idx)
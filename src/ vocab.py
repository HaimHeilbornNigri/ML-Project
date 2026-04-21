from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq

    def build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, tokens):
        return [self.word2idx.get(t, 1) for t in tokens]
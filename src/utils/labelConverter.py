import torch

class CTCLabelConverter(object):
    def __init__(self, cfg):
        self.device = cfg.device
        self.character = cfg.dict_
        self.batch_max_length = cfg.max_str_len

        dict_character = list(self.character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character

    def encode(self, text):
        length = [len(s) for s in text]

        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(self.device), torch.IntTensor(length).to(self.device))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts
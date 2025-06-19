# https://github.com/songlab-cal/gpn/blob/main/gpn/data.py
# tokenizer for GPN-MSA
import numpy as np

class Tokenizer(object):
    def __init__(self, vocab="-ACGT?"):
        # -: gap/unknown/pad (simple for now, could split in the future)
        # ?: mask
        unk = vocab.index("-")
        self.table = np.full((256,), unk, dtype=np.uint8)
        for i, c in enumerate(vocab):
            self.table[ord(c)] = i
        self.vocab = vocab
        self.mask_token = "?"
        self.pad_token = "-"

    def __call__(self, x):
        return self.table[np.char.upper(x).view(np.uint8)]

    def __len__(self):
        return len(self.vocab)

    def mask_token_id(self):
        return self.vocab.index("?")

    def unk_token_id(self):
        return self.vocab.index("-")

    def pad_token_id(self):
        return self.vocab.index("-")

    def nucleotide_token_id_start(self):
        return self.vocab.index("A")

    def nucleotide_token_id_end(self):
        return self.vocab.index("T") + 1
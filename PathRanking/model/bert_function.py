import unicodedata

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def judge_ignore(word):
    if len(_clean_text(word)) == 0:
        return True
    for char in word:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            return True
    return False

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

class Vocab(object):
    def __init__(self, bert_vocab_path):
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_vocab_path, do_lower_case=False
        )

    def convert_tokens_to_ids(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor(token_ids, dtype=torch.long)
        mask = torch.ones(len(ids), dtype=torch.long)
        return ids, mask

    def subword_tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        token_start_idxs = torch.cumsum(torch.tensor([0] + subword_lengths[:-1]), dim=0)
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        tokens = ["[PAD]" if judge_ignore(t) else t for t in tokens]
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(len(subword_ids), dtype=torch.uint8)
        token_starts[token_start_idxs] = 1
        return subword_ids, mask, token_starts

    def tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        return subwords

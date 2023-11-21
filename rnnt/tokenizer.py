from typing import Union
from os import PathLike
import json


class JSONLoader:
    def __init__(self, file_path: Union[str, PathLike]) -> None:
        self.file_path = file_path

    def load(self):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        return data


class CharTokenizer:
    _oov_key = "oov"
    _sos_key = "sos"
    _eos_key = "eos"
    _pad_key = "pad"
    _phi_key = "phi"
    _token_to_id_key = "token_to_id"
    _special_tokens_key = "special_tokens"

    def __init__(self):
        self._token_to_id = dict()
        self._id_to_token = dict()
        self.special_tokens = dict()

    def vocab_size(self):
        return len(self._token_to_id)

    def load_tokenizer(self, tokenizer_path):
        data = JSONLoader(tokenizer_path).load()
        self._token_to_id = data[self._token_to_id_key]
        self.special_tokens = data[self._special_tokens_key]
        self._id_to_token = {value: key for key, value in self._token_to_id.items()}
        return self

    def ids2tokens(self, ids):
        tokens = []

        # Iterate over rows and columns
        for row_ids in ids:
            row_tokens = [self._id_to_token[id] for id in row_ids]
            tokens.append(row_tokens)

        return tokens

    def tokens2ids(self, sentence):
        return [self._token_to_id[token] for token in sentence]

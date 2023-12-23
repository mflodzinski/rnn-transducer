import math
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Union, Tuple
from torchaudio.transforms import Resample
from torchaudio.compliance.kaldi import mfcc        

import torchaudio
import numpy as np


class BaseDataLoader:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.idx = 0


    def _get_padded_tokens(self, text: str, max_len: int) -> Tensor:
        pad_idx = self.tokenizer.special_tokens["pad"][1]
        sos_idx = self.tokenizer.special_tokens["sos"][1]
        eos_idx = self.tokenizer.special_tokens["eos"][1]

        tokens = self.tokenizer.tokens2ids(text)
        num_tokens = len(tokens)
        num_pad = max_len - num_tokens

        tokens = [sos_idx] + tokens + [eos_idx]
        tokens = tokens + [pad_idx] * num_pad

        return torch.IntTensor(tokens)

    def prepare_audio(self, audio_path: Union[str, Path]) -> Tensor:
        x, sr = torchaudio.load(audio_path, normalize=True)
        x = Resample(sr, self.config.sample_rate)(x)
        x = mfcc(x)
        x = torch.unsqueeze(x, 0)
        x = x.permute(0, 2, 1)
        return x

    def prepare_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace(" ", "_")
        text = "".join(char if char.isalpha() or char == "_" else "" for char in text)
        text = text.strip()
        return text




class WordDataLoader(BaseDataLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        config,
        batch_size: int,
    ) -> None:
        super().__init__(tokenizer, config)
        self.batch_size = batch_size
        self.df = pd.read_csv(file_path)
        self.num_examples = len(self.df)
        self.idx = 0

    def _get_padded_aud(
        self, aud_path: Union[str, Path], max_duration: float, word: str
    ) -> Tensor:
        max_len = 1 + math.ceil(
            (max_duration - self.config.win_size) / self.config.hop_size
        )
        first_slash_index = aud_path.find("/")
        path = (
            aud_path[: first_slash_index + 1]
            + "MFCC/"
            + aud_path[first_slash_index + 1 :]
        )
        path, ext = path.split(".")
        path = f"{path}/{word}.npy"
        aud = torch.tensor(np.load(path))

        n = max_len - aud.shape[1]
        zeros = torch.zeros(size=(1, n, aud.shape[-1]))
        return torch.cat([zeros, aud], dim=1), aud.shape[1]


    def __len__(self):
        length = self.num_examples // self.batch_size
        return length + 1 if self.num_examples % self.batch_size > 0 else length

    def get_max_duration(self, start_idx: int, end_idx: int) -> float:
        durations = (
            self.df.loc[start_idx:end_idx, "end"]
            - self.df.loc[start_idx:end_idx, "start"]
        )
        return durations.max()

    def get_max_text_length(self, start_idx: int, end_idx: int) -> int:
        texts = self.df["text"].iloc[start_idx:end_idx]
        max_length = max(len(str(text)) for text in texts)
        return max_length

    def get_audios(self, start_idx: int, end_idx: int) -> Tensor:
        max_duration = self.get_max_duration(start_idx, end_idx)
        results = [
            self._get_padded_aud(path, max_duration, word)
            for path, word in zip(
                self.df["audio_path"].iloc[start_idx:end_idx],
                self.df["text"].iloc[start_idx:end_idx],
            )
        ]
        result, lengths = [t[0] for t in results], [t[1] for t in results]
        result = torch.stack(result, dim=1)

        return torch.squeeze(result), torch.IntTensor(lengths)

    def get_texts(self, start_idx: int, end_idx: int) -> Tuple[Tensor, torch.IntTensor]:
        args = self.df["text"].iloc[start_idx:end_idx]
        lengths = [len(x) + 2 for x in args.values] # +2 because of SOS and EOS tokens
        max_len = self.get_max_text_length(start_idx, end_idx)
        result = torch.stack(
            [self._get_padded_tokens(text, max_len) for text in args], dim=0
        )
        return result.to(torch.int32), torch.IntTensor(lengths)

    def __iter__(self):
        self.idx = 0
        while self.idx * self.batch_size < self.num_examples:
            start = self.idx * self.batch_size
            end = min((self.idx + 1) * self.batch_size, self.num_examples)
            self.idx += 1
            yield *self.get_audios(start, end), *self.get_texts(start, end)





class SentenceDataLoader(BaseDataLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        config,
        batch_size: int,
    ) -> None:
        super().__init__(tokenizer, config)
        self.batch_size = batch_size
        self.df = pd.read_csv(file_path)
        self.num_examples = len(self.df)
        self.idx = 0

    def _get_padded_aud(
        self,
        aud_path: Union[str, Path],
        max_duration: float,
    ) -> Tensor:
        max_len = 1 + math.ceil(
            (max_duration * self.config.sample_rate - self.config.win_size)
            / self.config.hop_size
        )

        first_slash_index = aud_path.find("/")
        path = (
            aud_path[: first_slash_index + 1]
            + "MFCC/"
            + aud_path[first_slash_index + 1 :]
        )
        path, ext = path.split(".")
        path = f"{path}.npy"
        aud = torch.tensor(np.load(path))

        n = max_len - aud.shape[1]
        zeros = torch.zeros(size=(1, n, aud.shape[-1]))
        return torch.cat([zeros, aud], dim=1), aud.shape[1]

    def __len__(self):
        length = self.num_examples // self.batch_size
        return length + 1 if self.num_examples % self.batch_size > 0 else length

    def get_max_duration(self, start_idx: int, end_idx: int) -> float:
        max_duration = self.df["duration"].iloc[start_idx:end_idx].max()
        return float(max_duration)

    def get_max_text_length(self, start_idx: int, end_idx: int) -> int:
        texts = self.df["text"].iloc[start_idx:end_idx]
        max_length = max(len(str(text)) for text in texts)
        return max_length

    def get_audios(self, start_idx: int, end_idx: int) -> Tensor:
        max_duration = self.get_max_duration(start_idx, end_idx)
        results = [
            self._get_padded_aud(path, max_duration)
            for path in self.df["audio_path"].iloc[start_idx:end_idx]
        ]
        result, lengths = [t[0] for t in results], [t[1] for t in results]
        result = torch.stack(result, dim=1)
        return torch.squeeze(result), torch.IntTensor(lengths)

    def get_texts(self, start_idx: int, end_idx: int) -> Tuple[Tensor, torch.IntTensor]:
        args = self.df["text"].iloc[start_idx:end_idx]
        lengths = [len(x) + 2 for x in args.values] # +2 because of SOS and EOS tokens
        max_len = self.get_max_text_length(start_idx, end_idx)
        result = torch.stack(
            [self._get_padded_tokens(text, max_len) for text in args], dim=0
        )
        return result.to(torch.int32), torch.IntTensor(lengths)

    def __iter__(self):
        self.idx = 0
        while self.idx * self.batch_size < self.num_examples:
            start = self.idx * self.batch_size
            end = min((self.idx + 1) * self.batch_size, self.num_examples)
            self.idx += 1
            yield *self.get_audios(start, end), *self.get_texts(start, end)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    csv_path = "files/test_sentences.csv"
    folder = "files/MFCC"
    csv_file = pd.read_csv(csv_path)
    import os

    def prepare_audio(audio_path: Union[str, Path]) -> Tensor:
        x, sr = torchaudio.load(audio_path, normalize=True)
        x = Resample(sr, 16000)(x)
        # print(audio_path, start)
        x = mfcc(x)
        x = torch.unsqueeze(x, 0)
        return x

    for idx, row in csv_file.iterrows():
        audio_path = row["audio_path"]

        first_slash_index = audio_path.find("/")
        path = (
            audio_path[: first_slash_index + 1]
            + "MFCC/"
            + audio_path[first_slash_index + 1 :]
        )
        path, ext = path.split(".")
        path = f"{path}.npy"
        os.makedirs("/".join(path.split('/')[:-1]), exist_ok=True)

        mfcc_coef = prepare_audio(audio_path)
        print(path)

        np.save(path, mfcc_coef)

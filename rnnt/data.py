import math
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Union, Tuple
from torchaudio.transforms import Resample, MFCC
import torchaudio


class BaseData:
    def __init__(
        self,
        tokenizer,
        max_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _get_padded_aud(
        self,
        aud_path: Union[str, Path],
        max_duration: int,
    ) -> Tensor:
        max_len = 1 + math.ceil(
            max_duration * 16000 / 200
        )
        aud = self.prepare_audio(aud_path)
        n = max_len - aud.shape[1]
        zeros = torch.zeros(size=(1, n, aud.shape[-1]))

        return torch.cat([zeros, aud], dim=1), aud.shape[1]

    def _get_padded_tokens(self, text: str, max_len:int) -> Tensor:
        #max_len +=1
        sos_idx = self.tokenizer.special_tokens[self.tokenizer._sos_key][1]
        text = self.prepare_text(text)
        tokens = self.tokenizer.tokens2ids(text)
        #tokens = [sos_idx] + tokens
        eos_idx = self.tokenizer.special_tokens[self.tokenizer._eos_key][1]
        #tokens.append(eos_idx)
        length = max_len - len(tokens)
        pad_idx = self.tokenizer.special_tokens[self.tokenizer._phi_key][1]
        tokens = tokens + [pad_idx] * length
        return torch.IntTensor(tokens)

    def prepocess_lines(self, data: str) -> List[str]:
        return [item.split("_") for item in data]

    def prepare_audio(self, audio_path: Union[str, Path]) -> Tensor:
        x, sr = torchaudio.load(audio_path, normalize=True)
        x = Resample(sr, 16000)(x)
        x = MFCC(
            n_mfcc=25,
            melkwargs={"n_fft": 400, "hop_length": 200, "n_mels": 32, "center": False},
        )(x)
        x = x.permute(0, 2, 1)
        return x

    def prepare_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace(" ", "_")
        text = "".join(char if char.isalpha() or char == "_" else "" for char in text)
        text = text.strip()
        return text


class DataLoader(BaseData):
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        batch_size: int,
        max_len: int,
    ) -> None:
        super().__init__(tokenizer, max_len)
        self.batch_size = batch_size
        self.df = pd.read_csv(file_path)
        self.num_examples = len(self.df)
        self.idx = 0

    def __len__(self):
        length = self.num_examples // self.batch_size
        return length + 1 if self.num_examples % self.batch_size > 0 else length

    def get_max_duration(self, start_idx: int, end_idx: int) -> float:
        return self.df['duration'].iloc[start_idx:end_idx].max()
    
    def get_max_text_length(self, start_idx: int, end_idx: int) -> int:
        texts = self.df['text'].iloc[start_idx:end_idx]
        max_length = max(len(str(text)) for text in texts)
        return max_length


    def get_audios(self, start_idx: int, end_idx: int) -> Tensor:
        max_duration = self.get_max_duration(start_idx, end_idx)
        results = [
            self._get_padded_aud(path, max_duration)
            for path in self.df['audio_path'].iloc[start_idx:end_idx]
        ]
        result, lengths = [t[0] for t in results], [t[1] for t in results]
        result = torch.stack(result, dim=1)

        return torch.FloatTensor(torch.squeeze(result)), torch.IntTensor(lengths)

    def get_texts(self, start_idx: int, end_idx: int) -> Tuple[Tensor, torch.IntTensor]:
        args = self.df['text'].iloc[start_idx:end_idx]
        lengths = [len(x) for x in args.values]
        max_len = self.get_max_text_length(start_idx, end_idx)
        result = torch.stack([self._get_padded_tokens(text, max_len) for text in args], dim=0)
        return torch.IntTensor(result), torch.IntTensor(lengths)

    def __iter__(self):
        self.idx = 0
        while self.idx * self.batch_size < self.num_examples:
            start = self.idx * self.batch_size
            end = min((self.idx + 1) * self.batch_size, self.num_examples)
            self.idx += 1
            a,b = self.get_audios(start, end)
            c,d = self.get_texts(start, end)
            yield a,b, c,d

import math
import os
import numpy as np
import pandas as pd
import torch
from tokenizer import CharTokenizer


class AudioProcessor:
    def __init__(self, config):
        self.config = config

    def get_padded_features(self, audio_path: str, max_duration: int):
        max_len = math.ceil(
            (max_duration - self.config.win_size) / self.config.hop_size
        )
        feature_path = f"{os.path.splitext(audio_path)[0]}.npy"
        feature = torch.tensor(np.load(feature_path))

        feature_length = feature.shape[1]
        pads = torch.zeros((1, max_len - feature_length, feature.shape[-1]))
        feature = torch.cat([feature, pads], dim=1)
        return feature, feature_length

    def get_max_duration(self, df: pd.DataFrame, start_idx: int, end_idx: int):
        return df["duration"].iloc[start_idx:end_idx].astype(int).max()

    def get_audios(self, df: pd.DataFrame, start_idx: int, end_idx: int):
        max_duration = self.get_max_duration(df, start_idx, end_idx)
        features_lengths_tuples = [
            self.get_padded_features(audio_path, max_duration)
            for audio_path in df["audio_path"].iloc[start_idx:end_idx]
        ]
        features, features_lengths = zip(*features_lengths_tuples)
        return torch.cat(features, dim=0), torch.tensor(features_lengths, dtype=torch.int32)


class TextProcessor:
    def __init__(self, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.stoi[self.tokenizer.special_tokens.get("pad", None)]
        self.sos_idx = tokenizer.stoi[self.tokenizer.special_tokens["sos"]]
        self.eos_idx = tokenizer.stoi[self.tokenizer.special_tokens["eos"]]

    def get_padded_indices(self, tokens: list[str], max_len: int):
        ids = self.tokenizer.tokens2ids(tokens)
        ids_length = len(ids)
        ids = [self.sos_idx] + ids + [self.eos_idx]
        if self.pad_idx is not None:
            ids = ids + [self.pad_idx] * (max_len - ids_length)
        return torch.tensor(ids, dtype=torch.int32), torch.tensor(ids_length, dtype=torch.int32)

    def get_max_text_length(self, transcripts: pd.Series):
        return max(len(str(transcript)) for transcript in transcripts)

    def get_indices(self, df: pd.DataFrame, start_idx: int, end_idx: int):
        transcripts = df["transcript"].iloc[start_idx:end_idx]
        max_len = self.get_max_text_length(transcripts)
        ids_lengths_tuples = [
            self.get_padded_indices(transcript, max_len) for transcript in transcripts
        ]
        ids, ids_lengths = zip(*ids_lengths_tuples)
        return torch.stack(ids, dim=0), torch.stack(ids_lengths, dim=0)


class DataLoader:
    def __init__(
        self,
        transcript_path: str,
        batch_size: int,
        audio_processor: AudioProcessor,
        text_processor: TextProcessor,
    ):
        self.df = pd.read_csv(transcript_path)
        self.batch_size = batch_size
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.num_examples = len(self.df)

    def __len__(self):
        length = self.num_examples // self.batch_size
        return length + 1 if self.num_examples % self.batch_size > 0 else length

    def __iter__(self):
        self.idx = 0
        while self.idx * self.batch_size < self.num_examples:
            start = self.idx * self.batch_size
            end = min((self.idx + 1) * self.batch_size, self.num_examples)
            self.idx += 1
            audios, audios_lengths = self.audio_processor.get_audios(
                self.df, start, end
            )
            indices, indices_lengths = self.text_processor.get_indices(
                self.df, start, end
            )
            yield audios, audios_lengths, indices, indices_lengths

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

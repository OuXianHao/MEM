# Copyright 2024 Bytedance Ltd. and/or its affiliates

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """Prompt/response text SFT dataset that returns tokenized tensors with loss_mask."""

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='question',
                 response_key='answer',
                 prompt_dict_keys=None,
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error',
                 cache_dir='~/.cache/verl/sft'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.prompt_dict_keys = prompt_dict_keys
        self.response_dict_keys = response_dict_keys
        self.max_length = max_length
        self.truncation = truncation

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files(self):
        dfs = [pd.read_parquet(p) for p in self.parquet_files]
        self.dataframe = pd.concat(dfs)

    def __len__(self):
        return len(self.dataframe)

    def _extract_field(self, row_dict, key, dict_keys):
        value = row_dict[key]
        if dict_keys:
            cur = value
            for k in dict_keys:
                cur = cur[k]
            value = cur
        return str(value)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx].to_dict()
        prompt = self._extract_field(row, self.prompt_key, self.prompt_dict_keys)
        response = self._extract_field(row, self.response_key, self.response_dict_keys)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            if len(prompt_ids) >= overflow:
                prompt_ids = prompt_ids[overflow:]
                input_ids = prompt_ids + response_ids
                loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            else:
                input_ids = input_ids[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.where(input_ids != pad_id, 1, 0)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask.long(),
            'position_ids': position_ids.long(),
            'loss_mask': loss_mask,
        }


class TokenizedSFTDataset(Dataset):
    """Read pre-tokenized SFT parquet with columns input_ids/attention_mask/position_ids/loss_mask."""

    REQUIRED_COLUMNS = ['input_ids', 'attention_mask', 'position_ids', 'loss_mask']

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 cache_dir='~/.cache/verl/tokenized_sft'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files(self):
        dfs = [pd.read_parquet(p) for p in self.parquet_files]
        self.dataframe = pd.concat(dfs)

        for c in self.REQUIRED_COLUMNS:
            if c not in self.dataframe.columns:
                raise ValueError(f'Missing required column `{c}` in tokenized parquet.')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(row['attention_mask'], dtype=torch.long)
        position_ids = torch.tensor(row['position_ids'], dtype=torch.long)
        loss_mask = torch.tensor(row['loss_mask'], dtype=torch.float32)

        l0 = input_ids.numel()
        if not (l0 == attention_mask.numel() == position_ids.numel() == loss_mask.numel()):
            raise ValueError(f'Inconsistent lengths at idx={idx}.')

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }

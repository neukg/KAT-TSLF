import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset
from transformers import BartTokenizer
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

@dataclass
class Example(object):
    src: str
    knowl: List[str]
    target: str
    label: Optional[List[int]] = None

class KATDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        # **dataset_kwargs
    ):
        super().__init__()
        if '.' in type_path:
            data_file = Path(data_dir).joinpath(type_path)
        else:
            data_file = Path(data_dir).joinpath(type_path + ".pkl")
        self.examples = self.load_file(data_file)
        if n_obs is not None and n_obs >= 1:
            self.examples = self.examples[:n_obs]

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else "" 
        self.pad_token_id = self.tokenizer.pad_token_id

    def load_file(self, filename):
        filename = Path(filename)
        if filename.suffix in ('.pkl', '.pt'):
            return torch.load(filename)
        examples = []
        with filename.open() as f:
            for line in f:
                jsonobj = json.loads(line)
                ex = Example(
                    jsonobj["context"], 
                    [jsonobj["knowledge"]] if isinstance(jsonobj["knowledge"], str) else jsonobj["knowledge"], 
                    jsonobj["response"],
                )
                examples.append(ex)
        return examples

    def read_targets(self):
        return [t.target for t in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> Dict[str, str]:
        return self.examples[index]


class KATDataCollator:
    def __init__(self, tokenizer: BartTokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            input_ids, attention_mask, kno_input_ids, kno_attention_mask, labels = self._encode(batch)

        else:
            assert 0

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "kno_input_ids": kno_input_ids,
            "kno_attention_mask": kno_attention_mask,
            "labels": labels,
        }
        return batch

    def _encode(self, batch:List[Example]) -> Dict[str, torch.Tensor]:
        src_texts = [x.src for x in batch]
        tgt_texts = [x.target for x in batch]
        kno_texts = []
        max_num_kno = self.data_args.max_num_kno
        nums = [len(x.knowl) for x in batch]
        max_num_kno = min(max(nums), max_num_kno)
        for x in batch:
            kno = x.knowl[:max_num_kno]
            if len(kno) < max_num_kno:
                kno = kno + ['pad'] * (max_num_kno - len(kno))
            kno_texts.extend(kno)
        generator_datas = self.tokenizer.prepare_seq2seq_batch(
            src_texts,
            tgt_texts=tgt_texts,
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="longest",  
            return_tensors="pt",
        ).data
        input_ids, attention_mask, labels = (
            generator_datas["input_ids"],
            generator_datas["attention_mask"],
            generator_datas["labels"],
        )
        kno_datas = self.tokenizer.batch_encode_plus(
            kno_texts, max_length=self.data_args.max_kno_length,
            padding="longest", truncation=True, return_tensors="pt",
        )
        kno_input_ids = kno_datas["input_ids"]
        _, dylen = kno_input_ids.shape
        kno_input_ids = kno_input_ids.reshape(-1, max_num_kno, dylen)
        kno_attention_mask = kno_datas["attention_mask"]
        kno_attention_mask = kno_attention_mask.reshape(-1, max_num_kno, dylen)
        return input_ids, attention_mask, kno_input_ids, kno_attention_mask, labels
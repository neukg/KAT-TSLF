import json
from pathlib import Path
from kat.data import Example
from tqdm import tqdm
import random
import torch

SEP = '</s>'
SEPITDD = '&lt; SEP &gt;'

def load_examples(src_file, knl_file, tgt_file):
    examples = []
    num_kno = 0
    num_kno_tokens = 0
    num_sam = 0
    with open(src_file) as fs, open(knl_file) as fk, open(tgt_file) as ft:
        for src, kno, tgt in zip(fs, fk, ft):
            src = src.strip().replace(SEPITDD, SEP)
            knos = set()
            for doc in kno.split(SEPITDD):
                knos.add(doc.strip())
            knos = list(knos)
            num_kno += len(knos)
            num_sam += 1
            for k in knos:
                num_kno_tokens += len(k.split(' '))
            tgt = tgt.strip()
            examples.append(Example(src, knos, tgt))
    print("num tokens per kno:", num_kno_tokens / num_kno)
    print("num kno per dialog:", num_kno / num_sam)
    return examples

path = Path('dataset/ITDD_data')
outdir = Path('dataset/cmudog_kat')
outdir.mkdir(exist_ok=True, parents=True)

examples = load_examples(path / 'src-test-tokenized.txt', path / 'knl-test-tokenized.txt', path / 'tgt-test-tokenized.txt')

torch.save(examples, outdir / 'test.pkl')

examples = load_examples(path / 'src-train-tokenized.txt', path / 'knl-train-tokenized.txt', path / 'tgt-train-tokenized.txt')
num_exp = len(examples)
torch.save(examples, outdir / 'train.pkl')
torch.save(examples[:num_exp // 2], outdir / 'train_2.pkl')
torch.save(examples[:num_exp // 4], outdir / 'train_4.pkl')
torch.save(examples[:num_exp // 8], outdir / 'train_8.pkl')
torch.save(examples[:num_exp // 16], outdir / 'train_16.pkl')
torch.save(examples[:num_exp // 32], outdir / 'train_32.pkl')
torch.save(examples[:num_exp // 64], outdir / 'train_64.pkl')
torch.save(examples[:num_exp // 128], outdir / 'train_128.pkl')
torch.save(examples[:num_exp // 256], outdir / 'train_256.pkl')
torch.save(examples[:num_exp // 512], outdir / 'train_512.pkl')


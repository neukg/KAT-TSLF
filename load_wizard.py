import json
from pathlib import Path
from kat.data import Example
from tqdm import tqdm
import random
import torch
from typing import List

SEP = '</s>'

def examples_stat(examples:List[Example]):
    num_ex = len(examples)
    num_kno = 0
    kno_len = 0
    ctx_len = 0
    res_len = 0
    num_tun = 0
    for e in examples:
        ctx = e.src
        ctx_len += len(ctx.replace(SEP, " ").split(' '))
        res_len += len(e.target.split(' '))
        num_tun += len(ctx.split(SEP))
        num_kno += len(e.knowl)
        for kno in e.knowl:
            kno_len += len(kno.split(' '))
    print("****** Stat ******")
    print("num examples:", num_ex)
    print("avg. kno:", num_kno / num_ex)
    print("avg. kno len:", kno_len / num_kno)
    print("avg. turn:", num_tun / num_ex)
    print("avg. context len:", ctx_len / num_ex)
    print("avg. response len:", res_len / num_ex)
    print("******************")

def load_episodes(file):
    episodes = []
    with open(file) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes

def to_seq2seq(episodes):
    samples = []
    for epi in episodes:
        history = []
        for turn in epi:
            ctx = turn['context'].replace('\n', ' ').strip()
            res = turn['response'].replace('\n', ' ').strip()
            history.append(ctx)
            samples.append((SEP.join(history[-5:]), res))
            history.append(res)
    return samples

def to_examples(episodes):
    samples = []
    num_kno = 0
    num_kno_tokens = 0
    for epi in tqdm(episodes):
        history = []
        for turn in epi:
            ctx = turn['context'].replace('\n', ' ').strip()
            res = turn['response'].replace('\n', ' ').strip()
            kno = [k.split('__knowledge__')[1].strip() for k in turn['knowledge_sentences']]
            for k in kno:
                num_kno_tokens += len(k.split(' '))
            num_kno += len(kno)
            history.append(ctx)
            samples.append(Example(SEP.join(history[-5:]), kno, res))
            history.append(res)
    print("mean kno per sample:", num_kno / len(samples), ", num tokens per kno:", num_kno_tokens / num_kno)
    return samples

def save_seq2seq(samples, src_file, tgt_file):
    with open(src_file, 'w') as fs, open(tgt_file, 'w') as ft:
        for src, tgt in samples:
            fs.write(src + '\n')
            ft.write(tgt + '\n')

episodes = load_episodes('dataset/wizard/test_seen.jsonl')
outdir = Path('dataset/wizard_kat')
outdir.mkdir(exist_ok=True, parents=True)
samples = to_examples(episodes)
examples_stat(samples)
# assert 0
torch.save(samples, outdir / 'test_seen.pkl')

episodes = load_episodes('dataset/wizard/test_unseen.jsonl')
outdir = Path('dataset/wizard_kat')
outdir.mkdir(exist_ok=True, parents=True)
samples = to_examples(episodes)
examples_stat(samples)
torch.save(samples, outdir / 'test_unseen.pkl')

episodes = load_episodes('dataset/wizard/train.jsonl')
num_epi = len(episodes)
print("num eposodes:", num_epi)
outdir = Path('dataset/wizard_kat')
outdir.mkdir(exist_ok=True, parents=True)
samples = to_examples(episodes)
examples_stat(samples)
torch.save(samples, outdir / 'train.pkl')

torch.save(to_examples(episodes[:num_epi // 2]), outdir / 'train_2.pkl')
torch.save(to_examples(episodes[:num_epi // 4]), outdir / 'train_4.pkl')
torch.save(to_examples(episodes[:num_epi // 8]), outdir / 'train_8.pkl')
torch.save(to_examples(episodes[:num_epi // 16]), outdir / 'train_16.pkl')
torch.save(to_examples(episodes[:num_epi // 32]), outdir / 'train_32.pkl')
torch.save(to_examples(episodes[:num_epi // 64]), outdir / 'train_64.pkl')
torch.save(to_examples(episodes[:num_epi // 128]), outdir / 'train_128.pkl')
torch.save(to_examples(episodes[:num_epi // 256]), outdir / 'train_256.pkl')
torch.save(to_examples(episodes[:num_epi // 512]), outdir / 'train_512.pkl')


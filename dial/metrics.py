from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk import word_tokenize

import language_evaluation
from typing import List
from collections import defaultdict, Counter
import re
import math
import sys


def mean(lst):
    return sum(lst) / len(lst)

def _calc_ngram_dict(tokens:List[str], ngram:int, dict_ref=None):
    ngram_dict = defaultdict(int) if dict_ref is None else dict_ref
    total = len(tokens)
    for i in range(0, total - ngram + 1):
        item = tuple(tokens[i:i + ngram])
        ngram_dict[item] += 1
    return ngram_dict

def _calc_cover(cand, gold, ngram):
    cand_dict = _calc_ngram_dict(cand, ngram)
    gold_dict = _calc_ngram_dict(gold, ngram)
    cover = 0
    total = 0
    for token, freq in cand_dict.items():
        if token in gold_dict:
            cover += min(freq, gold_dict[token])
        total += freq
    return cover, total

def _calc_cover_rate(cands, golds, ngram):
    """
    calc_cover_rate
    """
    cover = 0.0
    total = 0.000001
    for cand_tokens, gold_tokens in zip(cands, golds):
        cur_cover, cur_total = _calc_cover(cand_tokens, gold_tokens, ngram)
        cover += cur_cover
        total += cur_total
    return cover / total

def _calc_bp(cands, golds):
    c_count = 0.000001
    r_count = 0.0
    for cand_tokens, gold_tokens in zip(cands, golds):
        c_count += len(cand_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp

def calc_corpus_bleu(cands, golds):
    bp = _calc_bp(cands, golds)
    cover_rate1 = _calc_cover_rate(cands, golds, 1)
    cover_rate2 = _calc_cover_rate(cands, golds, 2)
    cover_rate3 = _calc_cover_rate(cands, golds, 3)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
    return bleu1, bleu2, bleu3

# def calc_corpus_bleu_new(cands, golds):
#     golds = [[gold] for gold in golds]
#     sf = SmoothingFunction().method7
#     bleu1 = corpus_bleu(golds, cands, smoothing_function=sf, weights=[1, 0, 0, 0])
#     bleu2 = corpus_bleu(golds, cands, smoothing_function=sf, weights=[0.5, 0.5, 0, 0])
#     bleu3 = corpus_bleu(golds, cands, smoothing_function=sf, weights=[0.34, 0.33, 0.33, 0])
#     return bleu1, bleu2, bleu3

def calc_sentence_bleu(cands, golds):
    bleu1 = []
    bleu2 = []
    bleu3 = []
    sf = SmoothingFunction().method7
    for hyp, ref in zip(cands, golds):
        try:
            b1 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[1, 0, 0, 0])
        except ZeroDivisionError:
            b1 = 0.0
        try:
            b2 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[0.5, 0.5, 0, 0])
        except ZeroDivisionError:
            b2 = 0.0
        try:
            b3 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[0.34, 0.33, 0.33, 0])
        except ZeroDivisionError:
            b3 = 0.0
        bleu1.append(b1)
        bleu2.append(b2)
        bleu3.append(b3)
    return mean(bleu1), mean(bleu2), mean(bleu3)

def calc_corpus_bleu_new(hypothesis, references):
    # hypothesis = [normalize_answer(hyp).split(" ") for hyp in hypothesis]
    # references = [[normalize_answer(ref).split(" ")] for ref in references]
    references = [[gold] for gold in references]
    sf = SmoothingFunction(epsilon=1e-12).method1
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=sf)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=sf)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=sf)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=sf)
    return b1, b2, b3, b4

def _calc_distinct_ngram(cands, ngram):
    ngram_total = 0.00001
    ngram_distinct_count = 0.00001
    pred_dict = defaultdict(int)
    for cand_tokens in cands:
        _calc_ngram_dict(cand_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total

def _calc_sent_distinct_ngram(cand, ngram):
    ngram_total = 0.0000000001
    ngram_distinct_count = 0.0
    ngram_dict = defaultdict(int)
    for i in range(0, len(cand) - ngram + 1):
        item = tuple(cand[i:i + ngram])
        ngram_dict[item] += 1
    for _, freq in ngram_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total

def calc_corpus_distinct(cands):
    distinct1 = _calc_distinct_ngram(cands, 1)
    distinct2 = _calc_distinct_ngram(cands, 2)
    return distinct1, distinct2

def calc_sentence_distinct(cands):
    distinct1 = mean([_calc_sent_distinct_ngram(c, 1) for c in cands])
    distinct2 = mean([_calc_sent_distinct_ngram(c, 2) for c in cands])
    return distinct1, distinct2

def calc_corpus_f1(cands, golds):
    golden_word_total = 0.00000001
    pred_word_total = 0.00000001
    hit_word_total = 0.00000001
    for response, golden_response in zip(cands, golds):
        common = Counter(response) & Counter(golden_response)
        hit_word_total += sum(common.values())
        golden_word_total += len(golden_response)
        pred_word_total += len(response)
    p = hit_word_total / pred_word_total 
    r = hit_word_total / golden_word_total
    f1 = 2 * p * r / (p + r)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).split(' ')

def calc_rouge(cands, golds):
    rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)
    predictions = [' '.join(c) for c in cands]
    answers = [' '.join(g) for g in golds]
    rouge_result = rouge_evaluator.run_evaluation(predictions, answers)
    return rouge_result

def dialogue_evaluation(ori_cands, ori_golds):
    assert len(ori_cands) == len(ori_golds), f"num cand: {len(ori_cands)}, num gold: {len(ori_golds)}"
    cands = []
    golds = []
    help_tokenize = lambda x: word_tokenize(x.lower())
    for cand, gold in zip(ori_cands, ori_golds):
        cands.append(help_tokenize(cand.lower()))
        golds.append(help_tokenize(gold.lower()))
    cbleu1, cbleu2, cbleu3, cbleu4 = calc_corpus_bleu_new(cands, golds)
    sbleu1, sbleu2, sbleu3 = calc_sentence_bleu(cands, golds)
    cdiv1, cdiv2 = calc_corpus_distinct(cands)
    sdiv1, sdiv2 = calc_sentence_distinct(cands)
    cf1 = calc_corpus_f1(cands, golds)
    rouge_result = calc_rouge(cands, golds)
    result = {
        'cf1': cf1,
        'bleu1': cbleu1,
        'bleu2': cbleu2,
        'bleu3': cbleu3,
        'bleu4': cbleu4,
        'dist1': cdiv1,
        'dist2': cdiv2,
    }
    result.update(rouge_result)
    result = {k: round(100 * v, 6) for k, v in result.items()}
    return result

def file_dialogue_evaluation(cand_file, gold_file):
    print(f"cand file: {cand_file}, gold file: {gold_file}")
    cands = []
    golds = []
    with open(cand_file, 'r', encoding='utf-8') as f:
        for line in f:
            cands.append(line.strip())
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            golds.append(line.strip())
    results = dialogue_evaluation(cands, golds)
    print(results)

if __name__ == "__main__":
    cand_file = sys.argv[1]
    gold_file = sys.argv[2]
    file_dialogue_evaluation(cand_file, gold_file)
    
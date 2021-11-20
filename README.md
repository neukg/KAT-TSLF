# KAT-TSLF
Source code of paper "A Novel Three-Stage Learning Framework for Low-Resource Knowledge-Grounded Dialogue Generation"

## Environments
* python 3.7
* transformers 4.2.2
* NLTK
* pytorch
* language_evaluation (install from SKT project)

## Datasets and Models 
1. Download Wizard-of-Wikipedia, CMU_DoG and pseudo dataset (used in stage II) from [here](https://cloud.189.cn/t/qAna2iUf6vmy
https://stuneueducn-my.sharepoint.com/:u:/g/personal/20151119_stu_neu_edu_cn/EUyf3Jpqeu5Fj2Eamv16CK8Bp6Z3jDnCjGouQauh5CiI1g?e=Ccbdbl).
2. Download BART pre-trained on Reddit Conversation Corpus and Wikipedia dumps from [here](https://drive.google.com/file/d/1f2VFmTkmOh4w05Dll2a9x-o8YjTimAnK/view?usp=sharing) and [here](https://drive.google.com/file/d/1GjMP8cRAJfWXCYoUpR5xp2CUhSkLMgZY/view?usp=sharing).
3. (Optional) Download pre-trained checkpoint from [here](https://cloud.189.cn/t/UbmUjeVBfaAz
https://stuneueducn-my.sharepoint.com/:u:/g/personal/20151119_stu_neu_edu_cn/EXj-F55Y1AlNqh_SnQm9vt0BWHnoN3oldH-gQRFKPN_MYg?e=L8Urej). 

## (Optional) Run Stage II 
```bash
bash scripts/warmup.sh
```

## Run Stage III 
Low-resource on Wizard-of-Wikipedia: 
```bash
bash scripts/wizardlr.sh
```
Zero-resource on Wizard-of-Wikipedia: 
```bash
bash scripts/zr_wizard.sh
```
Low-resource on CMU_DoG: 
```bash
bash scripts/cmudoglr.sh
```
Zero-resource on CMU_DoG: 
```bash
bash scripts/zr_cmudog.sh
```
(Please adjust *beam size* appropriately)

## Cite
```
@inproceedings{liu-etal-2021-three,
    title = "{A} {T}hree-{S}tage {L}earning {F}ramework for {L}ow-{R}esource {K}nowledge-{G}rounded {D}ialogue {G}eneration",
    author = "Liu, Shilei  and
      Zhao, Xiaofeng  and
      Li, Bochao  and
      Ren, Feiliang  and
      Zhang, Longhui  and
      Yin, Shujuan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.173",
    pages = "2262--2272",
    abstract = "Neural conversation models have shown great potentials towards generating fluent and informative responses by introducing external background knowledge. Nevertheless, it is laborious to construct such knowledge-grounded dialogues, and existing models usually perform poorly when transfer to new domains with limited training samples. Therefore, building a knowledge-grounded dialogue system under the low-resource setting is a still crucial issue. In this paper, we propose a novel three-stage learning framework based on weakly supervised learning which benefits from large scale ungrounded dialogues and unstructured knowledge base. To better cooperate with this framework, we devise a variant of Transformer with decoupled decoder which facilitates the disentangled learning of response generation and knowledge incorporation. Evaluation results on two benchmarks indicate that our approach can outperform other state-of-the-art methods with less training data, and even in zero-resource scenario, our approach still performs well.",
}
```
# KAT-TSLF
Source code of paper "A Novel Three-Stage Learning Framework for Low-Resource Knowledge-Grounded Dialogue Generation"

## Environments
* python 3.6+
* transformers 3.2+
* NLTK
* pytorch
* language_evaluation (install from SKT project)

## Datasets and Models 
1. Download Wizard-of-Wikipedia and CMU_DoG from here.
2. Download BART (pre-trained on Reddit Conversation Corpus and Wikipedia dumps) from here.
3. Download pseudo dataset (used in stage II) from here.

## Run Stage II 
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
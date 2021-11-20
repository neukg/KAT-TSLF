# KAT-TSLF
Source code of paper "A Novel Three-Stage Learning Framework for Low-Resource Knowledge-Grounded Dialogue Generation".

## Environments
* python 3.7
* transformers 4.2.2
* NLTK
* pytorch
* language_evaluation (install from [SKT](https://github.com/bckim92/sequential-knowledge-transformer) project)

## Datasets and Models 
1. Download Wizard-of-Wikipedia, CMU_DoG and pseudo dataset (used in stage II) from [[OneDrive]](https://stuneueducn-my.sharepoint.com/:u:/g/personal/20151119_stu_neu_edu_cn/EUyf3Jpqeu5Fj2Eamv16CK8Bp6Z3jDnCjGouQauh5CiI1g?e=Ccbdbl) [[189 Clond]](https://cloud.189.cn/t/qAna2iUf6vmy) [[Google Drive]](https://drive.google.com/file/d/1XVDs-sTlTZfd1vQAAtSFsxMx6V7TZ1HE/view?usp=sharing).
2. Download BART pre-trained on Reddit Conversation Corpus and Wikipedia dumps from [[Google Drive]](https://drive.google.com/file/d/1f2VFmTkmOh4w05Dll2a9x-o8YjTimAnK/view?usp=sharing) and [[Google Drive]](https://drive.google.com/file/d/1GjMP8cRAJfWXCYoUpR5xp2CUhSkLMgZY/view?usp=sharing).
3. (Optional) Download pre-trained checkpoint from [[OneDrive]](https://stuneueducn-my.sharepoint.com/:u:/g/personal/20151119_stu_neu_edu_cn/EXj-F55Y1AlNqh_SnQm9vt0BWHnoN3oldH-gQRFKPN_MYg?e=L8Urej) [[189 Clond]](https://cloud.189.cn/t/UbmUjeVBfaAz) [[Google Drive]](https://drive.google.com/file/d/1ZuMV9fnBrg-rgFh7btaIC0cyjgtkQAjL/view?usp=sharing). 

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
}
```
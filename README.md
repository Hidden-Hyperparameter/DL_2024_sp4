# Starter Code for Deep Learning Coding Project 4

## Acknowledgement

We thank [Shusheng Xu](https://github.com/xssstory) and [Yunfei Li](https://github.com/IrisLi17) for their initial contribution and [Wei Fu](https://github.com/garrett4wade) with document formatting and package version testing.

## Dataset

Download the datasets from https://cloud.tsinghua.edu.cn/d/1b60ac7277994f60b2e2/ and organize the folders as follows:

```
dl_course_hw4
├── Datasets
    └── CCPC
    ├── CLS
    └── couplet
├── generation
├── classification
└── README.md
```

## Generation

```bash
cd generation
# Seq2Seq generation
python train.py  --seq2seq --model-type transformer [optional arguments]
python train.py  --seq2seq --model-type lstm [optional arguments]
# Unconditional generaton
python train.py --model-type transformer [optional arguments]
python train.py --model-type lstm [optional arguments]
# evaluate
python evaluation.py
```

## Classification

```bash
cd classification
# training
python train.py [optional arguments]
# evaluate
python evaluation.py
```


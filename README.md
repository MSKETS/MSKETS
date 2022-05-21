# MSKETS

 Source code and datasets for "MSKETS: Multi-Source Knowledge Enhanced language representation with Type Selection". [here](https://github.com/MSKETS/MSKETS.git)

## Requirements

```latex
Python == 3.7
Pytorch >= 1.10
transformers >= 4.10.0
dgl ==  0.7.2
```

## Prepare

- Download the `pytorch_model.bin` from [here]( [hfl/chinese-roberta-wwm-ext at main (huggingface.co)](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main) ), and save it to the `pretrained_models/chinese-roberta-wwm-ext/` directory.
- Download the `CMeKG and CN-DBpedia` from [here](https://pan.baidu.com/s/1Z3o88tqH349aO9n1AC-rvQ )(提取码：qaj5), and save it to the` knowledge/` directory.
- Download the datasets from [here]( [数据集-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.22060218.J_2657303350.1.61151343Gf0Rpz&dataId=95414) ), place them in the `datasets/` directory.

## Run tasks

+ fine-tune on CMeEE: `python train_NER.py`, and predict an answer:`python predict_NER.py`
+ fine-tune on CMeIE: `python train_RE.py`, and predict an answer:`python predict_RE.py`
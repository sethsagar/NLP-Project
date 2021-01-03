# NLP-Project-Paws

Paraphrase Adversaries from Word Scrambling - A group project for CSE 576 Topics/Natural Language Processing

## The Team

1.Anirudh Krishna Lakshmanan

2.Sagar Seth 

3.Zhemin Zhang 

### Ex-Members

1.Yuanyuan Tian 

2.Xinyu Zhao 


## Getting Started

The datasets are provided in data_qqp and data_PAWS_qqp folders. The preprocessed data is provided in the next section. The model code is present in Models.ipynb.

Note: The code has been set up to be executed in colab. The file paths will need to be changed to accomodate running it in other systems.

### Setup

Use the following link to access pretrained models and the preprocessed [here](https://drive.google.com/open?id=17vbD6yC9KYcYqk6RdJfdSu5qfl81OsOn). Create a directory called Colab and place the `Models.ipynb` file in it. Create a folder for Checkpoints named `CHKPT_`. Place the corresponding folders obtained from extracting in the Colab directory.

Note: Do not run the preprocess data blocks since that will replace the existing preprocessed data files. Only do so if you want to extract other features.

### Setting run parameters for both models
 
- The base model used can be changed by changing the following block:

```
MODEL_CLASS, TOKENIZER_CLASS, PRETRAINED = (BertForSequenceClassification, BertTokenizer, 'bert-base-cased')
```

- If PAWS dataset is to be included, change the value of `PAWS_QQP` as `True`.

- The other model parameters such as learning rate can be changed by specifying them in the model details.

### Note for running custom model

- Please execute the class `Model` corresponding to the architecture you want to use.

## Results obtained

All the models are done using DistilBERT base model. The custom models are trained on both QQP and PAWS_QQP.

| No | Model Details | QQP (dev) | PAWS_QQP (dev) |
|----|---------------|:---------:|:--------------:|
| **1** | Default on QQP (train) | 88.58 | 31.76 | 
| **2** | Default on QQP + PAWS_QQP (train) | 88.10 | 80.50 | 
| **3** | Custom Model, Attention on both sentence and phrase; Cross-Attention; Residual skip connection included | 87.26 | 57.75 | 
| **4** | Custom Model, Attention on both sentence and phrase; Cross-Attention; No residual skip connection | 85.63 | 55.83 | 
| **5** | Custom Model, Attention on both sentence and phrase; Self-Attention; No residual skip connection | 84.94 | 56.28 | 
| **6** | Custom Model, Attention on phrase only; Self-Attention; No residual skip connection | 85.49 | 53.91 | 
| **7** | Custom Model, No attention; No residual skip connection | 86.08 | 54.06 | 

## Instructor and TA

Instructor: *Chitta Baral*

TA: *Pratyay Banerjee* 

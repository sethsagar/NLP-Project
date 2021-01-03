import torch
from transformers import *
import numpy as np
import scipy as scipy
import pandas as pd
import os
import ast
import tqdm as tqdm
import spacy


MODEL_CLASS, TOKENIZER_CLASS, PRETRAINED = (DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-cased')
PAWS_QQP = True

SPACY_CORE = spacy.load("en_core_web_sm")

DATA_FOLDER = 'data_PAWS_qqp'
FILE_NAMES = ['train.tsv', 'dev.tsv']

class Tokenizer:
    # init
    def __init__(self, tokenizer_class, pretrained_weights):
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        
    # tokenize
    def tokenize_data(self, data, file_path, group_sent=True):
        bar = tqdm.tqdm(total=len(data))
        tokenized = None
        if group_sent == False:
            tokenized = pd.DataFrame(columns = ['i1', 's1', 'a1', 'n11', 'n12', 'm1', 'i2', 's2', 'a2', 'n21', 'n22', 'm2', 'y'])
        else:
            tokenized = pd.DataFrame(columns = ['i', 's', 'a', 'y'])
        count = 0
        flag = True
        i = 0
        for index, row in data.iterrows():
            token_row = {}
            # print(row)
            # QQP Dataset
            # sent1 = str(row['question1'])
            # sent2 = str(row['question2'])
            # token_row['y'] = int(row['is_duplicate'])
            
            # PAWS Dataset
            sent1 = str(row['sentence1'])
            sent2 = str(row['sentence2'])
            token_row['y'] = int(row['label\r'])
            # if count / 10000 < 14:
            #     count += 1
            #     continue
            
            if group_sent == True:
                encoding = self.tokenizer.encode_plus(sent1, sent2, return_token_type_ids=True, max_length=256, pad_to_max_length=True)
                token_row['i'] = encoding['input_ids']
                token_row['s'] = encoding['token_type_ids']
                token_row['a'] = encoding['attention_mask']
              
            else:
                encoding1 = self.tokenizer.encode_plus(sent1, return_token_type_ids=True, max_length=128, pad_to_max_length=True)

                token_row['i1'] = encoding1['input_ids']
                token_row['s1'] = encoding1['token_type_ids']
                token_row['a1'] = encoding1['attention_mask']

                starts1 = np.zeros(10, dtype=int)
                ends1 = np.ones(10, dtype=int)
                mask1 = np.zeros(10, dtype=int)
                doc1 = SPACY_CORE(sent1)
                j = 0
                for chunk in doc1.noun_chunks:
                    starts1[j] = chunk.start
                    ends1[j] = chunk.end
                    mask1[j] = 1
                    j += 1
                    if j == 10:
                        break

                token_row['n11'] = starts1
                token_row['n12'] = ends1
                token_row['m1'] = mask1

                encoding2 = self.tokenizer.encode_plus(sent2, return_token_type_ids=True, max_length=128, pad_to_max_length=True)

                token_row['i2'] = encoding2['input_ids']
                token_row['s2'] = encoding2['token_type_ids']
                token_row['a2'] = encoding2['attention_mask']


                starts2 = np.zeros(10, dtype=int)
                ends2 = np.ones(10, dtype=int)
                mask2 = np.zeros(10, dtype=int)
                doc2 = SPACY_CORE(sent2)
                j = 0
                for chunk in doc2.noun_chunks:
                    starts2[j] = chunk.start
                    ends2[j] = chunk.end
                    mask2[j] = 1
                    j += 1
                    if j == 10:
                        break

                token_row['n21'] = starts2
                token_row['n22'] = ends2
                token_row['m2'] = mask2

            tokenized = tokenized.append(token_row, ignore_index=True)
            bar.update()
            count += 1
            if count % 10000 == 0:
                tokenized.to_parquet(file_path + str(i) + '.parquet')
                tokenized = tokenized.iloc[0:0]
                i+=1
        if count % 10000 != 0:
            tokenized.to_parquet(file_path + str(i) + '.parquet')
        bar.close()
        return tokenized

data_path = './' + DATA_FOLDER + '/'
tokenizer = Tokenizer(TOKENIZER_CLASS, PRETRAINED)

for file_name in FILE_NAMES:
    data = pd.read_csv(os.path.expanduser(data_path+file_name), sep='\t', lineterminator='\n', error_bad_lines=False)
    data = data.dropna()
    tokens = tokenizer.tokenize_data(data, data_path + file_name.split('.')[0] + '_seperate', group_sent=False)
    # tokens = tokenizer.tokenize_data(data, data_path + file_name.split('.')[0] + '_grouped')
    # tokens.to_csv(data_path + file_name.split('.')[0] + '.csv')
    # display(tokens)
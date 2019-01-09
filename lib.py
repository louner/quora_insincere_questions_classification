import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_train_val():
    train = pd.read_csv('data/train.csv')
    train, val = train_test_split(train, test_size=0.2)
    train.to_csv('data/train')
    val.to_csv('data/val')

def evaluate(df):
    val_X, val_y = tfidf.transform(df['question_text']), df['target']
    pred = cls.predict(val_X)
    print(precision_score(pred, val_y), recall_score(pred, val_y), f1_score(pred, val_y))


def train_tfidf(X):
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    tfidf.fit(X)
    return tfidf

def build_embed_layer():
    glove = pd.read_csv('data/glove.840B.300d/glove.840B.300d.txt', sep=' ')
    words = glove.iloc[:, 0]
    word2id = {}
    for word in words:
        word2id[word] = len(word2id)
        
    embeddings = glove.iloc[:, 1:].values
    embed_layer = Embedding.from_pretrained(torch.Tensor(embeddings))
    return word2id, embed_layer

from torch.nn import Embedding
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
import json
from pdb import set_trace

from pytorch_pretrained_bert import BertTokenizer, BertModel

class TokToID(object):
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
    
    def toID(self, sentence):
        try:
            #set_trace()
            toks = self.tokenizer.tokenize(sentence)
            ids = self.tokenizer.convert_tokens_to_ids(toks)
            return ids
        except:
            set_trace()
        
    def __call__(self, item):
        item[0] = self.toID(item[0])
        return item

class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, item):
        item[0] = torch.from_numpy(np.array(item[0]))
        return item
    
class FixSentencesLength(object):
    def __init__(self, sentence_length=128, padding=0):
        self.sentence_length = sentence_length
        self.padding = padding
        
    def fix_length(self, sentence_ids):
        if len(sentence_ids) >= self.sentence_length:
            return sentence_ids[:self.sentence_length]
        else:
            return sentence_ids + [self.padding]*(self.sentence_length - len(sentence_ids))
         
    def __call__(self, item):
        item[0] = self.fix_length(item[0])
        return item

class QuoraInsinereQustion(Dataset):
    def __init__(self, fpath):
        self.df = pd.read_csv(fpath)
        self.transform = transforms.Compose([
            TokToID(),
            FixSentencesLength(),
            ToTensor()
        ])
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        item = self.df.iloc[index]
        sample = [item['question_text'], item['target']]
        if self.transform:
            try:
                sample = self.transform(sample)
            except:
                print(sample)
                raise
        return sample


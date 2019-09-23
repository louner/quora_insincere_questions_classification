from torch.nn import Embedding
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
import json
from pdb import set_trace

from pytorch_pretrained_bert import BertTokenizer, BertModel

import matplotlib.pyplot as plt
from IPython import display
from time import time

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy, Precision, Recall

from gensim.utils import tokenize
from imblearn.over_sampling import RandomOverSampler

from gensim.corpora import Dictionary
from gensim.utils import tokenize

dictionary_path = 'data/dictionary'
batch_size = 512
shuffle = True
num_workers = 4
sentence_length = 128
bert_hidden_size = 768
learning_rate = 1e-3
kernel_weights = [2,3,4]
out_channels = 8

def build_dictionary(train_path, test_path, no_below=2, keep_n=2000000):
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    df = pd.concat([train, test])
    sentences = df['question_text'].values

    documents = [list(tokenize(sentence)) for sentence in sentences]
    dictionary = Dictionary(documents=documents)

    dictionary.filter_extremes(no_below=no_below, keep_n=keep_n)
    dictionary.save(dictionary_fpath)
    return dictionary

def build_glove_embed(glove_path, dictionary):
    glove = pd.read_csv(glove_path, sep=' ')
    glove_word2id = dict([(word, id) for id,word in enumerate(glove.iloc[:, 0])])
    glove_embeddings = glove.iloc[:, 1:].values
    
    word2id = {}
    absent_words = []
    embeddings = deque()
    for word in dictionary.token2id:
        if word in glove_word2id:
            word2id[word] = len(word2id)
            embeddings.append(glove_embeddings[glove_word2id[word]])
            
        else:
            absent_words.append(word)
            
    embeddings = list(embeddings)
    
    for word in absent_words:
        word2id[word] = len(word2id)

    mean, std = np.mean(embeddings, axis=0), np.std(embeddings, axis=0)
    absent_words_embeddings = np.random.normal(loc=mean, scale=std, size=(len(absent_words), embed_dim))
    embeddings = np.concatenate([embeddings,absent_words_embeddings], axis=0)
    print(len(word2id), embeddings.shape)

    embed_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
    torch.save(embed_layer, embed_layer_fpath)

def load_dictionary():
    dictionary = Dictionary().load(dictionary_path)
    return dictionary

class TokToID(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def toID(self, sentence):
        try:
            #set_trace()
            ids = [self.dictionary.token2id.get(tok, -1) for tok in tokenize(sentence)]
            ids = [id for id in ids if id >= 0]
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
    def __init__(self, fpath, over_sample=False, transform=None):
        df = pd.read_csv(fpath)
        if over_sample:
            ros = RandomOverSampler()
            X, _ = ros.fit_resample(df, df['target'])
            self.df = pd.DataFrame(X, columns=df.columns)
        else:
            self.df = df
        
        if transform is None:
            self.transform = transforms.Compose([
                TokToID(load_dictionary()),
                FixSentencesLength(sentence_length),
                ToTensor()
            ])
        else:
            self.transform = transform

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
class Metrics:
    def __init__(self):
        self.train_metrics = pd.DataFrame()
        self.val_metrics = pd.DataFrame()
        
    def append_row(self, df, row):
        row = pd.DataFrame([row], columns=row.keys())
        return pd.concat([df, row], axis=0)

    def add_more_metrics(self, metric):
        if 'Precision' in metric:
            metric['Precision'] = metric['Precision'][1].item()

        if 'Recall' in metric:
            metric['Recall'] = metric['Recall'][1].item()

        f1 = 2*(metric['Recall']*metric['Precision'])/((metric['Recall'] + metric['Precision']+1e-20))
        if f1 > 1:
            f1 = 0
        metric['F1'] = f1
        
    def add_train(self, metric):
        self.add_more_metrics(metric)
        self.train_metrics = self.append_row(self.train_metrics, metric)
        
    def add_val(self, metric):
        self.add_more_metrics(metric)
        self.val_metrics = self.append_row(self.val_metrics, metric)

    def is_best_val(self):
        return self.val_metrics['accuracy'].max() == self.val_metrics['accuracy'].iloc[-1]

def plot_metrics(metrics):
    plt.gcf().clear()
    #plt.plot(metrics.train_metrics['Accuracy'].values, color='r')
    plt.plot(metrics.train_metrics['F1'].values, color='r')
    plt.plot(metrics.train_metrics['Loss'].values, color='g')
    #plt.plot(metrics.train_metrics['Precision'].values, color='y')
    #plt.plot(metrics.train_metrics['Recall'].values, color='b')

    plt.plot(metrics.val_metrics['Loss'].values, color='b')
    #plt.plot(metrics.val_metrics['Accuracy'].values, color='y')
    plt.plot(metrics.val_metrics['F1'].values, color='y')
    display.display(plt.gcf())
    display.clear_output(wait=True)

def build_dl(fpath, frac=0.00001, over_sample=False, transform=None):
    dataset = QuoraInsinereQustion(fpath, over_sample=over_sample, transform=transform)
    dataset.df = dataset.df.sample(frac=frac)
    dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dl

def train(model, train_dl, val_dl):
    #loss = CrossEntropyLoss()
    loss = loss = CrossEntropyLoss()
    trainable_tensors = [p[1] for p in model.named_parameters() if p[0].startswith('conv') or p[0].startswith('fc')]
    optimizer = Adam(params=trainable_tensors, lr=learning_rate)

    trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss)
    evaluator = create_supervised_evaluator(model=model, metrics={'Loss': Loss(loss), 'Accuracy': Accuracy(), 'Precision': Precision(), 'Recall': Recall()})
    metrics = Metrics()

    epoch_st = time()
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        evaluator.run(train_dl)
        metrics.add_train(evaluator.state.metrics)

        global epoch_st
        elasped_time = int(time()-epoch_st)
        epoch_st = time()

        print(f"epoch {trainer.state.epoch} {evaluator.state.metrics} {elasped_time}")
        evaluator.run(val_dl)
        metrics.add_val(evaluator.state.metrics)
        
        plot_metrics(metrics)

    trainer.run(train_dl, max_epochs=100)

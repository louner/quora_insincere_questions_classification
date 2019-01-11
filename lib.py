'''
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
'''

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

word2id_fpath = 'data/word2id'
batch_size = 256
shuffle = True
num_workers = 4
sentence_length = 32
bert_hidden_size = 768
learning_rate = 1e-3
kernel_weights = [2,3,4]
out_channels = 8

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

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for p in self.bert_model.parameters():
            p.requires_grad = False
        
        self.convs = []
        for kernel_weight in kernel_weights:
            conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=(kernel_weight, bert_hidden_size)
            )
            self.convs.append(conv)
            setattr(self, 'conv-%d'%(kernel_weight), conv)
        
        self.dropout = torch.nn.Dropout(p=0.5)
        
        self.fc1 = torch.nn.Linear(
            in_features=len(kernel_weights)*out_channels,
            out_features=2
        )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, X):
        #set_trace()
        segment_tensor = torch.zeros_like(X)
        # 12, sentence_length, 768
        encoder, _ = self.bert_model(X, segment_tensor)

        last_layer = encoder[-1]
        # for cnn (N, C, H, W) format, append in_channel in dim=1
        last_layer = torch.unsqueeze(last_layer, 1)
        #last_layer = self.dropout(last_layer)
        
        conv_outs = []
        for conv in self.convs:
            # batch_size, out_channels, sentence_length-kernel_weight+1, 1
            conv_out = conv(last_layer)
            # batch_size, out_channels, 1
            # dim=2 is maxed out
            conv_out, _ = torch.max(conv_out, dim=2)
            
            conv_outs.append(conv_out)

        # batch_size, out_channels*len(kernel_weights), 1
        conv_out = torch.cat(conv_outs, dim=1)
        conv_out = torch.squeeze(conv_out)
        # batch_size, out_channels*len(kernel_weights)
        conv_out = conv_out.view(X.shape[0], -1)
        
        fc_out = self.fc1(conv_out)
        logits = self.softmax(fc_out)
        
        return logits
            
def build_dl(fpath, frac=0.00001):
    dataset = QuoraInsinereQustion(fpath)
    dataset.df = dataset.df.sample(frac=frac)
    dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dl

if __name__ == '__main__':
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam, SGD

    train_dl = build_dl('data/train', frac=0.0001)
    val_dl = build_dl('data/val', frac=0.0001)

    model = CNN()
    #loss = CrossEntropyLoss()
    loss = CrossEntropyLoss(weight=torch.Tensor([1, 16]))
    trainable_tensors = [p[1] for p in model.named_parameters() if p[0].startswith('conv') or p[0].startswith('fc')]
    optimizer = Adam(params=trainable_tensors, lr=learning_rate)

    from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
    from ignite.metrics import Loss, Accuracy, Precision, Recall
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

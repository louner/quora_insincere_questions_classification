from torch.nn import Embedding
import torch
import numpy as np
import pandas as pd
import json
from pdb import set_trace
from lib import *
from pytorch_pretrained_bert import BertTokenizer, BertModel

word2id_fpath = 'data/word2id'
batch_size = 32
shuffle = True
num_workers = 4
sentence_length = 32
bert_hidden_size = 768
learning_rate = 1e-3

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        self.cnn1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(4,bert_hidden_size)
        )
        
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=(2,2)
        )
        
        self.dropout = torch.nn.Dropout(p=0.5)
        
        self.fc1 = torch.nn.Linear(
            in_features=1984,
            out_features=2
        )
        
        self.softmax = torch.nn.Softmax()
        
    def forward(self, X):
        #set_trace()
        segment_tensor = torch.zeros_like(X)
        encoder, _ = self.bert_model(X, segment_tensor)

        last_layer = encoder[-1]
        last_layer = torch.unsqueeze(last_layer, 1)

        cnn1_out = self.cnn1(last_layer)
        pool1_out = self.pool1(cnn1_out)

        pool1_out = pool1_out.view(batch_size, -1)
        
        pool1_out = self.dropout(pool1_out)
        pool1_out = self.fc1(pool1_out)
        logits = self.softmax(pool1_out)
        
        return logits

dataset = QuoraInsinereQustion('data/train')
train_dl = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
)

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
model = CNN()
#loss = CrossEntropyLoss()
loss = CrossEntropyLoss(weight=torch.Tensor([1, 16]))
trainable_tensors = [p[1] for p in model.named_parameters() if p[0].startswith('cnn') or p[0].startswith('fc')]
optimizer = Adam(params=trainable_tensors, lr=learning_rate)

from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    evaluator.run(train_dl)
    metrics.add_train(evaluator.state.metrics)

    global epoch_st
    elasped_time = int(time()-epoch_st)
    epoch_st = time()

    print(f"epoch {trainer.state.epoch} {evaluator.state.metrics} {elasped_time}")

trainer.run(train_dl, max_epochs=5)

from glove import *
import sys
sys.path.append('attention-is-all-you-need-pytorch')
from transformer.Models import Encoder

dictionary = load_dictionary()
embed_layer = torch.load(embed_layer_fpath)

n_src_vocab = len(dictionary)
len_max_seq = sentence_length
d_word_vec = embed_dim
n_layers = 1
d_model = embed_dim
d_inner = 2048
n_head = 8
d_k = int(d_model/8)
d_v = d_k
dropout = 0

class FixSentencesLength(object):
    def __init__(self, sentence_length=128, padding=0):
        self.sentence_length = sentence_length
        self.padding = padding

    def fix_length(self, sentence_ids):
        if len(sentence_ids) >= self.sentence_length:
            sentence_ids = sentence_ids[:self.sentence_length]
            pos = [pos for pos in range(1, len(sentence_ids)+1)]
            
        else:
            pos = [pos for pos in range(1, len(sentence_ids)+1)] + [0]*(self.sentence_length - len(sentence_ids))
            sentence_ids = sentence_ids + [self.padding]*(self.sentence_length - len(sentence_ids))
        return sentence_ids, pos

    def __call__(self, item):
        item[0] = self.fix_length(item[0])
        return item

class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_layer=None, seed=None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            
        encoder = Encoder(
            n_src_vocab=n_src_vocab,
            len_max_seq=len_max_seq,
            d_word_vec=d_word_vec,
            n_layers=n_layers,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        encoder.src_word_emb = embed_layer
        self.encoder = encoder
        
        for p in self.encoder.src_word_emb.parameters():
            p.require_grad = False
        
        self.fc1 = torch.nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim
        )
        
        self.fc2 = torch.nn.Linear(
            in_features=embed_dim,
            out_features=int(embed_dim/2)
        )

        self.fc3 = torch.nn.Linear(
            in_features=int(embed_dim/2),
            out_features=2
        )

        self.fc_act = torch.nn.Tanh()
        
    def forward(self, X):
        src_seq, src_pos = X[:, 0, :], X[:, 1, :]
        encoder_out, = self.encoder(src_seq, src_pos)
        encoder_out, _ = torch.max(encoder_out, dim=1)
        
        fc_out = self.fc1(encoder_out)
        fc_out = self.fc_act(fc_out)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc_act(fc_out)
        logits = self.fc3(fc_out)
        
        return logits

#if __name__ == '__main__':
transform = transforms.Compose([
                TokToID(load_dictionary()),
                FixSentencesLength(sentence_length),
                ToTensor()
            ])

train_dl = build_dl('data/train.csv', frac=0.01, over_sample=True, transform=transform)
val_dl = build_dl('data/val', frac=0.1)

torch.set_num_threads(8)
device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

embed_layer = torch.load(embed_layer_fpath)
model = TransformerEncoder(embed_layer, seed=1234)

loss = CrossEntropyLoss()

trainable_tensors = [p[1] for p in model.named_parameters() if p[0].startswith('conv') or p[0].startswith('fc')]
optimizer = Adam(params=trainable_tensors, lr=learning_rate)

trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss, device=device)
evaluator = create_supervised_evaluator(model=model, metrics={'Loss': Loss(loss), 'Accuracy': Accuracy(), 'Precision': Precision(), 'Recall': Recall()}, device=device)
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

    #if int(trainer.state.epoch) % 5 == 0:
    #    torch.save(model, 'model/%d'%(trainer.state.epoch))
        
    plot_metrics(metrics)

trainer.run(train_dl, max_epochs=200)

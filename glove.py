from lib import *
from gensim.corpora import Dictionary
from gensim.utils import tokenize
from collections import deque

embed_dim = 300
dictionary_fpath = 'data/dictionary'
word2id_fpath = 'data/word2id'
embed_layer_fpath = 'data/embed_layer'
batch_size = 1024

def build_dictionary():
    train, test = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    df = pd.concat([train, test])
    sentences = df['question_text'].values

    documents = [list(tokenize(sentence)) for sentence in sentences]
    dictionary = Dictionary(documents=documents)

    dictionary.filter_extremes(no_below=2, keep_n=2000000)
    len(dictionary.token2id)
    dictionary.save(dictionary_fpath)

def build_glove_embed():
    glove = pd.read_csv('data/glove.840B.300d/glove.840B.300d.txt', sep=' ')
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

    #set_trace()
    mean, std = np.mean(embeddings, axis=0), np.std(embeddings, axis=0)
    absent_words_embeddings = np.random.normal(loc=mean, scale=std, size=(len(absent_words), embed_dim))
    embeddings = np.concatenate([embeddings,absent_words_embeddings], axis=0)
    print(len(word2id), embeddings.shape)

    embed_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
    torch.save(embed_layer, embed_layer_fpath)

    with open(word2id_fpath, 'w') as f:
        json.dump(dictionary, f)

class CNNGlove(torch.nn.Module):
    def __init__(self, embed_layer=None, use_bn=False, seed=None):
        super().__init__()
        
        self.use_bn = use_bn
        
        if seed is not None:
            torch.manual_seed(seed)

        self.embed_layer = embed_layer
        if embed_layer is None:
            print('initilize embed randomly')
            self.embed_layer = Embedding(num_embeddings=len(dictionary),
                                         embedding_dim=embed_dim,
                                         padding_idx=0)
        
        for p in self.embed_layer.parameters():
            p.require_grad = False
        
        self.convs = []
        for kernel_weight in kernel_weights:
            conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=(kernel_weight, embed_dim)
            )
            self.convs.append(conv)
            setattr(self, 'conv-%d'%(kernel_weight), conv)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.batch_norm_conv = torch.nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm_fc1 = torch.nn.BatchNorm1d(num_features=len(kernel_weights)*out_channels)
        self.batch_norm_fc2 = torch.nn.BatchNorm1d(num_features=int(len(kernel_weights)*out_channels/2))

        self.fc1 = torch.nn.Linear(
            in_features=len(kernel_weights)*out_channels,
            out_features=len(kernel_weights)*out_channels
        )
        
        self.fc2 = torch.nn.Linear(
            in_features=len(kernel_weights)*out_channels,
            out_features=int(len(kernel_weights)*out_channels/2)
        )

        self.fc3 = torch.nn.Linear(
            in_features=int(len(kernel_weights)*out_channels/2),
            out_features=2
        )

        self.relu = torch.nn.Tanh()

    def forward(self, X):
        #set_trace()

        embed = self.embed_layer(X)
        embed = self.dropout(embed)
        embed = torch.unsqueeze(embed, dim=1)
        
        conv_outs = []
        for conv in self.convs:
            # batch_size, out_channels, sentence_length-kernel_weight+1, 1
            conv_out = conv(embed)
            '''
            if self.use_bn:
                conv_out = self.batch_norm(conv_out)
            '''
            # batch_size, out_channels, 1
            # dim=2 is maxed out
            conv_out, _ = torch.max(conv_out, dim=2)
            conv_outs.append(conv_out)
        # batch_size, out_channels*len(kernel_weights), 1
        conv_out = torch.cat(conv_outs, dim=1)
        # batch_size, out_channels*len(kernel_weights)
        #conv_out = self.batch_norm(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.view(X.shape[0], -1)
        
        fc_out = self.fc1(conv_out)
        if self.use_bn:
            fc_out = self.batch_norm_fc1(fc_out)
        fc_out = self.relu(fc_out)
        
        fc_out = self.fc2(fc_out)
        if self.use_bn:
            fc_out = self.batch_norm_fc2(fc_out)

        fc_out = self.relu(fc_out)
        
        fc_out = self.fc3(fc_out)
        logits = fc_out
        
        return logits

class RNNGlove(torch.nn.Module):
    def __init__(self, embed_layer=None, seed=None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.embed_layer = embed_layer
        if embed_layer is None:
            print('initilize embed randomly')
            self.embed_layer = Embedding(num_embeddings=len(dictionary),
                                         embedding_dim=embed_dim,
                                         padding_idx=0)
        
        for p in self.embed_layer.parameters():
            p.require_grad = False
            
        self.dropout = torch.nn.Dropout(p=0.5)
        
        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=1
        )
        
        lstm_dim = embed_dim*1
        
        self.fc1 = torch.nn.Linear(
            in_features=lstm_dim,
            out_features=lstm_dim
        )
        
        self.fc2 = torch.nn.Linear(
            in_features=lstm_dim,
            out_features=int(lstm_dim/2)
        )

        self.fc3 = torch.nn.Linear(
            in_features=int(lstm_dim/2),
            out_features=2
        )

        self.fc_act = torch.nn.Tanh()

    def forward(self, X):
        embed = self.embed_layer(X)
        embed = self.dropout(embed)
        
        output, (h_n, c_n) = self.lstm(embed)
        #lstm_output = h_n.view(X.shape[0], -1)
        lstm_output, _ = torch.max(output, dim=1)
        lstm_output = self.dropout(lstm_output)
        
        fc_out = self.fc1(lstm_output)
        fc_out = self.fc_act(fc_out)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc_act(fc_out)
        logits = self.fc3(fc_out)
        
        return logits

if __name__ == '__main__':
    torch.set_num_threads(8)
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    embed_layer = torch.load(embed_layer_fpath)
    #model = CNNGlove(embed_layer, seed=1234, use_bn=True)
    model = RNNGlove(embed_layer, seed=1234)

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

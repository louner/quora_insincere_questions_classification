from lib import *
from collections import deque

embed_dim = 300
dictionary_fpath = 'data/dictionary'
word2id_fpath = 'data/word2id'
embed_layer_fpath = 'data/embed_layer'
batch_size = 1024

class DPCNNLayer(torch.nn.Module):
        def __init__(self, in_channels=64, out_channels=64):
                    super().__init__()
        self.max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=2)
        self.activation = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
                x = self.max_pool(x)

        convx = self.activation(x)
        convx = self.conv1(convx)

        convx = self.activation(convx)
        convx = self.conv1(convx)

        x = convx + x
        return x

class DPCNNGlove(torch.nn.Module):
        def __init__(self, embed_layer=None, use_bn=False, seed=None, update_emb=False, dpcnn_layers_num=2, cnn_filter_num=64):
                    super().__init__()

        if seed is not None:
                        torch.manual_seed(seed)

        self.embed_layer = embed_layer
        if embed_layer is None:
                        print('initilize embed randomly')
            self.embed_layer = Embedding(num_embeddings=len(dictionary),
                                                             embedding_dim=embed_dim,
                                                                                                      padding_idx=0)

                    if not update_emb:
                                    for p in self.embed_layer.parameters():
                                                        p.require_grad = False

        self.conv1 = torch.nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filter_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = torch.nn.Conv1d(in_channels=cnn_filter_num, out_channels=cnn_filter_num, kernel_size=3, stride=1, padding=1, bias=True)

        self.dpcnn_layers = torch.nn.ModuleList([DPCNNLayer(in_channels=cnn_filter_num, out_channels=cnn_filter_num)for i in range(dpcnn_layers_num)])
        
        self.dropout = torch.nn.Dropout(p=0.5)
        
        self.fc1 = torch.nn.Linear(
                            in_features=31*cnn_filter_num,
                                        out_features=2
                                                )
                
            def forward(self, x):
                        #set_trace()
        x = self.embed_layer(x)
        x = self.dropout(x)
        
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])

        x = self.conv1(x)
        x = self.conv2(x)

        for dpcnn_layer in self.dpcnn_layers:
                        x = dpcnn_layer(x)

        x = self.dropout(x)
        x = x.view(x.shape[0], -1)

        logits = self.fc1(x)

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
    model = DPCNNGlove(embed_layer=embed_layer, use_bn=True, seed=1234, update_emb=True, dpcnn_layers_num=2)

    loss = CrossEntropyLoss()

    torch.set_num_threads(8)
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    optimizer = Adam(params=model.parameters(), lr=learning_rate)

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
        #evaluator.run(val_dl)
        #metrics.add_val(evaluator.state.metrics)

        #if int(trainer.state.epoch) % 5 == 0:
        #    torch.save(model, 'model/%d'%(trainer.state.epoch))

        plot_metrics(metrics)

    trainer.run(train_dl, max_epochs=200)

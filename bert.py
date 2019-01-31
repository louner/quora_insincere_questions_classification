from lib import *

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

class CNNBERT(torch.nn.Module):
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
        
        #self.softmax = torch.nn.Softmax(dim=1)
        
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
        conv_out = self.dropout(conv_out)
        # batch_size, out_channels*len(kernel_weights)
        conv_out = conv_out.view(X.shape[0], -1)
        
        fc_out = self.fc1(conv_out)
        logits = fc_out
        #logits = self.softmax(fc_out)
        
        return logits

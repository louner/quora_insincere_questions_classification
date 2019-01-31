from lib import *
from gensim.corpora import Dictionary
from gensim.utils import tokenize
from collections import deque

embed_dim = 300
dictionary_fpath = 'data/dictionary'
word2id_fpath = 'data/word2id'
embed_layer_fpath = 'data/embed'
batch_size = 1024

def build_dictionary():
    df = pd.read_csv('data/train.csv')
    sentences = df['question_text'].values

    documents = [list(tokenize(sentence)) for sentence in sentences]
    dictionary = Dictionary(documents=documents)

    dictionary.filter_extremes(no_below=5)
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
    return word2id, embeddings

class TokToID(object):
        def __init__(self, dictionary=dictionary):
                    self.dictionary = dictionary

    def toID(self, sentence):
                try:
                                #set_trace()
            ids = [self.dictionary.get(tok, -1) for tok in tokenize(sentence)]
            ids = [id for id in ids if id >= 0]
            return ids
        except:
                        set_trace()

    def __call__(self, item):
                item[0] = self.toID(item[0])
        return item


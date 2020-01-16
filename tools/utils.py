import pickle
import numpy as np
import torch

def writepkl(path,obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def loadpkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    return obj

def creatDict(path_train,path_dev):
    train_sentences = loadpkl(path_train)
    dev_sentences = loadpkl(path_dev)
    sentences = train_sentences + dev_sentences

    print("length of train: ",len(sentences))
    print("sentence example: ",sentences[0])
    word2idx = {}
    idx2word = {}
    index = 1
    for sentence in sentences:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = index
                idx2word[index] = word
                index += 1
    print("total number of words: ",len(word2idx))
    writepkl("../data/pos2idx.pkl",word2idx)
    writepkl("../data/idx2pos.pkl", idx2word)
    print(word2idx)
    print(idx2word)

    print("done")

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r',encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals =  vals[1:]
        word2emb[word] = np.array(vals,dtype=np.float32)
        #print(word2emb[word])
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        #print(type(word2emb[word]))
        weights[idx] = word2emb[word]
    return weights

def initglove(idx2word):
    d = loadpkl(idx2word)
    emb_dim = 300
    glove_file = '../data/glove.6B.%dd.txt' % emb_dim
    weights = create_glove_embedding_init(d, glove_file)
    print(weights.shape)
    np.save('../data/glove6b_init_%dd.npy' % emb_dim, weights)


#initglove("../data/idx2word.pkl")
#creatDict("../data/train_postag.pkl","../data/dev_postag.pkl")

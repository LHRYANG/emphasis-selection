import os
import pickle
import torch
from torch.utils.data import Dataset,DataLoader
from tools.utils import loadpkl
import time
import numpy as np
class SEMDataset(Dataset):
    def __init__(self,name,word2idx,pos2idx,dataroot="data"):
        super(SEMDataset,self).__init__()
        assert name in ['train','dev']

        self.word2idx = loadpkl(word2idx)
        self.pos2idx = loadpkl(pos2idx)

        sentence_path = os.path.join(dataroot,'%s_sentences.pkl' % name)
        self.sentences = loadpkl(sentence_path)
        self.tensor_sentence = [[self.word2idx[word] for word in sentence]  for sentence in self.sentences]
        answer_path = os.path.join(dataroot,'%s_answers.pkl' % name)
        self.answers = loadpkl(answer_path)

        postag_path = os.path.join(dataroot,'%s_postag.pkl' % name)
        self.postag = loadpkl(postag_path)
        self.tensor_postag = [[self.pos2idx[word] for word in sentence] for sentence in self.postag]

        if not (len(self.sentences)==len(self.answers) and len(self.answers)==len(self.postag) and len(self.postag)==len(self.sentences)):
            print("not the right dataset!!")

        graph = loadpkl("./graph/graph.pkl")
        self.pos2id = loadpkl("./graph/pos2id.pkl")
        indicate = loadpkl("./graph/indicate.pkl")
        posid = loadpkl("./graph/posid.pkl")
        length = loadpkl("./graph/length.pkl")
        self.train_distribution = loadpkl("./data/train_distribution.pkl")
        self.test_distribution = loadpkl("./data/test_distribution.pkl")
        if name == 'train':
            self.graph = graph[0:2742]
            self.indicate = indicate[0:2742]
            self.posid = posid[0:2742]
            self.length = length[0:2742]
            self.distribution = self.train_distribution
        else:
            self.graph = graph[2742:]
            self.indicate = indicate[2742:]
            self.posid = posid[2742:]
            self.length = length[2742:]
            self.distribution = self.test_distribution



    def __getitem__(self,index):
        torch.manual_seed(time.time())
        rand = torch.randint(9,(1,))
        s_len = len(self.tensor_sentence[index])
        sentence = torch.zeros(38,dtype=torch.long)
        sentence[0:s_len] = torch.tensor(self.tensor_sentence[index])
        answer = -torch.ones((38),dtype=torch.long)
        answer[0:s_len] = torch.tensor(self.answers[index][rand])

        answers = -torch.ones(9,38)
        answers[:,:s_len] = torch.tensor(self.answers[index])

        postag = torch.zeros(38,dtype=torch.long)
        postag[0:len(self.tensor_postag[index])] = torch.tensor(self.tensor_postag[index])


        mask = torch.zeros(38,3,dtype=torch.float)
        mask[:s_len] = 1

        mask2 = np.zeros((38,38))
        np.fill_diagonal(mask2, 1.0)
        mask2[0:s_len,0:s_len] = 1
        mask2 = torch.tensor(mask2,dtype=torch.float)


        pos_len = int(self.length[index])
        graph = self.graph[index]





        #print("graph",graph.shape)
        for i in range(0,80):
            graph[i][i]=1
        D = np.zeros((80,80))
        hihi = 1/(np.sum(graph,axis=1)+0.1)
        np.fill_diagonal(D,hihi)
        # add this code
        graph = torch.tensor(graph, dtype=torch.float)
        #graph = torch.tensor(np.matmul(D,graph),dtype=torch.float)
        #print(graph)
        indicate = self.indicate[index]
        indi = np.zeros((38),dtype=np.int64)
        indx = np.where(indicate==1)[0].tolist()
        indi[:len(indx)]=indx

        posid = torch.tensor(self.posid[index],dtype=torch.long)

        return sentence, answer, postag, answers, s_len, mask,mask2,pos_len,graph,indi,posid, self.distribution[index] #self.sentences[index],self.answers[index][rand]

    def __len__(self):
        return len(self.sentences)





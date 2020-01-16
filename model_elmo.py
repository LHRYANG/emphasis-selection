import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from tools.utils import loadpkl
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        #self.w_emb = WordEmbedding(ntoken=args.ntoken,dropout=args.dropout)
        self.elmoLayer = ElmoLayer(args.options_file, args.weight_file)
        self.s_emb = SentenceEmbedding(in_dim=args.emb_dim+args.tag_emb_dim,num_hid=args.num_hid,dropout=args.dropout)
        #self.s_emb = SentenceEmbedding(in_dim=args.emb_dim , num_hid=args.num_hid,dropout=args.dropout)
        self.idx2word = loadpkl(args.idx2word)
        self.idx2word[0] = 'PAD'

        self.pos_emb2 = PosEmbedding(npos=args.ntag, emb_dim=args.tag_emb_dim, dropout=args.dropout)
        self.pos_emb = Pos2Embedding(npos=args.npos, emb_dim=args.pos_emb_dim)
        self.pos_emb.load_state_dict(torch.load('pos_emb_500.pkl'))
        self.posrnn = PosRNN()
        self.gc1 = GraphLayer(in_dim=args.pos_emb_dim, out_dim=512)
        self.gc2 = GraphLayer(in_dim=512, out_dim=512)

        #self.attn1 = TradAttn(args.num_hid)
        #self.attn2 = GraphAttn(args.num_hid,args.alpha,args.dropout)
        #self.log_softmax =
        #self.W1 = nn.Linear(args.num_hid*2+512,args.num_hid)
        self.W1 = nn.Linear(args.num_hid * 2+512, args.num_hid)
        self.W2 = nn.Linear(args.num_hid,args.num_class)

        self.wg1 = WordGraph(args.emb_dim, args.wordG_out_dim, dropout=args.dropout)
        self.wg2 = WordGraph(args.wordG_out_dim, args.wordG_out_dim, dropout=args.dropout)

    def forward(self,x,p,mask,pos_x,adj,indicate):
        w = self.elmoLayer(x,self.idx2word)
        w_emb = F.normalize(w, dim=2)
        w_adj = buildG(w_emb, mask)
        wg_emb = F.relu(self.wg1(w_emb, w_adj))
        wg_emb = self.wg2(wg_emb, w_adj)  # [B,s,512/1024]

        # add tag
        p = self.pos_emb2(p)
        w = torch.cat([w, p], dim=-1)
        # add tag
        s = self.s_emb(w)  # [B,max_len,1024]
        # add graph

        pos_w = self.pos_emb(pos_x)  # [B,len,300]
        pos_w = F.normalize(pos_w, dim=2)
        pos_w = F.relu(self.gc1(pos_w, adj))
        pos_w = F.normalize(pos_w, dim=2)
        pos_w = self.gc2(pos_w, adj)  # [B,80,300]
        new_pos_w = torch.zeros((pos_w.shape[0], 38, 512), requires_grad=True).to(device)
        for b in range(len(pos_w)):
            for j in range(38):
                if indicate[b][j] == 0:
                    new_pos_w[b][j] = pos_w[b][79]
                    continue
                new_pos_w[b][j] = pos_w[b][indicate[b][j]]
        #new_pos_w = self.posrnn(new_pos_w)
        #s = torch.cat([s,wg_emb,new_pos_w],dim=2)
        s = torch.cat([s,new_pos_w], dim=2)
        # add graph

        #s = self.attn2(s,mask)
        s = torch.tanh(self.W1(s))#[]
        prob = self.W2(s)
        return prob

class PosRNN(nn.Module):
    def __init__(self,in_dim=512,out_dim=512,bidirect=True,dropout=0.5):
        super(PosRNN,self).__init__()
        self.rnn = nn.GRU(in_dim, out_dim, bidirectional=bidirect, dropout=dropout,batch_first=True)
        self.in_dim = in_dim
        self.num_hid = out_dim
        self.nlayers = 1
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        return weight.new(*hid_shape).zero_()

    def forward(self, x):
        batch = x.size(0)
        hidden = self.init_hidden(batch)  # [B,4,1024]
        a, b = self.rnn(x, hidden)  # a.data:[1444(sum_of_length),2*num_hid],b:[4(bi*2),128,512]
        return a

class Pos2Embedding(nn.Module):
    def __init__(self,npos=74, emb_dim=300, dropout=0.5):
        super(Pos2Embedding, self).__init__()
        self.pos_emb = nn.Embedding(npos, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.pos_emb(x)
        emb = self.dropout(emb)
        return emb
class GraphLayer(nn.Module):
    def __init__(self,in_dim, out_dim,dropout=0.5):
        super(GraphLayer, self).__init__()
        self.Wk = nn.Linear(in_dim,out_dim)
        self.Wv = nn.Linear(in_dim,out_dim)
        self.Wq = nn.Linear(in_dim,out_dim)
        self.head = 3
        self.W = nn.Linear(in_dim,out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x,adj):
        #[B,seq,300],[B,seq,seq]
        key = self.Wk(x)
        value = self.Wv(x)
        query = self.Wq(x)
        weight = torch.matmul(query, torch.transpose(key,1,2)) #[B,seq,seq]
        weight = weight.exp()
        weight = adj*weight
        weight = weight/torch.sum(weight,dim=2,keepdim=True)
        x = torch.matmul(weight,value)
        return x

def buildG(w_emb,mask):
    #[B,s,h]
    adj = torch.matmul(w_emb,torch.transpose(w_emb,1,2))#[B,s,s]
    adj = adj.exp()
    adj = adj*mask
    adj = adj/torch.sum(adj,dim=2,keepdim=True)

    return adj


class WordGraph(nn.Module):
    def __init__(self, in_dim,out_dim,dropout=0.5):
        super(WordGraph, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim)

        self.G = nn.Linear(in_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
    def forward(self, w_emb2,adj):
        w_emb = self.W(w_emb2)  # [B,seq,out_dim]
        aggre = torch.matmul(adj, w_emb)
        w_emb = w_emb + aggre*F.sigmoid(self.G(w_emb2))
        w_emb = self.dropout(w_emb)
        return w_emb

class PosEmbedding(nn.Module):
    def __init__(self,npos=46,emb_dim=100,dropout=0.5):
        super(PosEmbedding, self).__init__()
        self.pos_emb = nn.Embedding(npos,emb_dim,padding_idx=0)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        emb = self.pos_emb(x)
        emb = self.dropout(emb)
        return emb



class SentenceEmbedding(nn.Module):
    def __init__(self, in_dim=300, num_hid=1024, nlayers=2, bidirect=True,dropout=0.5):
        super(SentenceEmbedding, self).__init__()
        self.rnn = nn.GRU(in_dim,num_hid,num_layers=nlayers,bidirectional=bidirect,dropout=dropout,batch_first=True)
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        return weight.new(*hid_shape).zero_()
    def forward(self, x):
        batch = x.size(0)
        hidden = self.init_hidden(batch) #[B,4,1024]
        a,b = self.rnn(x,hidden)#a.data:[1444(sum_of_length),2*num_hid],b:[4(bi*2),128,512]
        return a

class ElmoLayer(nn.Module):
    def __init__(self,options_file, weight_file):
        super(ElmoLayer, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0.3)
    def forward(self,x,idx2words):
        words =[[idx2words[w.cpu().item()] for w in sub_x]for sub_x in x]
        #print(words)
        #print(r_s)
        character_ids = batch_to_ids(words).to(device)

        elmo_output = self.elmo(character_ids)
        elmo_representation = torch.cat(elmo_output['elmo_representations'], -1)

        return elmo_representation


class WordEmbedding(nn.Module):
    def __init__(self,ntoken=5341+1,emb_dim=300, dropout=0.5):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken,emb_dim,padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim
        self.init_embedding()

    def init_embedding(self,np_file="data/glove6b_init_300d.npy"):
        weight_init = torch.from_numpy(np.load(np_file))
        self.emb.weight.data[1:] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb

class TradAttn(nn.Module):
    def __init__(self,nhid):
        super(TradAttn,self).__init__()
        self.nhid = nhid
        self.W = nn.Linear(2*nhid,nhid)
        self.v = nn.Linear(nhid,1)

    def forward(self,hidden, mask):
        # hidden [B,38,nhid*2] mask[B,38,3]

        mask = mask[:,:,0] #[B,38]

        u = self.v(torch.tanh(self.W(hidden)))#[B,38,nhid*2]->[B,38,nhid]->[B,38,1]

        #masked softmax
        u = u.exp()
        u = mask.unsqueeze(2)*u #[B,38,1]
        sums = torch.sum(u,dim=1,keepdim=True)#[B,1,1]
        a = u/sums #[B,38,1]
        z = hidden*a #
        return z

class GraphAttn(nn.Module):

    def __init__(self,nhid,dropout,alpha):
        super(GraphAttn,self).__init__()
        self.nhid = 2*nhid
        self.W = nn.Linear(self.nhid,self.nhid)
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Linear(2*self.nhid,1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,hidden,mask):

        # hidden[B,38,2*nhid]mask[B,38,3]
        h = self.W(hidden)
        N = h.size()[1]
        b = h.size()[0]

        # [B,38,nhid]->[B,38*38,4*nhid]->[B,N,N,2*nhid]
        a_input = torch.cat([h.repeat(1, 1, N).view(b, N * N, -1), h.repeat(1, N, 1)], dim=2).view(b, N, -1,
                                                                                                   2 * self.nhid)
        # e = self.leakyrelu(self.a(a_input).squeeze(3)) #[B,N,N,2*self.nhid]->[B,N,N,1]->[B,N,N]
        e = self.a(a_input).squeeze(3)
        u = e.exp()
        u = mask * u  # [B,38,1]
        sums = torch.sum(u, dim=2, keepdim=True)  # [B,N,1]
        a = u / sums  # [B,38,1]
        h_prime = torch.matmul(a, h)  # [B,N,N]X[B,N,self.nhid]->[B,N,self.nhid]

        return h_prime



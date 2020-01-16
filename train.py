from torch.utils.data import DataLoader
from datasets import SEMDataset
from model_elmo import Model
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as data
from evaluate import eval
import os
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--num_hid', type=int, default=2048)#512,2048(elmo)
    parser.add_argument('--emb_dim', type=int, default=2048)#300,2048(elmo)
    parser.add_argument('--tag_emb_dim', type=int, default=100)
    parser.add_argument('--pos_emb_dim', type=int, default=512)#300,512(elmo)
    parser.add_argument('--wordG_out_dim', type=int, default=1024)  # 512,1024(transformer)
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--idx2word', type=str, default='data/idx2word.pkl')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ntoken', type=int, default=5342)
    parser.add_argument('--ntag', type=int, default=46)
    parser.add_argument('--npos', type=int, default=74)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--options_file', type=str, default="https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json")
    parser.add_argument('--weight_file', type=str, default="https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--model_name', type=str, default="model_elmo", help='saved models name')
    args = parser.parse_args()
    return args

def train(model,train_loader,dev_loader,args):
    optim = torch.optim.Adam(model.parameters(),lr=0.0001)
    log_softmax = nn.LogSoftmax(dim=2).to(device)
    div_loss = nn.KLDivLoss(reduction='none')
    softmax = nn.Softmax(dim=2).to(device)
    best_socre = 0
    best_epoch = -1
    for epoch in range(args.epochs):
        train_tq = tqdm(train_loader,desc='E{:03d}'.format(epoch),ncols=0)
        dev_tq = tqdm(dev_loader,desc='T{:03d}'.format(epoch),ncols=0)
        total_loss = 0
        model.eval()
        matchm,rankm ,score , acc, o_acc, b_acc,i_acc = evaluate(model, dev_tq,softmax,epoch)
        print(rankm,acc, o_acc, b_acc,i_acc)
        if score>best_socre:
            best_socre = score
            best_epoch = epoch
            best_match = matchm
            torch.save(model.state_dict(),args.model_name+'.pkl')

        model.train()
        for (s,a,p,a_s,s_len,mask,mask2,pos_len,graph,indicate,posid,distribution) in train_tq:
            s = s.to(device) #[B, 38]
            a = a.to(device) #[B, 38]
            p = p.to(device) #[B,38]
            a_s = a_s.to(device)#[B,9,38]
            s_len = s_len.to(device)#[B]
            mask = mask.to(device)
            mask2 = mask2.to(device)
            pos_len = pos_len.to(device)
            graph = graph.to(device)
            #indicate = indicate.to(device)
            posid = posid.to(device)
            distribution = distribution.to(device)

            prob = model(s,p,mask2,posid,graph,indicate) #[B,38,3]
            prob = log_softmax(prob)


            # div loss
            #distribution = distribution/(torch.sum(distribution,dim=2,keepdim=True)+0.00001)
            #div_l = div_loss(prob,distribution)
            _, p_max = prob.max(dim=2)

            #total = s_len.sum()
            #loss = div_l.sum()/total

            weight = torch.ones(prob.shape[0],38,3).to(device)
            weight[:,:,1:3]=5
            clamp_a = torch.clamp(a,0,3)
            #-----------------------
            #calculate loss
            new_a = torch.unsqueeze(clamp_a,-1)
            new_b = torch.zeros(prob.shape[0],38,args.num_class).to(device)
            new_b.scatter_(2, new_a, 1)
            total = s_len.sum()
            loss = (new_b*mask*prob).sum()
            loss = -loss/total
            #------------------------


            total_loss += loss.item()*total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optim.step()
            optim.zero_grad()
            fmt = '{:.4f}'.format
            train_tq.set_postfix(loss=fmt(loss.item()))
        print("total loss:",total_loss)

    print(best_socre)
    print(best_match)
    print(best_epoch)

def evaluate(model,dev_tq,softmax,epoch):

    total_o = 0
    total_b = 0
    total_i = 0
    o_acc = 0
    b_acc = 0
    i_acc = 0
    acc = 0
    total = 0
    with torch.no_grad():
        for (s,a,p,a_s,s_len,mask,mask2,pos_len,graph,indicate,posid,distribution) in dev_tq:

            s = s.to(device)  # [B, 38]
            p = p.to(device)
            #mask = mask.to(device)
            mask2 = mask2.to(device)

            pos_len = pos_len.to(device) #[64]
            graph = graph.to(device)    #[64,80,80]
            #indicate = indicate.to(device)#[64,80]
            posid = posid.to(device) #[64,80]

            prob = model(s,p,mask2,posid,graph,indicate) #[dev_set,38,3]
            prob = softmax(prob)
            prob = prob.cpu()
            ac, o, b, i, num_o, num_b, num_i, num = accuracy(prob,distribution,mask[:,:,0])
            acc+=ac
            o_acc+=o
            b_acc+=b
            i_acc+=i
            total_o+=num_o
            total_b+=num_b
            total_i+=num_i
            total += num
            prob = prob.numpy() #[B,38,3]

            #print(prob)
            with open("reslut"+str(epoch)+".txt",'a') as f:
                for i in range(prob.shape[0]):
                    for j in range(s_len[i]):
                        f.write('a \t b \t' +str(1-prob[i][j][0])+'\n')
                    f.write('\n')

    matchm,rankm = eval("result"+str(epoch)+".txt","train_dev_data/dev.txt")
    return matchm,rankm,sum(matchm.values()),acc/total, o_acc/total_o, b_acc/total_b, i_acc/total_i

def accuracy(prob,label,mask):
    #prob [B,38,3] label [B,38,3],mask:[B,38]

    prob_max = torch.argmax(prob,dim=2)
    label_max = torch.argmax(label,dim=2)
    acc = torch.sum((prob_max==label_max).float()*mask)
    o_prob = prob_max == 0
    o_label = label_max == 0
    b_prob = prob_max == 1
    b_label = label_max == 1
    i_prob = prob_max == 2
    i_label = label_max == 2
    o_acc = torch.sum((o_prob==o_label).float()*mask*o_label.float())
    b_acc = torch.sum((b_prob==b_label).float()*mask*b_label.float())
    i_acc = torch.sum((i_prob==i_label).float()*mask*i_label.float())
    total_o = torch.sum(o_label.float()*mask)
    total_b = torch.sum(b_label.float()*mask)
    total_i = torch.sum(i_label.float()*mask)
    return acc, o_acc, b_acc, i_acc, total_o,total_b,total_i,torch.sum(mask)


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)

    train_set = SEMDataset("train",'data/word2idx.pkl','data/pos2idx.pkl')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    dev_set = SEMDataset('dev','data/word2idx.pkl','data/pos2idx.pkl')
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    m = Model(args).to(device)

    #m = nn.DataParallel(m).to(device)
    train(m,train_loader,dev_loader,args)



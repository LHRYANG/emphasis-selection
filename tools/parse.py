from nltk.parse.corenlp import CoreNLPParser
import nltk
import numpy as np
from utils import loadpkl
from utils import writepkl


def getNodes(parent,index):
    global count, count4
    if parent.label() not in pos2id:
        pos2id[parent.label()] = count4
        count4 += 1
    sub_graph[index][index] = 1
    subposid[index] = pos2id[parent.label()]
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == ROOT:
                pass
                #print("======== Sentence =========")
                #print("Sentence:", " ".join(node.leaves()))
                #global count
                #count = 0
            else:
                #print("Label:", node.label())
                #print("Leaves:", node.leaves())

                count = count + 1
                #id2pos[count] = node.label()
                sub_graph[index][count]=1
                sub_graph[count][index]=1
                if node.height() == 2:
                    sub_indicate[count] = 1

            getNodes(node,count)
        else:
            #print("Word:", node)
            pass



'''
print('---------------------------')
getNodes(tree,0)
print(id2pos)
print(graph[0:count+1,0:count+1])
print(parse)
'''


if __name__ == '__main__':

    parser = CoreNLPParser()

    train_s = loadpkl("../data/train_sentences.pkl")
    test_s  = loadpkl("../data/dev_sentences.pkl")

    all_s = train_s+test_s

    count2 = 0
    count3 = 0
    max_len = 0
    count4 = 2
    pos2id = {'ROOT':1}
    graph = np.zeros((len(all_s), 80, 80))
    indicate = np.zeros((len(all_s), 80))
    posid = np.zeros((len(all_s),80))
    length = np.zeros((len(all_s)))
    for ha, s in enumerate(all_s):
        str_s = ' '.join(s)
        #print(str_s)
        sub_graph = np.zeros((80, 80))
        sub_indicate = np.zeros((80))
        subposid = np.zeros((80))
        parse = next(parser.raw_parse(str_s))
        ROOT = 'ROOT'
        tree = parse


        count = 0
        getNodes(tree,0)
        print(s)
        print(tree)
        print(subposid)
        print(sub_indicate)
        print(sub_graph[0:count + 1, 0:count + 1])
        break
        length[ha] = count
        if ha % 100 == 0:
            print(ha)
        if len(tree.leaves())!=len(s) or tree.leaves()!=s:
            continue
        graph[ha]=sub_graph
        indicate[ha]=sub_indicate
        posid[ha]=subposid
    writepkl("../graph/length.pkl", length)
    writepkl("../graph/graph.pkl",graph)
    writepkl('../graph/pos2id.pkl',pos2id)
    writepkl("../graph/indicate.pkl", indicate)
    writepkl('../graph/posid.pkl', posid)

'''
if __name__ == '__main__':

    parser = CoreNLPParser()
    parse = next(parser.raw_parse("I love playing football"))
    print(parse)
'''
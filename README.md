This is the pytorch implementation of "Sentence structure and word relationship modeling for emphasis selection". 
## Requirement
- pytorch
- standfordnlp and nltk 
- pretrained elmo 
- java 

## How to run 
1. download **stanford-corenlp-full-2018-10-05** and unzip the file 
2. cd tools <br />
   run **preprocess.py**  run twice, change the directory for train and dev data respectively. 
   After this step you will get 
   - train(dev)\_answers.pkl
   - train(dev)\_postag.pkl 
   - train(dev)\_sentences.pkl 
2. If you want to use pretrained glove embedding dowoload **glove6B.300d.txt** to data directory 
3. run **createDict funtion in util.py**  twice to create POS vocabulary and word vocabulary for train and dev. 
   after this step, you will get
   - idx2pos.pkl
   - pos2idx.pkl 
   - word2idx.pkl
   - idx2word.pkl 
6. create tag graph <br />
   first run **parser_server.py** to make the parser run in background then,  
   run **parse.py** to create tag graph (needs java and standfordnlp)
   you will get 
   - length.pkl 
   - graph.pkl
   - pos2id.pkl
   - indicate.pkl
   - posid.pkl
7. download pretrained elmo model
8. run train.py 

in datasets.py, it need the train/dev_distribution.pkl, you can comment the corresponding line  or you can use the data provided in data/
The model does actually not use this data.
   

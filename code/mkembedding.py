import numpy as np
import itertools
import torch
import pandas as pd
from torch import nn
from torch.autograd import Variable
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
import gensim
def rd_split(text):
    text=str(text)
    text=text.split()
    return text


train_df=pd.read_csv("../dataset/with_stop.csv")
for q in ['q1', 'q2']:
    train_df[q + '_n'] = train_df[q]
vocabs={}
vocabs_cnt=0
word2vec=KeyedVectors.load_word2vec_format("../dataset/glove_word2vec.txt")
for i,row in train_df.iterrows():
    if i != 0 and i % 1000 == 0:
        print(i)
    for q in ['q1','q2']:
        q_idx=[]
        for word in rd_split(row[q]):
            if word not in vocabs:
                vocabs_cnt+=1
                vocabs[word]=vocabs_cnt
        q_idx.append(vocabs[word])
    train_df.at[i,q+'_n']=q_idx
embedding=1*np.random.randn(len(vocabs)+1,300)
embedding[0]=0

for word,i in vocabs.items():
    if word in word2vec.vocab:
        embedding[i]=word2vec.word_vec(word)
#embedding索引文件
np.save("embedded_idxed.npy",embedding)
#q1_n存的是每个单词在上面np中的索引值
train_df.to_csv("with_stop_idxed.csv")



            
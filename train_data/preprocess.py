import re
import gensim
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import gensim
import itertools
import pandas as pd 
def clean_split(text):
    text=str(text)
    text=text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    return text
    
TRAIN_FILE="train.csv"
train_df=pd.read_csv(TRAIN_FILE)



stops = set(stopwords.words('english'))
qlist=[]
tp1=[]
tp2=[]
qlist.append(tp1)
qlist.append(tp2)
for idx, row in train_df.iterrows():
    for i,q in enumerate(['question1','question2']):
        processed_list=[]
        for word in clean_split(row[q]):
            if word in stops:
                continue
            processed_list.append(word)
        processed_str=' '.join(processed_list)
        qlist[i].append(processed_str)
s_1=pd.Series(qlist[0])
s_2=pd.Series(qlist[1])

train_df=train_df.assign(q1=s_1)
train_df=train_df.assign(q2=s_2)

train_df.to_csv("no_stop.csv")


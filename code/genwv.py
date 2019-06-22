from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim


glove_file = datapath('D:/学习相关/大三下/机器学习与数据挖掘/作业2/dataset/glove.840B.300d.txt')
tmp_file = get_tmpfile("D:/学习相关/大三下/机器学习与数据挖掘/作业2/dataset/glove_word2vec.txt")
glove2word2vec(glove_file, tmp_file)

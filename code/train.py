import torch
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
matplotlib.use('Agg')
hidden_size=50
num_layers=1
train_df=pd.read_csv('../train_data/with_stop_idxed.csv')
np_embedding=np.load('../train_data/embedded_idxed.npy').astype(np.float32)
text_sz=np_embedding.size
print(text_sz)
class SeLSTM(nn.Module):
    def __init__(self,np_embedding):
        super().__init__()
        self.embedding=nn.Embedding.from_pretrained(torch.from_numpy(np_embedding))
        self.lstm=nn.LSTM(300,hidden_size,num_layers,batch_first=True);
    def distance(self,q1,q2):
        return torch.exp(-torch.sum(torch.abs(q1 - q2), dim=0))
    def once(self,q,q_lenlist):
        sorted_indices = np.flipud(np.argsort(q_lenlist))
        q_lenlist = np.flipud(np.sort(q_lenlist))
        q_lenlist = q_lenlist.copy()
        ordered_questions = [torch.LongTensor(q[i]) for i in sorted_indices]
        ordered_questions = torch.nn.utils.rnn.pad_sequence(ordered_questions, batch_first=True)
        embeddings = self.embedding(ordered_questions)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, q_lenlist, batch_first=True)
        out, (hn, cn) = self.lstm(packed)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=int(q_lenlist[0]))
        result = torch.FloatTensor(unpacked.size())
        for i, encoded_matrix in enumerate(unpacked):
            result[sorted_indices[i]] = encoded_matrix
        return result
    def forward(self,q1,q1_lenlist,q2,q2_lenlist):
        left=self.once(q1,q1_lenlist)
        right=self.once(q2,q2_lenlist)
        score = torch.zeros(left.size()[0])
        #因为第batch_first所以是b*t*s, i是batch号，lenlist[i]-1才是行号...
        for i in range(left.size()[0]):
            res1 = left[i, q1_lenlist[i] - 1, :]
            res2 = right[i, q2_lenlist[i] - 1, :]
            score[i] = self.distance(res1, res2)
        return score

model=SeLSTM(np_embedding)
Y = train_df['is_duplicate']#.as_matrix().astype(np.int32)
X = train_df[['q1_n', 'q2_n']]#.astype(str).values
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=int(0.1*len(train_df)))
X_train = X_train.astype(str).values
X_validation = X_validation.astype(str).values
Y_train = Y_train.as_matrix().astype(np.int32)
Y_validation = Y_validation.as_matrix().astype(np.int32)
for i in range(X_train.shape[0]):
    for j in range(2):
        X_train[i,j]=np.array(eval(X_train[i,j]))
for i in range(X_validation.shape[0]):
    for j in range(2):
        X_validation[i,j]=np.array(eval(X_validation[i,j]))

class pair_dataset(Dataset):
    def __init__(self,pairs,labels):
        self.datapairs=pairs
        self.labels=labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        llen=len(self.datapairs[index][0])
        rlen=len(self.datapairs[index][1])
        return self.datapairs[index][0],llen,self.datapairs[index][1],rlen,self.labels[index]

class Collate:
    def collate(self, batch):
        # batch = list of tuples where each tuple is of the form ([i1, i2, i3], [j1, j2, j3], label)
        q1_list = []
        q1_len=[]
        q2_list = []
        q2_len=[]
        labels=[]
        for example in batch:
            q1_list.append(example[0])
            q1_len.append(example[1])
            q2_list.append(example[2])
            q2_len.append(example[3])
            labels.append(example[4])

        return q1_list, q1_len, q2_list, q2_len, labels

    def __call__(self, batch):
        return self.collate(batch)


train_ds=pair_dataset(X_train,Y_train)
valid_ds=pair_dataset(X_validation,Y_validation)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01 )
num_epochs = 50
train_loader=DataLoader(train_ds,batch_size=100,shuffle=True,collate_fn=Collate())
valid_loader=DataLoader(valid_ds,batch_size=300,collate_fn=Collate())
threshold = torch.torch.FloatTensor([0.5])
total_step = len(train_loader)
best=0
for epoch in range(num_epochs):
    model.train(True)
    corr_num=0
    loss_history=[]
    for i,(left,llen,right,rlen,y) in enumerate(train_loader):
        Labels=torch.FloatTensor(y)
        optimizer.zero_grad()
        score=model(left,llen,right,rlen)
        predictions = (score > threshold).float() * 1
        total = Labels.size()[0]    
        correct = (predictions == Labels).sum().item()
        corr_num += correct
        loss = criterion(score, Labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            loss_history.append(loss.item())
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, np.mean(loss_history), (correct / total) * 100))
    model.train(False)
    valid_corr=0
    valid_history=[]
    with torch.no_grad():
        for left,llen,right,rlen,y in valid_ds:
            Labels=torch.FloatTensor(y)
            score=model(left,llen,right,rlen)
            predictions=(score>threshold).float()*1
            total=Labels.size()[0]
            correct = (predictions == Labels).sum().item()
            corr_num += correct
            loss = criterion(score, Labels)
            loss.backward()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, np.mean(valid_history), (correct / total) * 100))

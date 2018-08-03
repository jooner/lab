import string

import numpy as np
from tqdm import tqdm, tnrange, tqdm_notebook
from pandas import read_csv
from nltk import word_tokenize

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

whitepapers = read_csv('whitepapers.csv')
# merge description and document text
whitepapers['text'] = whitepapers.description + ' ' + whitepapers.document_text
# filter down to relevant entries
df = whitepapers.drop(columns=['description', 'document_text', 'document_tokens'])
del whitepapers

# tokenize (aka .split()++, thank you nltk)
train_txt = ''
for _, row in df.iterrows():
    train_txt += row['text'].lower()
tokens = word_tokenize(train_txt)
del df

# word2idx and idx2word setup
unique_tokens = set(tokens)
w2x = {word: idx for (idx, word) in enumerate(unique_tokens)}
x2w = {idx: word for (idx, word) in enumerate(unique_tokens)}
indices = [w2x[w] for w in tokens]

# generate training data
window = 2
train_data = []
for idx in range(len(indices)):
    for r in range(-window, window + 1):
        cxt = idx + r
        if not ((cxt < 0) or (cxt >= len(indices)) or (idx == cxt)):
            train_data.append([indices[idx], indices[cxt]])
train_data = np.array(train_data)
train_data = torch.LongTensor(train_data)

# record vocab_size
vocab_size = len(unique_tokens)
# sanity check
for [x,y] in train_data[200100:200105]:
    print(x2w[int(x)], x2w[int(y)])
# clean memory
del indices
del tokens

# Continuous Bag-of-Words Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 context_size, batch_size):
        super(CBOW, self).__init__()
        self.batch_size = batch_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.out = nn.Softmax(dim=2)
    
    def forward(self, x):
        x = self.embed(x).view(self.batch_size, 1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.out(x).squeeze()

model = CBOW(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, context_size=2, batch_size=256).cuda()

def one_hot(idx_batch):
    one_hot_mat = torch.zeros((len(idx_batch), vocab_size)).float()
    indices = torch.LongTensor(idx_batch).view(-1, 1)
    one_hot_mat.scatter_(1, indices, 1.0)
    return one_hot_mat

def mat_loss(pred, gt):
    delta = pred.float() - gt.float()
    norm = torch.norm(delta, p=2, dim=1)
    return (torch.sum(norm) / gt.shape[1])

def batchify(data, batch_size, use_cuda=False):
    rm_size = len(data) % batch_size
    x, y = data[:-rm_size, 0], data[:-rm_size, 1]
    if use_cuda:
        x = x.view(-1, batch_size).cuda()
    else:
        x = x.view(-1, batch_size)
    y = y.view(-1, batch_size)
    return x, y

x, y = batchify(train_data, batch_size=256, use_cuda=True)

def train(x_train, y_train, num_epochs, use_cuda=False):
    loss_fn = mat_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.5)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx in tqdm(range(x_train.shape[0])):
            x = x_train[batch_idx, :]
            y = y_train[batch_idx, :]
            model.zero_grad()
            log_prob = model(x)
            gt = one_hot(y)
            if use_cuda:
                gt = gt.cuda()
            loss = loss_fn(log_prob, gt)
            loss.backward()
            scheduler.step()
            total_loss += loss.data
        print(total_loss)
        torch.save(model, 'models/model_{}'.format(total_loss))
        print("Successfully Saved model_{}!".format(total_loss))

train(x, y, num_epochs=100, use_cuda=True)


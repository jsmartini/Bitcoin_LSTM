import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from progress.bar import Bar

dataset = pd.read_csv("BTC_USD_2013-09-30_2021-03-03-CoinDesk.csv")

close = dataset["Closing Price (USD)"]


close.iloc[:2500].plot()
close.iloc[2500:].plot()
#plt.show()

train = list(close.iloc[:2500].items())
test = list(close.iloc[2500:].items())
sc = MinMaxScaler(feature_range=(0,1))
train_scale = sc.fit_transform(np.array(train))

ts = 60
X, Y = [], []
print(train)
for i in range(ts, 2500):
    X.append(train[i-ts:i][:])
    Y.append(train[i][1])

X = np.array(X)
Y = np.array(Y)
X = np.reshape(X, (X.shape[0], 2*X.shape[1], 1))
X = torch.Tensor(X)
Y = torch.Tensor(Y)
X_test, Y_test = [], []
for i in range(ts, len(test)):
    X_test.append(test[i-ts:i][:])
    Y_test.append(train[i][1])

X_test = torch.Tensor(X_test)
X_test = X_test.reshape(X_test.shape[0], 2*X_test.shape[1], 1)
print(X_test.shape[1:])
print(X_test.shape)
print(X.shape)
Y_test = torch.Tensor(Y_test)
class btcnet(nn.Module):

    def __init__(self, input_sz, h_sz, layer_num):
        super(btcnet, self).__init__()
        self.h_sz = h_sz
        self.layer_num = layer_num
        self.lstm1 = nn.LSTM(
            input_size=input_sz,
            hidden_size=h_sz,
            num_layers=layer_num,
            dropout=0.5
        )
        #self.bn1 = nn.BatchNorm1d(num_features=)
        self.lin1 = nn.Linear(self.h_sz,100)
        self.linear = nn.Linear(100, 1)
        self.h = (torch.zeros(layer_num,1,self.h_sz), torch.zeros(layer_num,1,self.h_sz))

    def forward(self, x):
        x, self.h = self.lstm1(x, self.h)
        x = self.lin1(x)
        x = self.linear(x)
        return x

BTCNET = btcnet(X.shape[1], 100, 10)
lossF = nn.MSELoss()
optimizer = optim.SGD(BTCNET.parameters(), lr=0.1)
print(BTCNET)

for epoch in range(2):
    print(f"EPOCH {epoch}/20")
    with Bar("Processing", max = X.shape[0]) as bar:
        for example, ans in zip(X, Y):
            bar.next()
            score= BTCNET(example.view(1,1,-1))
            loss = lossF(score, ans)
            loss.backward(retain_graph=True)
        bar.finish()
    optimizer.step()
err = []
test_loss = nn.MSELoss()
with torch.no_grad():
    with Bar("Processing", max=X_test.shape[0]) as bar:
        for test_ex, ans in zip(X_test, Y_test):
            bar.next()
            score= BTCNET(test_ex.view(1,1))
            err.append(test_loss(score, ans))
            print(f"Ans: {ans} Score: {score}")
    bar.finish()

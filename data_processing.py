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
for i in range(ts, 2500):
    X.append(train[i-ts:i][:])
    Y.append(train[i][0])
X = np.array(X)
Y = np.array(Y)
X = np.reshape(X, (X.shape[0], 2*X.shape[1], 1))
X = torch.Tensor(X)
Y = torch.Tensor(Y)
class btcnet(nn.Module):

    def __init__(self, input_sz):
        super(btcnet, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_sz,
            hidden_size=50,
            num_layers=5,
            bias = True,
            dropout=0.5
        )
        self.linear = nn.Linear(50, 1)


    def forward(self, x):
        x, h = self.lstm1(x)
        return self.linear(x), h

BTCNET = btcnet(X.shape[1])
lossF = nn.MSELoss()
optimizer = optim.SGD(BTCNET.parameters(), lr=0.1)
print(BTCNET)
for epoch in range(20):
    print(f"EPOCH {epoch}/20")
    with Bar("Processing", max = X.shape[0]) as bar:
        for example, ans in zip(X, Y):
            bar.next()
            score, out = BTCNET(example.view(1, 1,-1))
            loss = lossF(score.view(1,1), ans)
            loss.backward()
            optimizer.step()
        bar.finish()
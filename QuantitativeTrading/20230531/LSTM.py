# @Version: python3.10
# @Time: 2023/5/31 16:29
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: LSTM.py
# @Software: PyCharm
# @User: chent

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time

data = ts.get_k_data('000906', start='2020-01-01', end='2021-01-01')

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(data[['close']])
plt.xticks(range(0, data.shape[0], 20), data['date'].loc[::20], rotation=45)
plt.title("000906 Stock Price", fontsize=18, fontweight='bold')
plt.xlabel("date", fontsize=18)
plt.ylabel("Close Price (USD)", fontsize=18)
plt.show()

# 1.特征工程
print("特征工程")
price = data[['close']]
print(price.info())

scaler = MinMaxScaler(feature_range=(-1, 1))
price['close'] = scaler.fit_transform(price['close'].values.reshape(-1, 1))

# 2.数据集制作
print("数据集制作")
def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []

    # you can free play (seq_length)
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[: train_set_size, : -1, :]
    y_train = data[: train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]

lookback = 20
x_train, y_train, x_test, y_test = split_data(price, lookback)
print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)

# 3.模型构建——LSTM
print("模型构建——LSTM")
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# 模型训练
print("模型训练")
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE:", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time() - start_time
print("Training time: {}".format(training_time))


# 5.模型结果可视化
predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

sns.set_style("darkgrid")

fig = plt.figure()
fig.subplots_adjust(hspace=0.8, wspace=0.2)

# plt.subplot(1, 2, 1)
ax = sns.lineplot(x=original.index, y=original[0], label="Data", color="royalblue")
ax = sns.lineplot(x=predict.index, y=predict[0], label="Train Prediction(LSTM)", color="tomato")
ax.set_title("Stock price", size=14, fontweight="bold")
ax.set_xlabel("Days", size=14)
ax.set_ylabel("Cost (USD)", size=14)
ax.set_xticklabels("", size=10)
plt.show()

# plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color="royalblue")
ax.set_xlabel("Epoch", size=14)
ax.set_ylabel("Loss", size=14)
ax.set_title("Train Loss", size=14, fontweight="bold")
fig.set_figheight(6)
fig.set_figwidth(6)
plt.show()


print("finish")


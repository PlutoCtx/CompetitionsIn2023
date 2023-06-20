# @Version: python3.10
# @Time: 2023/6/18 19:01
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: Lstm.py
# @Software: PyCharm
# @User: chent

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

# 加载股票数据
data = pd.read_csv('stock_data.csv')

# 提取收盘价作为目标变量
close_prices = data['close'].values.reshape(-1, 1)

# 数据预处理：归一化处理
scaler = MinMaxScaler()
scaled_close_prices = scaler.fit_transform(close_prices)

# 划分训练集和测试集
train_size = int(len(scaled_close_prices) * 0.8)
train_data = scaled_close_prices[:train_size]
test_data = scaled_close_prices[train_size:]

def create_dataset(dataset, time_steps=1):
    Xs, ys = [], []
    for i in range(len(dataset) - time_steps):
        Xs.append(dataset[i:(i+time_steps), 0])
        ys.append(dataset[i+time_steps, 0])
    return np.array(Xs), np.array(ys)

# 构建训练集和测试集的特征和标签
time_steps = 30  # 时间步长，表示前几天的数据用于预测下一天的价格
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# 转换为3D形状 [样本数，时间步长，特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建LSTM模型
model = Sequential()
# model.add(LSTM(units=50, input_shape=(time_steps, 1)))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # 模型训练
# model.fit(X_train, y_train, epochs=20, batch_size=32)
#
# # 模型预测
# predicted_prices = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)
#
# # 反归一化处理
# y_test = scaler.inverse_transform([y_test])
#
# # 输出预测结果和实际值
# print('Predicted prices:', predicted_prices.flatten())
# print('Actual prices:', y_test.flatten())

print(12)

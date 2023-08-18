# @Version: python3.10
# @Time: 2023/6/18 19:01
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: Lstm.py
# @Software: PyCharm
# @User: chent

# # 创建一个新的图形窗口
# plt.figure()
#
# # 2444条数据
# # 绘制第一条线（蓝色实线、宽度为1.5）
# plt.plot(x_axis, predict_l, color='blue', linestyle='-', linewidth=1.5, label='predicted')
#
# # 绘制第二条线（红色虚线、宽度为1）
# plt.plot(x_axis, actual_l, color='red', linestyle='--', linewidth=1, label='actual')
#
# # 添加标题和标签
# plt.title('Comparison of Two Lines')
# plt.xlabel('date')
# plt.ylabel('actual value')
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()
# # print(len(predict_l))
# # print(len(actual_l))
# # print(len(predicted_prices))
# # print(len(y_test))




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def prepare_data(data):
    # 特征数据和目标数据
    X = []
    y = []
    for i in range(len(data)-30):
        X.append(data[i:i+30])
        y.append(data[i+30])
    return np.array(X), np.array(y)

# 加载历史股票数据（示例代码）
# stock_data = load_data()

# 加载股票数据
data = pd.read_csv('stock_data.csv')

# 提取收盘价作为目标变量
stock_data = data['close'].values.reshape(-1, 1)

# 数据处理与特征工程
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data)  # 标准化处理

# 准备训练数据
X_train, y_train = prepare_data(scaled_data)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(30, 5)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练LSTM模型
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 预测未来一段时间的股票数据
predicted_data = []
last_30_days = [scaled_data[-30:]]
for i in range(30):  # 预测未来30天的数据
    next_day = model.predict(np.array(last_30_days))[0]
    predicted_data.append(next_day)
    last_30_days = np.append(last_30_days[0][1:], next_day)
    last_30_days = np.reshape(last_30_days, (1, 30, 5))

# 还原预测结果
predicted_data = scaler.inverse_transform(predicted_data)

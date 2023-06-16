# # @Version: python3.10
# # @Time: 2023/4/6 10:11
# # @Author: MaxBrooks
# # @Email: 15905898514@163.com
# # @File: Multi-factor_stock_picking.py
# # @Software: PyCharm
# # @User: chent
#
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import tushare as ts

# # 选取股票代码为000906的股票从2018年11月2日到2021年7月2日的数据
# data = ts.get_k_data('000906', start='2010-01-01', end='2022-12-31')
# # 将数据保存到指定路径
# # 带有日期的数据建议储存到csv格式而不是excel格式，因为会乱码
# data.to_csv(r'000906.csv')


# 将日期设置为index，并且转化为datatime格式
data = pd.read_csv(r'000906.csv', index_col='date', parse_dates=['date'])
data = data.drop(columns='code')
#
# # 其实大概检测只要看这三个数据,异常：超过1000的请重新选择数据
# print(data.describe().loc[['min', 'max', 'mean'], :])
#
# # 数据特征处理
# base1 = data['close']
# base2 = data.shift(1)
# base2.columns = [f'v{j}-1' for j in range(len(data.columns))]
# base3 = pd.concat([base1, base2], axis=1)
# for i in range(2, 7):
#     base4 = data.shift(i)
#     base4.columns = [f'v{j}-{i}' for j in range(len(data.columns))]
#     base3 = pd.concat([base3, base4], axis=1)
# base3.dropna(inplace=True)
# x = base3.drop(columns=['close'])
# y = base3['close']
#
#
# # 数据分割 8: 2
# x_train, x_test = x.iloc[:int(len(x)*0.8), :], x.iloc[int(len(x)*0.8):, :]
# y_train, y_test = y[:int(len(x)*0.8)], y[int(len(x)*0.8):]
#
# # 数据特征处理
# model1 = XGBRegressor()
# rfa = RFECV(model1, cv=5, scoring='neg_mean_absolute_error')
# rfa.fit(x_train, y_train)
# # 特征权重数据可视化
# # plt.bar(x_train.columns, rfa.grid_scores_)
# # plt.show()
#
# # 特征选择
# col_select = x_train.columns[rfa.support_]
# x_train, x_test = x_train.loc[:, col_select], x_test.loc[:, col_select]
#
# # 数据标准化
# std = StandardScaler()
# std.fit(x_train)
# x_train_std, x_test_std = std.transform(x_train), std.transform(x_test)
#
# # 模型调参
# params = dict(gamma=[0.1, 0.5, 0.7, 0.05],
#             min_child_weight=[1, 3, 5],
#             subsample=[0.2, 0.4, 0.7],
#             colsample_bytree=[0.2, 0.4, 0.7],
#             n_estimators=[100, 200, 300],
#             max_depth=[3, 4, 5])
# model = GridSearchCV(XGBRegressor(), params, scoring='neg_mean_squared_error', n_jobs=5, cv=5)
#
# # 模型拟合
# model.fit(x_train_std, y_train)
# # print(model.best_params_)
# print("123456")
#
# # 结果预测
# plt.plot(y_test.index, y_test, label='True')
# y_pred = (model.predict(x_test_std))
# plt.plot(y_test.index, y_pred, label='pred')
# plt.title('MAE:%.2f' % (mean_absolute_error(y_test, y_pred)))
# gca = plt.gca()
# mul = plt.MultipleLocator(45)
# gca.xaxis.set_major_locator(mul)
# plt.legend()
# plt.show()
#
#
# # 导入模型
# model1 = RandomForestRegressor()
# model2 = GradientBoostingRegressor()
# model3 = XGBRegressor()
# model4 = StackingRegressor([('m1', model1), ('m2', model3)], cv=5, n_jobs=10)
# model4.fit(x_train_std, y_train)
# # 结果预测
# plt.plot(y_test.index, y_test, label='True')
# y_pred = model4.predict(x_test_std)
# plt.plot(y_test.index, y_pred, label='pred')
# plt.title('MAE:%.2f' % (mean_absolute_error(y_test, y_pred)))
# gca = plt.gca()
# mul = plt.MultipleLocator(45)
# gca.xaxis.set_major_locator(mul)
# plt.legend()
#
# # 置信区间
# num = np.random.normal(loc=y_test.values, scale=np.ones(len(y_test))*0.1, size=(1000, len(y_test)))
# l, u = st.t.interval(0.95, len(y_test)-1, loc=np.mean(num, axis=0), scale=np.std(num, axis=0))
# plt.fill_between(y_test.index, l, u, alpha=0.4, color='r')
# plt.show()
#
# joblib.dump(model, "GBDT.model")
#

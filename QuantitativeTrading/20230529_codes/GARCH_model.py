# @Version: python3.10
# @Time: 2023/5/29 14:55
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: GARCH_model.py
# @Software: PyCharm
# @User: chent

import numpy as np  # 导入包
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
from datetime import datetime   # 导入datetime模块
from statsmodels.tsa.stattools import adfuller  # ADF单位根检验
from statsmodels.tsa import stattools       # 白噪声检验:Ljung-Box检验
# from statsmodels.tsa.arima_model import ARIMA   # 导入ARIMA模型
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   # 导入自相关和偏自相关包
import math
from statsmodels.tsa import stattools   # 残差序列的白噪声检验

# df = pd.read_csv('000906.csv', encoding='utf-8', index_col='date')  # 导入数据

# data = pd.read_csv(r'000906.csv', index_col='date', parse_dates=['date'])
# data = data.drop(columns='code')
df = pd.read_csv('000906.csv', encoding='utf-8', index_col='date')  # 导入数据

# 序列识别
df.index = pd.to_datetime(df.index)     # 转换数据格式
ts = df['open']


plt.rcParams['font.sans-serif'] = ['simhei']    # 字体为黑体
plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号 #时序图的绘制
ts.plot()
plt.xticks(rotation=45)         # 坐标角度旋转
plt.xlabel('日期')        # 横、纵坐标以及标题命名
plt.ylabel('开盘价')
plt.title('000906开盘价', loc='center')
plt.show()

result = adfuller(ts)       # 不能拒绝原假设，即原序列存在单位根
print(result)

ts1 = ts.diff().dropna()         # 一阶差分再进行ADF检验
result = adfuller(ts1)
print(result)

plt.rcParams['font.sans-serif'] = ['simhei']    # 字体为黑体
plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号
plt.xticks(rotation=45)         # 坐标角度旋转
plt.xlabel('日期')        # 横、纵坐标以及标题命名
plt.ylabel('开盘价')
plt.title('差分后的开盘价', loc='center')
ts1.plot()
plt.show()      # 一阶差分后的时序图


LjungBox = stattools.q_stat(stattools.acf(ts1)[1:12], len(ts1))[1]     # 显示第一个到第11个白噪声检验的p值
# LjungBox # 检验的p值大于0.05，因此不能拒绝原假设，差分后序列白噪声检验通过

# 模型识别与定阶   arima
model = sm.tsa.ARIMA(ts, order=(1, 1, 0))     # 白噪声检验通过，直接确定模型
# result = model.fit(disp=-1)
result = model.fit()
result.summary()        # 提取模型信息
print(result.summary)
print()


plot_acf(ts1, use_vlines=True, lags=30)   # 自相关函数图，滞后30阶
plt.show()

plot_pacf(ts1, use_vlines=True, lags=30)      # 偏自相关函数图
plt.show()

'''
    trend
'''
train_results = sm.tsa.arma_order_select_ic(ts1, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)
print('AIC', train_results.aic_min_order)       # 建立AIC值最小的模型
# print('BIC', train_results.bic_min_order)
model = sm.tsa.ARIMA(ts, (2, 1, 2)).fit()
model.summary()         # 提取模型系数等信息，保留三位小数；summary2保留四位小数

# 模型诊断
model.conf_int()    # 系数显著性检验


stdresid = model.resid / math.sqrt(model.sigma2)  # 标准化残差
plt.rcParams['font.sans-serif'] = ['simhei']    # 字体为黑体
plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号
plt.plot(stdresid)          # 标准化残差序列图
plt.xticks(rotation=45)         # 坐标角度旋转
plt.xlabel('日期')            # 横、纵坐标以及标题命名
plt.ylabel('标准化残差')
plt.title('标准化残差序列图', loc='center')

plot_acf(stdresid,lags=30)
plt.show()


LjungBox = stattools.q_stat(stattools.acf(stdresid)[1:13], len(stdresid))
LjungBox[1][-1]          # LjungBox检验的最后一个P值，大于0.05，通过白噪声检验

a = model.forecast(5)

fig, ax = plt.subplots(figsize=(6, 4))
ax = ts.ix['2018-09':].plot(ax=ax)
plt.show()
fig = model.plot_predict(5, 280)

# @Version: python3.10
# @Time: 2023/6/27 11:47
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: QT.py
# @Software: PyCharm
# @User: chent

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

specList = [
    '300617.SZ', '300346.SZ', '600745.SH', '600660.SH', '300633.SZ',
    '002301.SZ', '002174.SZ', '603444.SH', '002465.SZ', '300294.SZ',
    '300373.SZ', '300770.SZ', '603337.SH', '603906.SH', '603690.SH',
    '002511.SZ', '300246.SZ', '002791.SZ', '300443.SZ', '002338.SZ',
    '300572.SZ', '002912.SZ', '002603.SZ', '002714.SZ', '002869.SZ',
    '002829.SZ', '300677.SZ', '603367.SH', '002626.SZ', '603258.SH',
    '300552.SZ', '300723.SZ', '002920.SZ', '300115.SZ', '300709.SZ',
    '002409.SZ', '300763.SZ', '603583.SH', '300738.SZ', '603916.SH',
    '300502.SZ', '300206.SZ', '002920.SZ'
]
fields = ['open', 'high', 'low', 'close', 'avg_price', 'amp_rate']
minRate = 0.99
maxRate = 1.0255


def init(context):
    '''
        initial
    '''
    set_benchmark('000300.SH')
    set_commission(PerShare(type='stock', cost=0.0002))
    set_slippage(PriceSlippage(0.005))
    g.nowCnt = 0
    g.adjustLoop = 3
    g.stockLimit = 25
    g.preRes = pd.DataFrame()
    g.allowHandleBar = True
    g.allowSelect = True
    g.selectedStocks = specList
    log.info("初始化完成")


def create_dataset(dataset, time_steps=1):
    Xs, ys = [], []
    for i in range(len(dataset) - time_steps):
        Xs.append(dataset[i:(i + time_steps), 0])
        ys.append(dataset[i + time_steps, 0])
    return np.array(Xs), np.array(ys)

def get_stock_data(stock_name):
    '''
        get one stock data, from it being listed to 2023-06-27
        stock_name can be a list or string
        return stock_data
    '''
    stock_data = get_candle_stick(stock_name,
                                  end_date='20230627',
                                  fre_step='1d',
                                  fields=['open', 'high', 'low', 'close', 'volume'],
                                  skip_paused=False,
                                  fq='pre',
                                  bar_count=10000,
                                  is_panel=1)
    return stock_data


def get_model(stock_name):
    # 加载股票数据
    # data = pd.read_csv('stock_data.csv')
    # data = get_price([stock_name], None, '20170303', '1d', ['close', 'high', 'low', 'volume'], True, None, 1000, is_panel=1)
    data = get_stock_data(stock_name)

    # 提取收盘价作为目标变量
    close_prices = data['close'].values.reshape(-1, 1)

    # 数据预处理：归一化处理
    scaler = MinMaxScaler()
    scaled_close_prices = scaler.fit_transform(close_prices)

    # 划分训练集和测试集 8:2
    train_size = int(len(scaled_close_prices) * 0.8)
    train_data = scaled_close_prices[:train_size]
    test_data = scaled_close_prices[train_size:]

    # 构建训练集和测试集的特征和标签
    time_steps = 30  # 时间步长，表示前几天的数据用于预测下一天的价格
    X_train, y_train = create_dataset(train_data, time_steps)
    X_test, y_test = create_dataset(test_data, time_steps)

    # 转换为3D形状 [样本数，时间步长，特征数]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(time_steps, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 模型训练
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # 模型预测
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return model


def before_trading(context):
    g.allowHandleBar = True
    g.allowSelect = True


def after_trading(context):
    g.nowCnt = g.nowCnt + 1


def handle_bar(context, bar_dict):
    if g.allowSelect:
        try:
            selectStock(context, bar_dict)
        except:
            pass
        g.allowSelect = False
    checkDeficit(context, bar_dict)
    if g.allowHandleBar and (g.nowCnt % g.adjustLoop == 0):
        # 进行调仓
        g.allowHandleBar = False
        for i in list(context.portfolio.positions.keys()):
            if i not in g.selectedStocks:
                order(i, 0)
        for Stock in g.selectedStocks:
            log.info('买入: {}'.format(Stock))
            order_target_percent(Stock, 1 / g.stockLimit)


def selectStock(context, bar_dict):
    # 加载训练好的模型
    # model1 = joblib.load('GBDT0526.model')
    model1 = get_model()
    # stockList, X_array = collectData(datetime.date.today().strftime('%Y%m%d'))
    stockList, X_array = collectData(datetime.date.today())
    predictTomorrow = model1.predict(X_array)
    g.selectedStocks = []
    for i in range(len(stockList)):
        if predictTomorrow[i] == 1:
            g.selectedStocks.append(stockList[i])
    g.preRes['predict'] = predictTomorrow
    g.preRes.index = stockList
    log.info("选股结束")


def collectData(timeNow):
    stockList = get_index_stocks(symbol='000985.CSI', date=timeNow.strftime('%Y-%m-%d'))
    stockList = stockList[0:800]
    for tmp in specList:
        stockList.append(tmp)
    return stockList, collectFactors(timeNow, stockList)


def collectFactors(timeNow, stockList):
    start_time = timeNow - datetime.timedelta(days=21)
    end_time = timeNow - datetime.timedelta(days=0)
    startTime = start_time.strftime("%Y-%m-%d")
    endTime = end_time.strftime("%Y-%m-%d")
    prices = get_price(stockList, start_date=startTime, end_date=endTime,
                       fields=fields)
    trade_days = get_trade_days(startTime, endTime).strftime('%Y-%m-%d').tolist()
    X_array = []
    for i in range(len(stockList)):
        q = query(
            factor.boll,
            factor.ma,
            factor.bias,
            factor.wr,
            factor.obv,
            factor.macd,
            factor.kdj,
            factor.vmacd,
            factor.market_cap,
            factor.pe,
            factor.net_profit_growth_ratio
        ).filter(
            factor.symbol == stockList[i],
            factor.date == trade_days[-2]
        )

        res = []
        avgPrice = np.mean(prices[stockList[i]]['avg_price'])
        closePrice = prices[stockList[i]]['close'][-1]
        maxPrice = max(prices[stockList[i]]['high'])
        minPrice = min(prices[stockList[i]]['low'])
        nearestPrice = prices[stockList[i]]['open'][-1]
        earnRate = (nearestPrice / prices[stockList[i]]['open'][0]) - 1
        standardVal = np.std(prices[stockList[i]]['open'])
        tmpRes = [closePrice / avgPrice, maxPrice / avgPrice,
                  minPrice / avgPrice, nearestPrice / avgPrice,
                  earnRate, standardVal, nearestPrice]
        for j in tmpRes:
            res.append(j)
        X_array.append(np.array(res))
    log.info('{} 因子重选'.format(timeNow))
    return X_array


def checkDeficit(context, bar_dict):
    stockList = list(context.portfolio.positions)
    for stock in stockList:
        earnRate = bar_dict[stock].close / context.portfolio.positions[stock].cost_basis
        if earnRate < minRate:
            try:
                if stock in g.preRes.index:
                    if g.preRes.loc[stock, 'predict'] == -1:
                        order_target_value(stock, 0)
                else:
                    order_target_value(stock, 0)
            except:
                log.info("Error:" + stock)
                pass
        elif earnRate > maxRate:
            try:
                if stock in g.preRes.index:
                    if g.preRes.loc[stock, 'predict'] == -1:
                        order_target_value(stock, 0)
                else:
                    order_target_value(stock, 0)
            except:
                log.info("Error:" + stock)
                pass



# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime

from sklearn.externals import joblib
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
    set_benchmark('000300.SH')
    set_commission(PerShare(type='stock',cost=0.0002))
    set_slippage(PriceSlippage(0.005))
    g.nowCnt = 0
    g.adjustLoop = 3
    g.stockLimit = 25
    g.preRes = pd.DataFrame()
    g.allowHandleBar = True
    g.allowSelect = True
    log.info("初始化完成")

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
        #进行调仓
        g.allowHandleBar = False
        for i in list(context.portfolio.positions.keys()):
            if i not in g.selectedStocks:
                order(i, 0)
        for Stock in g.selectedStocks:
            log.info('买入: {}'.format(Stock))
            order_target_percent(Stock, 1 / g.stockLimit)

def selectStock(context, bar_dict):
    #加载训练好的模型
    model1 = joblib.load('GBDT.model')
    stockList, X_array = collectData(get_datetime())
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

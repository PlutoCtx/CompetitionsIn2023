# @Version: python3.10
# @Time: 2023/4/10 16:40
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: GBDTTest02.py
# @Software: PyCharm
# @User: chent

import pandas as pd
import numpy as np
import datetime
from sklearn.externals import joblib

# 初始化
def init(context):
	set_benchmark('000300.SH')
	set_commission(PerShare(type='stock', cost=0.0002))
	set_slippage(PriceSlippage(0.005))
	# 初始化
	g.nowCnt = 0
	g.adjustLoop = 3
	g.stockLimit = 25
	g.preRes = pd.DataFrame()
	g.allowHandleBar = True
	g.allowSelect = True
	log.info("Initialize Finished")

# 开盘前准备
def before_trading(context):
	g.allowHandleBar = True
	g.allowSelect = True

# 开盘后结尾
def after_trading(context):
	g.nowCnt += 1

# 交易
def handle_bar(context, bar_dict):
	if g.allowSelect:
		try:
			selectStock(context, bar_dict)
		except:
			return
		g.allowSelect = False
	checkDeficit(context, bar_dict)
	if g.allowHandleBar and (g.nowCnt % g.adjustLoop == 0):
		g.allowHandleBar = False
		for i in list(context.portfolio.positions.keys()):
			if i not in g.selectedStocks:
				order(i, 0)
		for i in g.selectedStocks:
			log.info('Buy: {}'.format(i))
			order_target_percent(i, 1 / g.stockLimit)

# 选股
def selectStock(context, bar_dict):
	# 读取模型，获取数据
	model1 = joblib.load('GDBTnew225per.model')
	lastDate = get_datetime()
	stockList, X_array = collectData(lastDate)
		predictTomorrow = model1.predict(X_array)
	g.selectedStocks = []
	# 预测明日涨跌并存储
	for i in range(len(stockList)):
		if predictTomorrow[i] == 1:
			g.selectedStocks.append(stockList[i])
	g.preRes['predict'] = predictTomorrow
	g.preRes.index = stockList
	log.info("Selection Finished")

# 获取股票信息
def collectData(timeNow):
	stockList = get_index_stocks(
		symbol='000985.CSI',
		date=timeNow.strftime('%Y-%m-%d')
	)
	stockList = stockList[0:500]
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
	for tmp in specList:
		stockList.append(tmp)
	X_array = collectFactors(timeNow, stockList)
	return stockList, X_array

# 获取因子
def collectFactors(timeNow, stockList):
	# 获取前21天的数据
	startTime = timeNow - datetime.timedelta(days=21)
	endTime = timeNow - datetime.timedelta(days=0)
	startTime = startTime.strftime("%Y-%m-%d")
	endTime = endTime.strftime("%Y-%m-%d")
	prices = get_price(
		stockList,
		start_date=startTime,
		end_date=endTime,
		fields=['open', 'high', 'low', 'close', 'avg_price', 'amp_rate']
	)
	trade_days = get_trade_days(startTime,endTime).\
		strftime('%Y-%m-%d').tolist()
	X_array = []
	for i in range(len(stockList)):
		q = query(
			factor.macd,
			factor.wr,
			factor.obv,
			factor.kdj,
			factor.vmacd,
		).filter(
			factor.symbol == stockList[i],
			factor.date == trade_days[-2]
		)
		res = get_factors(q)
		if res.isnull().values.any():
			log.info(res)
			kk = -3
			while res.isnull().values.any():
				q = query(
					factor.macd,
					factor.wr,
					factor.obv,
					factor.kdj,
					factor.vmacd,
				).filter(
					factor.symbol == stockList[i],
					factor.date == trade_days[kk]
				)
				res = get_factors(q)
				kk -= 1
		res = res.values.tolist()
		res = res[0]

		avgPrice = np.mean(prices[stockList[i]]['avg_price'])
		closePrice = prices[stockList[i]]['close'][-1]
		maxPrice = max(prices[stockList[i]]['high'])
		minPrice = min(prices[stockList[i]]['low'])
		nearestPrice = prices[stockList[i]]['open'][-1]
		earnRate = (nearestPrice / prices[stockList[i]]['open'][0]) - 1
		standardVal = np.std(prices[stockList[i]]['open'])
		tmpRes = [
			closePrice / avgPrice,
			maxPrice / avgPrice,
			minPrice / avgPrice,
			nearestPrice / avgPrice,
			earnRate, standardVal,
			nearestPrice
		]
		for j in tmpRes:
			res.append(j)
		X_array.append(np.array(res))
	log.info('{} . Factors Collection Finished'.format(timeNow))
	return X_array

# 风控
def checkDeficit(context, bar_dict):
	stockList = list(context.portfolio.positions)
	for stock in stockList:
		earnRate = bar_dict[stock].close / context.portfolio.positions[stock].cost_basis
	# 止损线
	if earnRate < 0.98:
		if stock in g.preRes.index:
			if g.preRes.loc[stock, 'predict'] == -1:
				order_target_value(stock, 0)
		else:
			order_target_value(stock, 0)
	# 止盈线
	elif earnRate > 1.0225:
		try:
			if stock in g.preRes.index:
				if g.preRes.loc[stock, 'predict'] == -1:
					order_target_value(stock, 0)
			else:
				order_target_value(stock, 0)
		except:
			log.info("Error:" + stock)
			raise Exception
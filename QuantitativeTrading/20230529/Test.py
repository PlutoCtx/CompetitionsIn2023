# @Version: python3.10
# @Time: 2023/4/10 20:43
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: Test.py
# @Software: PyCharm
# @User: chent

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
# 股票策略模版
# 初始化函数,全局只运行一次
def init(context):
	# 设置基准收益：沪深300指数
	set_benchmark('000300.SH')
	# 打印日志
	log.info('策略开始运行,初始化函数全局只运行一次')
	# 设置股票每笔交易的手续费为万分之二(手续费在买卖成交后扣除,不包括税费,税费在卖出成交后扣除)
	set_commission(PerShare(type='stock', cost=0.0002))
	# 设置股票交易滑点0.5%,表示买入价为实际价格乘1.005,卖出价为实际价格乘0.995
	set_slippage(PriceSlippage(0.005))
	# 设置日级最大成交比例25%,分钟级最大成交比例50%
	# 日频运行时，下单数量超过当天真实成交量25%,则全部不成交
	# 分钟频运行时，下单数量超过当前分钟真实成交量50%,则全部不成交
	set_volume_limit(0.25, 0.5)
	# 设置要操作的股票：同花顺
	# context.security = '300033.SZ'
	# 回测区间、初始资金、运行频率请在右上方设置
	g.nowCnt = 0
	g.adjustLoop = 5
	g.stockLimit = 25
	g.preRes = pd.DataFrame()
	# 允许操作
	g.allowHandleBar = True
	g.allowSelect = True

	# 设置要操作的股票：同花顺
	context.security = ''

# 每日开盘前9:00被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):
	# 获取日期
	date = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
	# 打印日期
	log.info('{} 盘前运行'.format(date))
	# 允许操作
	g.allowHandleBar = True
	g.allowSelect = True

# 开盘时运行函数
def handle_bar(context, bar_dict):
	# # 获取时间
	# time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
	# # 打印时间
	# log.info('{} 盘中运行'.format(time))
	# # 获取股票过去20天的收盘价数据
	# closeprice = history(context.security, ['close'], 20, '1d', False, 'pre', is_panel=1)
	# # 计算20日均线
	# MA20 = closeprice['close'].mean()
	# # 计算5日均线
	# MA5 = closeprice['close'].iloc[-5:].mean()
	# # 获取当前账户当前持仓市值
	# market_value = context.portfolio.stock_account.market_value
	# # 获取账户持仓股票列表
	# stocklist = list(context.portfolio.stock_account.positions)
	#
	# # 如果5日均线大于20日均线,且账户当前无持仓,则全仓买入股票
	# if MA5 > MA20 and len(stocklist) ==0 :
	# 	# 记录这次买入
	# 	log.info("5日均线大于20日均线, 买入 %s" % (context.security))
	# 	# 按目标市值占比下单
	# 	order_target_percent(context.security, 1)
	#
	# # 如果5日均线小于20日均线,且账户当前有股票市值,则清仓股票
	# elif MA20 > MA5 and market_value > 0:
	# 	# 记录这次卖出
	# 	log.info("5日均线小于20日均线, 卖出 %s" % (context.security))
	# 	# 卖出所有股票,使这只股票的最终持有量为0
	# 	order_target(context.security, 0)

## 收盘后运行函数,用于储存自定义参数、全局变量,执行盘后选股等
	i = 0
	# 如果允许挑选
	if g.allowSelect:
		try:
			selectStock(context, bar_dict)
		except:
			log.info("selectStock failed")
			return
		g.allowSelect = False
	# 风险预防
	risk_prevention(context, bar_dict)
	# 允许交易股票
	if g.allowHandleBar and (g.nowCnt % g.adjustLoop == 0):
		g.allowHandleBar = False
		for i in list(context.portfolio.positions.keys()):
			if i not in g.selectedStocks:
				order(i, 0)

		for i in g.selectedStocks:
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
	startTime = (timeNow - datetime.timedelta(days=21)).strftime("%Y-%m-%d")
	endTime = (timeNow - datetime.timedelta(days=0)).strftime("%Y-%m-%d")
	prices = get_price(
		stockList,
		start_date=startTime,
		end_date=endTime,
		fields=['open', 'high', 'low', 'close', 'avg_price', 'amp_rate']
	)
	trade_days = get_trade_days(startTime, endTime).strftime('%Y-%m-%d').tolist()
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
		).order_by(
			valuation.market_cap.desc() # 总市值由大到小排序
		).limit(
			g.stockLimit
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
				).order_by(
					valuation.market_cap.desc() # 总市值由大到小排序
				).limit(
					g.stockLimit
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
		yields= (nearestPrice / prices[stockList[i]]['open'][0]) - 1
		standardVal = np.std(prices[stockList[i]]['open'])
		tmpRes = [
			closePrice / avgPrice,
			maxPrice / avgPrice,
			minPrice / avgPrice,
			nearestPrice / avgPrice,
			yields, standardVal,
			nearestPrice
		]
		for j in tmpRes:
			res.append(j)
		X_array.append(np.array(res))
	log.info('{} . Factors Collection Finished'.format(timeNow))
	return X_array




def risk_prevention(context, bar_dict):
	stockList = list(context.portfolio.positions)
	for stock in stockList:
		yields = bar_dict[stock].close / context.portfolio.positions[stock].cost_basia
		if yields < 0.98:
			if stock in g.preRes.index:
				if g.preRes.loc[stock, 'predict'] == -1:
					order_target_value(stock, 0)
			else:
				order_target_value(stock, 0)
		elif yields > 1.03:
			if stock in g.preRes.index:
				if g.preRes.loc[stock, 'predict'] == -1:
					order_target_value(stock, 0)
			else:
				order_target_value(stock, 0)
		log.info("risk_prevention finished")



def after_trading(context):
	g.nowCnt += 1
	# 获取时间
	time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
	# 打印时间
	log.info('{} 盘后运行'.format(time))
	log.info('一天结束')
# @Version: python3.10
# @Time: 2023/4/10 20:16
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: GBDT04.py
# @Software: PyCharm
# @User: chent

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datetime import date, timedelta, datetime
#初始化账户
def init(context):
	#设置基准收益：中证 800 指数
	set_benchmark('000906.SH')
	#打印日志
	log.info('策略开始运行')
	# 训练集 x 的天数
	context.num_x_days = 7
	# 训练集 y 的天数
	context.num_y_days = 7
	# 训练集 y 的涨幅
	context.index_rise = 0.02
	# 调仓周期
	context.cycle = 1
	# 止盈参数
	context.index_take_profit = 0.1
	# 止损参数
	context.index_stop_loss = -0.08
	# 持仓数量
	context.num_hold_stocks = 500
	# 调仓信号
	context.cycle_flag = True
	# 下一个调仓的日期
	context.transfer_positions_date = get_interval_date(get_datetime(), context.cycle_flag)
	train_date1 = datetime(2020,1,8)
	train_date2 = datetime(2020,5,1)

	#获取训练集 x1，x2，y1，y2
	x1, y1 = create_train_set(train_date1,context)
	x2, y2 = create_train_set(train_date2,context)
	x= np. concatenate((x1, x2), axis=0)
	y = y1+y2
	#数据预处理
	context.scaler = preprocessing.StandardScaler().fit(x)
	df_scaler = context.scaler.transform(x)
	# 这里划分数据以 0.2 的来划分 训练集训练结果 测试集测试结果
	X_train, X_test, y_train, y_test = train_test_split(df_scaler, y, test_size=0.2, random_state=0)
	#C, gamma = find_para(train_X, test_X, train_y, test_y)
	#使用 SVM 算法进行训练
	context.clf = svm.SVC(C=30.15, gamma=0.000110398)
	context.clf.fit(df_scaler, np.array(y))
	get_iwencai('所有 A 股','stocks_list')
	#根据 iwencai 筛选出的股票池
	context.stocks = list()
	#预测结果
	context.predicts=list()

#每日开盘前 9:00 被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):
	# 获取日期
	date = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
	# 打印日期
	log.info('{} 盘前运行'.format(date))

#设置买卖条件，每个交易频率(日/分钟/tick)调用一次
def handle_bar(context, bar_dict):
	#获取时间
	time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
	#打印时间
	log.info('当天时间'+str(time))
	##log.info(context.stocks_list)
	#获取当天数据，取得预测结果
	date = get_datetime()
	today_date = TransTime(date)
	num_stocks = len(list(context.portfolio.positions))
	if context.cycle_flag or today_date==context.transfer_positions_date or num_stocks < 5:
		print("-----------------进行调仓-------------------")
		context.cycle_flag = False
		context.transfer_positions_date = get_interval_date(get_datetime(),context.cycle)
		print("下一个调仓的的日期"+str(context.transfer_positions_date))
		buy_and_sell(context, bar_dict, date)

	TakeProfitandStopLoss(context,
						  bar_dict,
						  context.stocks,
						  context.predicts,
						  context.index_take_profit,
						  context.index_stop_loss
						  )
def buy_and_sell(context, bar_dict, date):
	context.group_num = 1
	# 因子分组
	context.num = 1
	# 资金分配
	context.clean_ty = "median_extremum-standardize" # 系统因子排序
	context.sys_sort = []
	# 因子比率
	context.sys_factors = {
		'equity_ratio': 0.1,
		'weighted_roe': 0.1,
		'parent_company_share_holders_net_profit_growth_ratio': 0.1,
		'pe': -0.1,
		'turnover_ratio_of_receivable': 0.1,
		'bias': -0.1,
		'macd': 0.1,
		'obv': 0.1,
		'arbr': 0.1,
		'boll':	0.1
	}
	# 用户因子筛选条件
	context.user_query = "" # 用户因子排序
	context.user_sort = []

	# 用户因子比率
	context.user_factors = {}
	# 筛选的股票
	context.securities = []
	nowtime_str = get_last_datetime().strftime("%Y-%m-%d")

	securities_df = sfactor_stock_scanner(
		context.stocks_list,
		query(
			factor.symbol,
			factor.equity_ratio,
			factor.weighted_roe,
			factor.parent_company_share_holders_net_profit_growth_ratio,
			factor.pe,
			factor.turnover_ratio_of_receivable,
			factor.bias,
			factor.macd,
			factor.obv,
			factor.arbr,
			factor.boll
		),
		context.sys_sort,
		context.sys_factors,
		context.user_query,
		context.user_sort,
		context.user_factors,
		nowtime_str,
		context.clean_ty,
	)
	if len(securities_df):
		index_s, index_e = length_split(len(securities_df), context.group_num, context.num)
		context.securities.extend(list(securities_df.iloc[index_s:index_e]['symbol']))

	#context.securities = context.securities[:800]
	print(context.securities)
	context.stocks, X_array = get_needed_data(
		date,
		context.securities,
		context.num_x_days
	)
	context.predicts = context.clf.predict(context.scaler.transform(X_array))
	for i in range(len(context. stocks)):
		#如果预测结果为 1 且未持仓，则买入
		if context.predicts[i] == 1 and context.stocks[i] not in context.portfolio.positions.keys():
			log.info('buying %s' %context.stocks[i])
			order_percent(context.stocks[i], 1/context.num_hold_stocks)
		#如果预测结果为-1 且已持仓，则清仓
		elif context.predicts[i] == -1 and context.stocks[i] in context.portfolio.positions.keys():
			log.info('selling %s' % context.stocks[i])
			order_target(context.stocks[i], 0)

def get_needed_data(current_date,stocks,num_x_days):
	time_end = TransTime(current_date)
	time_start = TransTime(current_date-timedelta(days=num_x_days))
	prices = get_price(stocks,
					   start_date=time_start,
					   end_date=time_end,
					   fields=['open','high', 'low', 'close', 'avg_price', 'amp_rate']
					   )
	X_array =[]
	for i in range(len(stocks)):
		#均价
		meanPrice = np.mean(prices[stocks[i]]['avg_price'])
		#收盘价
		finalPrice = prices[stocks[i]]['close'] [-1 ]
		#最大值
		maxPrice = max(prices[stocks[i]]['high'])
		#最小值
		minPrice = min(prices[stocks[i]]['low'])
		#现价
		nowPrice = prices[stocks[i]]['open'][-1]
		#涨跌幅
		quoteRate = (prices[stocks[i]]['open'][-1] / prices[stocks[i]]['open'][0]) - 1
		#标准差
		stdnow = np.std(prices[stocks[i]]['open'])
		#组织成向量
		X = np.array( [finalPrice / meanPrice, maxPrice / meanPrice, minPrice /
		meanPrice, nowPrice / meanPrice, quoteRate,stdnow, nowPrice])
		X_array.append(X)
	X_array = np.array(X_array)
	#返回当天的所有股票
	return stocks, X_array

#获取最佳参数惩罚系数 C 和 gamma
def find_para(X_train, X_test, y_train, y_test):
	C_list = []
	gamma_list = []
	score_list = []
	for i in range(25, 35, 0.01):
		for k in range(0.01, 1, 0.01):
			k = k / 200
			clf = svm.SVC(C=i, gamma=k)
			# 对训练数据进行训练
			clf.fit(X_train, y_train)
			C_list.append(i)
			gamma_list.append(k)
			# 对测试集数据进行打分
			score_list.append(clf.score(X_test, y_test))
	C = np.array(C_list)[np.array(score_list) == max(score_list)][0]
	gamma = np.array(gamma_list)[np.array(score_list) == max(score_list)][0]
	return C, gamma


# datetime 转换格式
def TransTime(date):
	return date.strftime("%Y%m%d")

# 获取训练数据
def create_train_set(select_data, context):
	label = []
	stock11 = get_index_stocks(symbol='000906.SH', date=select_data.strftime("%Y- %m-%d"))
	stocks, X_array = get_needed_data(select_data, stock11, context.num_x_days)
	traindata = select_data + timedelta(days=context.num_y_days)
	fdate = TransTime(traindata)
	# 选取所有涨幅>2%的股票为 1，其余为-1
	prices = get_price(stock11, start_date=select_data, end_date=fdate, fields=['open'])
	for i in range(len(stocks)):
		if (prices[stocks[i]]['open'][-1] / prices[stocks[i]]['open'][0] - 1 >
				context.index_rise):
			label.append(1)
	else:
		label.append(-1)
	# 返回训练数据
	return X_array, label


# 设置止损条件
def TakeProfitandStopLoss(context, bar_dict, stocks, predicts, index_take_profit, index_stop_loss):
	print("根据盈亏线进行止盈止损")
	last_date = get_datetime()
	for i in range(len(stocks)):
		if stocks[i] in context.portfolio.positions.keys():
			##print(stocks[i])
			if context.portfolio.positions[stocks[i]].available_amount > 0:
				purchase_price = context.portfolio.positions[stocks[i]].cost_basis
				dic_close = get_last_minute_close([stocks[i]], last_date)
				last_close2 = float(dic_close[stocks[i]]['close'][-1])
				#last_close = bar_dict[stocks[i]].close
				if(purchase_price != 0):
					rate = (last_close2-purchase_price)/purchase_price
				else:
					rate = 0
				print("股票"+str(stocks[i])+"：")
				print("购买价格为"+str(purchase_price))
				print("上一分钟收盘价："+str(last_close2))
				print("收益率："+str(rate))

	#cumulative_return=bar_dict[stocks[i]].close/context.portfolio.positions[stocks[i]].cost_basis
	#log.info(cumulative_return)
				if rate > index_take_profit:
					#print("达到止盈线，进行涨跌预测")
					#if context.predicts[i] == -1:
					log.info('()执行止损'.format(last_date))
					log.info('卖出股票{}'.format(stocks[i]))
					order_target_value(stocks[i],0)
					#else:
					# print("预测为涨，继续持有")
				if rate < index_stop_loss:
					#print("达到止损线，进行涨跌预测")
					#if context.predicts[i] == -1:
					log.info('()执行止盈'.format(last_date))
					log.info('卖出股票{}'.format(stocks[i]))
					try:
						order_target_value(stocks[i],0)
					except:
						print("卖不出去")
				#else:
				# print("预测为涨，继续持有")

def get_last_minute_close(symbol_list, current_date):
	justNow = current_date + timedelta(minutes= -1)
	justNow = justNow.strftime('%Y%m%d %H:%M')
	# print("上一分钟"+str(justNow))
	dic_close = get_price(symbol_list, None, justNow, '1m', ['close'], True, None, 30, is_panel=0)
	return dic_close

#跳过周末，获取想要的下一个交易日。输入为 datetime 格式
def get_interval_date(date, days):
	date = TransTime(date)
	date = pd.Timestamp(date).normalize()
	date_index = get_all_trade_days()
	trade_date = date_index[date_index.searchsorted(date, side='left') + days]
	trade_date = trade_date.strftime("%Y%m%d")
	return trade_date

def length_split(length, group_num, num):
	if length == 0 or group_num == 0:
		return 0, 0
	return (num - 1) * int(length / group_num), num * int(length / group_num) if num != group_num else length

## 收盘后运行函数,用于储存自定义参数、全局变量,执行盘后选股等
def after_trading(context):
	# 获取时间
	time = get_datetime().strftime('%Y-%m- %d %H:%M:%S')
	# 打印时间
	log.info('{} 盘后运行'.format(time))
	log.info('一天结束')
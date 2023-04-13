# @Version: python3.10
# @Time: 2023/4/10 16:03
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: __init__.py.py
# @Software: PyCharm
# @User: chent

import collections
import datetime

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# 初始化函数,全局只运行一次
def init(context):
	# 设置基准收益：000906.SH
	set_benchmark("000906.SH")
	# 初始化用于检测股市的全局变量
	g.is_machine_learning = False  # 是否需要训练模型
	g.selected_stocks = []
	g.cur_rounds = 0  # 当前轮数
	g.adjust_freq = 1  # 调整频率
	g.max_stocks_count = 20  # 最大股票数
	g.forecast_res = pd.DataFrame()  # 预测结果
	g.machine_learning_model = "XGB_GBDT_2022_906_3.plk"  # XGB 模型
	g.is_can_handle_bar = True  # 允许交易 flag
	g.is_can_select_stock_with_models = True  # 允许选择 flag
	# 训练模型
	if g.is_machine_learning:
		XGB = machine_learning()
		joblib.dump(XGB, g.machine_learning_model)


# 每日开盘前 9:00 被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):
	# 设置允许交易
	g.is_can_handle_bar = True  # 允许交易 flag
	g.is_can_select_stock_with_models = True  # 允许选择 flag
	print("开盘准备完毕...")


# 开盘时运行函数
def handle_bar(context, bar_dict):
	# 判断是否可以选股票
	if g.is_can_select_stock_with_models == True:
		try:
			# 选择股票
			select_stocks_with_model(context, bar_dict)
		except Exception as e:
			print(e)
			pass
		# 不再允许选择
		g.is_can_select_stock_with_models = False
	# 模型预测，检测亏损
	detection_loss(context, bar_dict)
	# 如果允许交易，则交易
	if (g.cur_rounds % g.adjust_freq == 0) and g.is_can_handle_bar:
		g.is_can_handle_bar = False
		now_date = get_datetime()
		print("时间:{} --> 执行调仓...".format(now_date))
		# 遍历所有的持仓的股票。判断购入或售出
		for stock in list(context.portfolio.positions.keys()):
			if stock not in g.selected_stocks:
				print("售出股票 --> ", stock)
				order_target_value(stock, 0)
	for selected_stock in g.selected_stocks:
		print("时间:{} --> 购买了股票: {}".format(now_date, selected_stock))
		stocks = []
		for i in range(5):
			stocks.append(selected_stock)
		order_target_percent(selected_stock, 1 / (g.max_stocks_count - len(stocks)))



# 量化选股
def select_stocks_with_model(context, bar_dict):
	try:
		print("开始选股...")
		# 导入模型
		XGB_model = joblib.load(g.machine_learning_model)
		now_date = get_datetime()
		stock_list, X_array = search_stocks_datas(now_date)
		print("本次特征向量数据共 --> ", len(X_array))
		g.selected_stocks = []
		tomorrow_predict = XGB_model.predict(X_array)
		print("预测数据 --> ", collections.Counter(tomorrow_predict))
		# print(tomorrow_predict)
		# 遍历预测结果，如果为 1，可以买入
		for i in range(len(stock_list)):
			if tomorrow_predict[i] == 1:
				g.selected_stocks.append(stock_list[i])
		g.forecast_res["predict"] = tomorrow_predict
		g.forecast_res.index = stock_list
		print("选股结束...")
	except Exception as e:
		print(e)

# 获取中证 800 股票集
def get_000906_sh(now_time, sh="000985.CSI"):
	stock_list = get_index_stocks(
		symbol="000906.SH", date=now_time.strftime("%Y-%m-%d")  # 中证 800 数据集
	)
	stock_list2 = get_index_stocks(symbol=sh, date=now_time.strftime("%Y-%m-%d"))
	res_stock_list = []
	try:
		res_stock_list.extend(stock_list)
		raise Exception("数据集")
	except:
		pass
	res_stock_list.extend(stock_list2[:100])
	return res_stock_list

# 收集数据
def search_stocks_datas(now_time):
	stock_list = get_000906_sh(now_time)
	print("原始股票集长度 --> ", len(stock_list))
	# spec 股
	spec_list = [
		"002714.SZ", "002409.SZ", "600745.SH", "300443.SZ",
		"002338.SZ", "300346.SZ", "600660.SH", "300115.SZ",
		"300373.SZ", "002301.SZ", "002174.SZ"]
	res_stocks = []
	res_stocks.extend(stock_list)
	res_stocks.extend(spec_list)
	res_stocks = list(set(res_stocks))
	return res_stocks, factors_collection(now_time, res_stocks)

# 收集预测特征
def factors_collection(now_time, stock_list, days=21):
	# 21 天跨度
	start_time = now_time - datetime.timedelta(days)
	end_time = now_time - datetime.timedelta(0)
	start_time_format = start_time.strftime("%Y-%m-%d")
	end_time_format = end_time.strftime("%Y-%m-%d")
	prices = get_price(
		stock_list,
		start_date=start_time_format,
		end_date=end_time_format,
		fields=["open", "high", "low", "close", "avg_price", "amp_rate"]
	)
	trade_days = (
		get_trade_days(start_time_format, end_time_format).strftime("%Y-%m-%d").tolist()
	)
	X_array = []
	for stock in stock_list:
		try:
			# 择时交易（MACD & RSI 指标组合）
			q = query(
				# 技术指标（技术因子）：共 7 个
				factor.macd,
				factor.rsi,
				factor.kdj,
				factor.boll,
				factor.obv,
				factor.arbr,
				factor.cr
			).filter(
				factor.symbol == stock, factor.date == trade_days[-2]
			).order_by(
				valuation.market_cap.desc()		# 总市值由大到小排序
			).limit(
				10
			)
			res = get_factors(q)
			res = res[0]
		except:
			pass
	try:
		# 21 天平均价格
		mean_price = np.mean(prices[stock]["avg_price"])
		# 最终收盘价
		final_price = prices[stock]["close"][-1]
		# 最高价
		max_price = max(prices[stock]["high"])
		# 最低价
		min_price = min(prices[stock]["low"])
		# 当天的价格
		today_price = prices[stock]["open"][-1]
		# 涨与跌的幅度
		price_rate = 0 + (prices[stock]["open"][-1] / prices[stock]["open"][0]) - 1
		# 当前的开盘价标准差
		today_std = np.std(prices[stock]["open"])
		# 特征向量
		X = np.array([
			# 基本因子：共 7 个
			final_price / mean_price,
			max_price / mean_price,
			min_price / mean_price,
			today_price / mean_price,
			price_rate, today_std,
			today_price
		])
		X_array.append(X)
	except:
		X_array.append(np.array([0, 0, 0, 0, 0, 0, 0]))
		print("股票数据缺失，特征置为空 ", stock)
	X_array = np.array(X_array)
	print("时间:{} --> 特征收集完毕".format(now_time))
	return X_array

# 检测亏损
def detection_loss(context, bar_dict):
	stock_list = list(context.portfolio.positions)
	for stock in stock_list:
		# 盈利比
		earn_rate = (
			bar_dict[stock].close / context.portfolio.positions[stock].cost_basis
		)
		if earn_rate < 0.98:
			try:
				# 在股票池且预测会亏钱就及时止损卖出，如果不在股票池也卖出
				if stock in g.forecast_res.index:
					if g.forecast_res.loc[stock, "predict"] == -1:
						print("止损,售出股票 --> ", stock)
						order_target_value(stock, 0)
				else:
					print("止损,售出股票 --> ", stock)
					order_target_value(stock, 0)
			except Exception as e:
				print("止损错误 ", stock, e)
				pass
		elif earn_rate > 1.02:
			try:
				# 在股票池且预测会亏钱就及时止损卖出，如果不在股票池也卖出
				if stock in g.forecast_res.index:
					if g.forecast_res.loc[stock, "predict"] == -1:
						print("止盈,售出股票 --> ", stock)
						order_target_value(stock, 0)
				else:
					print("止盈,售出股票 --> ", stock)
					order_target_value(stock, 0)
			except Exception as e:
				print("止盈错误 ", stock, e)
				pass

# 收盘后运行函数
def after_trading(context):
	g.cur_rounds += 1  # 轮数加一

# 获取股票数据及其特征向量（模型训练用）
def handle_stockes_with_X(today_time, days=21):
	start_time = today_time - datetime.timedelta(days)
	end_time = today_time - datetime.timedelta(0)
	start_time_format = start_time.strftime("%Y-%m-%d")
	end_time_format = end_time.strftime("%Y-%m-%d")
	# 获取股票列表
	stocks = get_index_stocks(symbol="000906.SH", date=today_time.strftime("%Y-%m-%d"))
	# 获取前三周到当天的价格数据（开盘价、最高价、最低价、收盘价、平均价、振幅）
	prices = get_price(
		stocks,
		start_date=start_time_format,
		end_date=end_time_format,
		fields=["open", "high", "low", "close", "avg_price", "amp_rate"]
	)
	# 特征向量集合
	X_array = []
	for stock in stocks:
		try:
			# 多因子选股
			q = query(
				# 财务因子：共 5 个
				factor.pb,
				factor.current_ratio,
				factor.tangible_assets_liabilities,
				factor.weighted_roe,
				factor.net_profit_growth_ratio).filter(
					factor.symbol == stock,
					factor.date == get_trade_days(start_time_format, end_time_format)
					.strftime("%Y-%m-%d")
					.tolist()[-2]
			)
			res = get_factors(q)
			res = res[0]
		except:
			pass
		# 21 天平均价格
		mean_price = np.mean(prices[stock]["avg_price"])
		# 最终收盘价
		final_price = prices[stock]["close"][-1]
		# 最高价
		max_price = max(prices[stock]["high"])
		# 最低价
		min_price = min(prices[stock]["low"])
		# 当天的价格
		today_price = prices[stock]["open"][-1]
		# 涨与跌的幅度
		price_rate = 0 + (prices[stock]["open"][-1] / prices[stock]["open"][0]) - 1
		# 当前的开盘价标准差
		today_std = np.std(prices[stock]["open"])
		# 特征向量
		X = np.array([
			final_price / mean_price,
			max_price / mean_price,
			min_price / mean_price,
			today_price / mean_price,
			price_rate,
			today_std, today_price
		])
		# 添加到特征向量集合里
		X_array.append(X)

	# 使用 numpy 处理集合数据
	X_array = np.array(X_array)
	return X_array, stocks


# 获取模型数据
def create_train_data(inflex_date):
	# 利率集合
	earn_rate = []
	# 获取股票及其特征
	X_array, stocks = handle_stockes_with_X(inflex_date)
	# 今天到未来一周
	today_date = inflex_date + datetime.timedelta(0)
	future_date = inflex_date + datetime.timedelta(7)
	# 格式化时间
	today_date_format = today_date.strftime("%Y-%m-%d")
	future_date_format = future_date.strftime("%Y-%m-%d")
	# 获取开盘价数据
	prices = get_price(
		stocks,
		start_date=today_date_format,
		end_date=future_date_format,
		fields=["open"]
	)
	# 遍历所有的股票
	for stock in stocks:
		if (prices[stock]["open"][1] / prices[stock]["open"][0]) > 1.06:
			earn_rate.append(1)
		else:
			earn_rate.append(-1)
	return X_array, earn_rate

# 模型学习
def machine_learning():
	try:
		from xgboost import XGBClassifier
	except Exception as e:
		print(e)
		pass
	learning_date_start_time = datetime.datetime(2012, 1, 1)
	learning_date_start_pass = 30
	learning_date_length = 60
	X_array, earn_rate = create_train_data(learning_date_start_time)
	for i in range(1, learning_date_length + 1):
		now_date = learning_date_start_time + datetime.timedelta(
			i * learning_date_start_pass
		)
		unit_X_array, unit_earn_rate = create_train_data(now_date)
		X_array = np.concatenate((X_array, unit_X_array), axis=0)
		earn_rate += unit_earn_rate
		print("训练模型进度 --> {}/{}".format(i, learning_date_length))
	print("数据收集完毕，训练模型中...")
	# X_array = preprocessing.scale(X_array) # 标准化
	print(X_array)
	X_train, X_test, y_train, y_test = train_test_split(
		X_array, earn_rate, test_size=0.3
	)
	print("给予训练的数据量 --> ", len(X_train))
	print("给予测试的数据量 --> ", len(X_test))
	XGB = XGBClassifier(n_estimators=100)
	XGB.fit(X_train, y_train)

	print("score --> ", XGB.score(X_test, y_test))
	print(
		"mean score --> ",
		cross_val_score(XGB, X_test, y_test, cv=5, scoring="accuracy").mean()
	)
	print("模型训练完成！")
	return XGB

# @Version: python3.10
# @Time: 2023/7/15 09:22
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: finalEdition02.py
# @Software: PyCharm
# @User: chent


import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
# timedelta用来计算时间 用来推算过去未来的时间
from datetime import date, timedelta, datetime


# 初始化账户
def init(context):
    '''
    初始化基本量
    :param context: 全局信息
    :return: 无返回值
    '''
    # 设置基准收益：中证800指数
    set_benchmark('000906.SH')
    # 打印日志
    log.info('策略开始运行')
    # 训练集x的天数
    context.num_x_days = 7
    # 训练集y的天数
    context.num_y_days = 7
    # 训练集y的涨幅
    context.index_rise = 0.02
    # 调仓周期
    context.cycle = 1
    # 止盈参数
    context.index_profit = 0.06
    # 止损参数
    # TODO 嫌太大
    context.index_loss = -0.06
    # TODO 持仓数量
    context.num_hold_stocks = 25

    # 调仓信号
    context.cycle_flag = True
    # 下一个调仓的日期
    context.transfer_positions_date = get_interval_date(get_datetime(), context.cycle_flag)
    # TODO 股票池更换
    get_iwencai('中证800指数成份股', 'stocks_list')


    train_date_st = datetime(2023, 1, 1)
    train_date_length = 60
    train_date_step = 30
    X_array, label = create_train(train_date_st, context)

    for i in range(1, train_date_length):
        now_date = train_date_st + timedelta(i * train_date_step)
        unit_X_array, unit_label = create_train(now_date, context)
        X_array = np.concatenate((X_array, unit_X_array), axis=0)
        label += unit_label

    train_size = 60 * train_date_length
    context.scaler = preprocessing.StandardScaler().fit(X_array)

    # 最后一列的数据
    X_train = X_array[:-train_size]
    # 行号为最后一行的数据
    X_test = X_array[-train_size:]
    Y_train = label[:-train_size]
    Y_test = label[-train_size:]
    # model = optimize_C(X_train, X_test, Y_train, Y_test)

    context.scaler = preprocessing.StandardScaler().fit(X_array)
    df_scaler = context.scaler.transform(X_array)
    # TODO 可能存在过拟合问题，把C改小/gamma改大
    context.clf = svm.SVC(C=30.15, gamma=0.001100009)
    context.clf.fit(df_scaler, np.array(label))


# 每日开盘前 9:00 被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):
    # 获取日期
    date = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    # 打印日期
    log.info('{} 盘前运行'.format(date))


# 设置买卖条件，每个交易频率(日/分钟/tick)调用一次
def handle_bar(context, bar_dict):
    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    # 打印时间
    log.info('当天时间' + str(time))
    ##log.info(context.stocks_list)
    # 获取当天数据，取得预测结果
    date = get_datetime()
    today_date = TransTime(date)

    # 账户信息里面的仓位信息 获取仓库的数量
    # 账户财产信息列表的长度
    num_stocks = len(list(context.portfolio.positions))
    if context.cycle_flag or today_date == context.transfer_positions_date or num_stocks < 5:
        print("-----------------进行调仓-------------------")
        context.cycle_flag = False
        context.transfer_positions_date = get_interval_date(get_datetime(), context.cycle)
        print("下一个调仓的的日期" + str(context.transfer_positions_date))
        buy_and_sell(context, bar_dict, date)

    TakeProfitandStopLoss(context, bar_dict, context.stocks, context.predicts, context.index_profit, context.index_loss)


def buy_and_sell(context, bar_dict, date):
    context.group_num = 1
    # 因子分组
    context.number = 1
    # 资金分配
    context.clean_ty = "median_extremum-standardize"
    # 系统因子排序
    context.sys_sort = []
    # 因子比率
    context.sys_factors = {'equity_ratio': 0.1,
                           'weighted_roe': 0.1,
                           'parent_company_share_holders_net_profit_growth_ratio': 0.1,
                           'pe': -0.1,
                           'turnover_ratio_of_receivable': 0.1,
                           'bias': -0.1,
                           'macd': 0.1,
                           'obv': 0.1,
                           'arbr': 0.1,
                           'boll': 0.1}
    # 用户因子筛选条件
    context.user_query = ""
    # 用户因子排序
    context.user_sort = []
    # 用户因子比率
    context.user_factors = {}
    # 筛选的股票
    context.securities = []
    nowtime_str = get_last_datetime().strftime("%Y-%m-%d")
    # 获取因子数据
    securities_df = sfactor_stock_scanner(
        context.stocks_list,
        query(factor.symbol,
              factor.equity_ratio,
              factor.weighted_roe,
              factor.parent_company_share_holders_net_profit_growth_ratio,
              factor.pe,
              factor.turnover_ratio_of_receivable,
              factor.bias,
              factor.macd,
              factor.obv,
              factor.arbr,
              factor.boll),
        context.sys_sort,
        context.sys_factors,
        context.user_query,
        context.user_sort,
        context.user_factors,
        nowtime_str,
        context.clean_ty,
    )
    if len(securities_df):
        index_s, index_e = length_split(len(securities_df), context.group_num, context.number)
        context.securities.extend(list(securities_df.iloc[index_s:index_e]['symbol']))

    # context.securities = context.securities[:800]
    print(context.securities)
    context.stocks, X_array = get_needed_data(date, context.securities, context.num_x_days)
    context.predicts = context.clf.predict(context.scaler.transform(X_array))
    for i in range(len(context.stocks)):
        # 如果预测结果为1且未持仓，则买入
        if context.predicts[i] == 1 and context.stocks[i] not in context.portfolio.positions.keys():
            log.info('buying %s' % context.stocks[i])
            order_percent(context.stocks[i], 1 / context.num_hold_stocks)
        # 如果预测结果为-1且已持仓，则清仓
        elif context.predicts[i] == -1 and context.stocks[i] in context.portfolio.positions.keys():
            log.info('selling %s' % context.stocks[i])
            order_target(context.stocks[i], 0)


def get_needed_data(current_date, stocks, num_x_days):
    time_start = TransTime(current_date - timedelta(days=num_x_days))
    time_end = TransTime(current_date)
    prices = get_price(stocks,
                       start_date=time_start,
                       end_date=time_end,
                       fields=['open', 'high', 'low', 'close', 'avg_price', 'amp_rate'])
    X_array = []
    for i in range(len(stocks)):
        # 均价
        meanPrice = np.mean(prices[stocks[i]]['avg_price'])
        # 收盘价
        finalPrice = prices[stocks[i]]['close'][-1]
        # 最大值
        maxPrice = max(prices[stocks[i]]['high'])
        # 最小值
        minPrice = min(prices[stocks[i]]['low'])
        # 现价
        nowPrice = prices[stocks[i]]['open'][-1]
        # 涨跌幅
        quoteRate = (prices[stocks[i]]['open'][-1] / prices[stocks[i]]['open'][0]) - 1
        # 标准差
        stdnow = np.std(prices[stocks[i]]['open'])
        # 组织成向量
        X = np.array(
            [finalPrice / meanPrice, maxPrice / meanPrice, minPrice / meanPrice, nowPrice / meanPrice, quoteRate,
             stdnow, nowPrice])
        X_array.append(X)
    X_array = np.array(X_array)
    # 返回当天的所有股票
    return stocks, X_array


# datetime转换格式
def TransTime(date):
    return date.strftime("%Y%m%d")


# 获取训练数据
def create_train(select_data, context):
    label = []
    stock11 = get_index_stocks(symbol='000906.SH', date=select_data.strftime("%Y-%m-%d"))
    stocks, X_array = get_needed_data(select_data, stock11, context.num_x_days)

    nowDate = select_data
    futureDate = select_data + timedelta(days=context.num_y_days)

    # 格式化时间串
    idate = nowDate.strftime("%Y-%m-%d")
    fdate = futureDate.strftime("%Y-%m-%d")
    prices = get_price(stocks, start_date=idate, end_date=fdate, fields=['open'])
    for i in range(len(stocks)):
        if ((prices[stocks[i]]['open'][1] / prices[stocks[i]]['open'][0]) - 1 > 0.0140865):
            label.append(1)
        else:
            label.append(-1)
    return X_array, label


# def optimize_C(X_train, X_test, Y_train, Y_test):
#     # TODO 可能有过拟合问题
#     clf = RandomForestClassifier(n_estimators=100)
#     clf.fit(X_train, Y_train)
#     print(clf.score(X_test, Y_test))
#     return clf


# 设置止损条件
def TakeProfitandStopLoss(context, bar_dict, stocks, predicts, index_profit, index_loss):
    print("根据盈亏线进行止盈止损")
    last_date = get_datetime()
    for i in range(len(stocks)):
        # 如果stock里面信息在用户账户资产信息里面
        if stocks[i] in context.portfolio.positions.keys():
            if context.portfolio.positions[stocks[i]].available_amount > 0:
                purchase_price = context.portfolio.positions[stocks[i]].cost_basis
                dic_close = get_last_minute_close([stocks[i]], last_date)
                last_close3 = float(dic_close[stocks[i]]['close'][-1])
                # last_close = bar_dict[stocks[i]].close
                if (purchase_price != 0):
                    rate = (last_close3 - purchase_price) / purchase_price
                else:
                    rate = 0
                print("股票" + str(stocks[i]) + "：")
                print("购买价格为" + str(purchase_price))
                print("上一分钟收盘价：" + str(last_close3))
                print("收益率：" + str(rate))
                # cumulative_return=bar_dict[stocks[i]].close/context.portfolio.positions[stocks[i]].cost_basis
                # log.info(cumulative_return)
                if rate > index_profit:
                    log.info('()执行止损'.format(last_date))
                    log.info('卖出股票{}'.format(stocks[i]))
                    context.stocks, X_array = get_needed_data(last_date, context.securities, context.num_x_days)
                    context.predicts = context.clf.predict(context.scaler.transform(X_array))
                    if context.predicts[i] == -1:
                        order_target_value(stocks[i], 0)

                if rate < index_loss:
                    log.info('()执行止盈'.format(last_date))
                    log.info('卖出股票{}'.format(stocks[i]))
                    try:
                        order_target_value(stocks[i], 0)
                    except:
                        print("卖不出去")


def get_last_minute_close(symbol_list, current_date):
    justNow = current_date + timedelta(minutes=-1)
    justNow = justNow.strftime('%Y%m%d %H:%M')
    dic_close = get_price(symbol_list, None, justNow, '1m', ['close'], True, None, 30, is_panel=0)
    return dic_close


# 跳过周末，获取想要的下一个交易日。输入为datetime格式
def get_interval_date(date, days):
    date = TransTime(date)
    # 获取时间的时间戳 时间戳表示从1970年1月1日开始按秒计算得到的偏移量 normalize表示将其格式化成午夜值
    date = pd.Timestamp(date).normalize()
    # 显示所有的交易日
    date_in = get_all_trade_days()
    # 用于在排序的数组arr中查找索引
    trade_date = date_in[date_in.searchsorted(date, side='left') + days]
    trade_date = trade_date.strftime("%Y%m%d")
    return trade_date


def length_split(length, group_num, number):
    if length == 0 or group_num == 0:
        return 0, 0

    return (number - 1) * int(length / group_num), number * int(length / group_num) if number != group_num else length


## 收盘后运行函数,用于储存自定义参数、全局变量,执行盘后选股等
def after_trading(context):
    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    # 打印时间
    log.info('{} 盘后运行'.format(time))
    log.info('一天结束')

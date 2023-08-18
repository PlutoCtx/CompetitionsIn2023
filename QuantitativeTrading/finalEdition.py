
import joblib
import numpy as np







def init(context):

    set_benchmark("000906.SH")

    log.info('策略开始运行')

    context.num_x_days = 7

    context.num_y_days = 7

    context.index_rise = 0.02

    context.cycle = 1

    context.index_profit = 0.06

    context.index_loss = 0.06

    context.num_hold_stocks = 500



    context.cycle_flag = True

    context.transfer_positions_date = get_interval_date(get_datetime(), context.cycle_flag)

    get_iwencai('所有A股', 'stocks_list')

    train_date_st = datetime(2012, 1, 1)
    train_date_length = 60
    train_date_step = 30

    X_array, label = create_train(train_date_st, context)
    for i in range(1, train_date_length):
        now_date = train_date_st + timedelta(i * train_date_step)
        unit_X_array, unit_label = create_train(now_date, context)
        X_array = np.concatenate((X_array, unit_X_array), axis=0)
        label += unit_label

    train_size = 60 * train_date_length
    context.scaler = proprocessing.StandardScaler().fit(X_array)


    X_train = X_array[: train_size]

    X_test = X_array[train_size:]
    Y_train = label[: train_size]
    Y_test = label[train_size:]

    model = optimize_C(X_train, X_test, Y_train, Y_test)

    context.scaler = preprocessing.StandardScaler().fit(X_array)
    df_scaler = context.scaler.transform(X_array)
    context.clf = svm.SVC(C=30.15, gama=0.001100009)

    context.clf.fit(df_scaler, np.array(label))
    joblib.dump(context.clf, 'CLF.joblib')
    log.info("模型训练存储")


def before_trading(context):

    date = get_datetime().strftime('%Y-%m-%d %h-%m-%s')

    log.info('{} 盘前运行'.format(date))


def handle_bar(context, bar_dict):

    time = get_datetime().strftime('%Y-%m-%d %h-%m-%s')

    log.info('当天时间' + str(time))


    date = get_datetime()
    today_date = TransTime(date)



    num_stocks = len(list(context.portfolio.positions))



    if context.cycle_flag or today_date == context.transfer_positions_date or num_stocks < 5:
        print("************ 调仓 **************")
        context.cycle_flag = False
        context.transfer_positions_date = get_interval_date(get_datetime(), context.cycle)
        print("下一个调仓日：" + str(context.transfer_positions_date))
        buy_and_sell(context, bar_dict, date)


        TakeProfitAndStopLoss(context, bar_dict, context.stocks, context.predicts, context.index_profit, context.index_loss)

def buy_and_sell(context, bar_dict, date):

    context.group_num = 1

    context.number = 1

    context.clean_ty = 'median_extremum-standardize'

    context.sys_factors = {}









    context.securities = []
    nowtime_str = get_last_datetime.strftime("%Y-%m-%d")

    securities_df = sfactor_stock_scanner(
        context.stocks_list,
        query(factor.symbol, factor.equity_ratio, factor.weighted_roe, factor.parent_company_share_holders_net_profit_growth_ratio, factor.pe, factor.turnover_ratio_of_receivable, factor.bias, factor.macd, factor.obv, factor.arbr, factor.boll),
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


        print(context.secruities)
        context.stocks, X_array = get_needed_data(date, context.secruities, context.num_x_days)
        context.predicts = context.clf.predict(context.scaler.transform(X_array))
        for i in range(len(context.stocks)):
            






















# !/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: yjp
# @software: PyCharm
# @file: task002.py
# @time: 2022-07-23 22:02
from matplotlib import pyplot as plt

# plt.style.use('fivethirtyeight')

languages = [ 'turnover_ratio_of_account_payable', 'weighted_roe', 'opt_profit_div_income', 'before_tax_profit_div_income',
             'current_ratio', 'net_profit_margin_on_sales', 'overall_assets_net_income_ratio','net_profit_growth_ratio','pb','market_cap']

popularity = [1, 1.5, 2, 3.3, 5, 6.1, 7.2, 8, 9, 19]

plt.barh(languages, popularity)



plt.tight_layout()

plt.show()
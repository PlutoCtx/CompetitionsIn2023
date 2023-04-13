# !/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: yjp
# @software: PyCharm
# @file: task01.py
# @time: 2022-07-23 0:40
import numpy as np
from matplotlib import pyplot as plt



languages = [ 'cdp', 'ma', 'arbr', 'cr',
             'psy', 'obv', 'pvt','bias','mtm','boll']

popularity = [1, 1.5, 2, 3.3, 5, 6.1, 7.2, 8, 9, 19]

plt.barh(languages, popularity)



plt.tight_layout()

plt.show()
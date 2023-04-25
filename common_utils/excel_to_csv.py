# @Version: python3.10
# @Time: 2023/4/25 13:39
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: excel_to_csv.py
# @Software: PyCharm
# @User: chent

import pandas as pd
data = pd.read_excel('D:\桌面\enWords.xlsx', 'Sheet1')
data.to_csv('data.csv', encoding='utf-8')


# import pandas as pd
# def xlsx_to_csv_pd():
#     data_xls = pd.read_excel('D:\桌面\enWords.xlsx', 'Words', index_col=0)
#     data_xls.to_csv('data01.csv', encoding='utf-8')
# if __name__ == '__main__':
#     xlsx_to_csv_pd()

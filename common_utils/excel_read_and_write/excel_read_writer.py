# @Version: python3.10
# @Time: 2023/4/25 11:54
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: excel_read_writer.py
# @Software: PyCharm
# @User: chent

import openpyxl
data = openpyxl.load_workbook("D:\桌面\enWords00.xlsx")   # 读取xlsx文件
table = data.get_sheet_by_name('Words')     # 获得指定名称的页
nrows = table.rows  # 获得行数 类型为迭代器
ncols = table.columns   # 获得列数 类型为迭代器

print(type(nrows))
for row in nrows:
    # print(row[8])  # 包含了页名，cell，值
    line = [col.value for col in row] # 取值
    table[9][row] = line[8][:3]
    print(line[8][:3])

# 读取单元格
print(table.cell(1, 1).value)
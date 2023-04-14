# @Version: python3.10
# @Time: 2023/4/14 20:02
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: loop_through_folder.py
# @Software: PyCharm
# @User: chent


import os
from mask import create_mask

# 下载的文件
folder_path = r"D:\ProgramingCodes\PycharmProjects\CompetitionsIn2023\ServiceOutsourcingInnovationandEntrepreneurshipCompetition\attempt02_heima\_31faceMaskDetector\Downloads"

# os.listdir(folder_path) 返回指定文件夹下的文件/文件夹名字的列表
image = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i in range(len(image)):
    print("the path of the image is", image[i])
    # image = cv2.imread(images[i])
    # c=c+1
    create_mask(image[i])

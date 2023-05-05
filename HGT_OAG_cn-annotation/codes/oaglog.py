# -*-  coding: utf-8 -*-
# @Time      :2021/3/23 9:33
# @Author    :huangzg28153
# @File      :oaglog.py
# @Software  :PyCharm
import logging
import time
import os
#
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 创建handler，用于写入日志文件
date_time = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
log_path = os.path.join(os.path.dirname(__file__)+'/logs/')
if not os.path.exists(log_path):
    os.mkdir(log_path)
#
log_name = log_path + date_time + '.log'
file_handler = logging.FileHandler(log_name,mode='w',encoding='utf-8')
file_handler.setLevel(logging.INFO)
# 定义输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - "
                              "%(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
# 将logger添加入handler中
logger.addHandler(file_handler)
# console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


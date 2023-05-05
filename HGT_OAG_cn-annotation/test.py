# -*-  coding: utf-8 -*-
# @Time      :2021/3/22 20:56
# @Author    :huangzg28153
# @File      :test.py
# @Software  :PyCharm
import numpy as np
import pandas as pd
type = [0,1,1,1,2,0,1,0,1,2,2,0]
ser = [0,1,2,3,4,5,6,0,1,2,3,4]
layer = [0,0,0,0,0,1,1,0,0,0,0,1]
sample = [0,0,0,0,0,0,0,1,1,1,1,1]

df = pd.DataFrame({"type":type,"ser":ser,"layer":layer,"sample":sample})


df.sort_values(by=["ser",'type',"sample","layer"],axis=0)
df.sort_values(by=["layer","sample","type","ser"],axis=0)
df.sort_values(by=["type","layer","sample","ser"],axis=0)
df['order'] = [0,2,4,5,6,9,11,1,3,7,8,10]
df = df.sort_values(by=['order'],axis=0)
df.sort_values(by=['layer','ser','type','sample'],axis=0)
df.sort_values(by=["sample","type",'ser',"layer"],axis=0)

########################################################
df.sort_values(by=['layer',"type","sample","ser"],axis=0).reset_index().index
#######################################################
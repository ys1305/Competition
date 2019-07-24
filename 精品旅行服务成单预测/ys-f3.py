
# coding: utf-8

import numpy as np
import pandas as pd


# action_info = pd.read_csv(
#     "../data/data/trainingset/action_train.csv", sep=",", encoding="UTF-8")  # 读取用户行为数据表


action_info_test = pd.read_csv(
    "../data/data/test/action_test.csv", sep=",", encoding="UTF-8")  # 读取用户行为数据表


def fun_action(action_info):

    def type_num(df, args):
        return len(df.loc[:, args])

    action_num = action_info.groupby(
        "userid").apply(type_num, args="actionType")

    # In[5]:

    action_num = pd.DataFrame(action_num)
    action_num.columns = ["F3.1"]
    # F3.1 所有动作_总次数

    def sum_action(df, args, scale):
        value_count = pd.value_counts(df.loc[:, args])
        sum_c = 0
        for i in value_count.index:
            if i in scale:
                sum_c += value_count[i]
        return sum_c
    # 按照id分组后统计次数
    action_1234 = action_info.groupby("userid").apply(
        sum_action, args="actionType", scale=[1, 2, 3, 4])
    action_56789 = action_info.groupby("userid").apply(
        sum_action, args="actionType", scale=[5, 6, 7, 8, 9])

    # In[9]:

    action_1234 = pd.DataFrame(action_1234)
    action_56789 = pd.DataFrame(action_56789)
    action_1234.columns = ["F3.2"]
    action_56789.columns = ["F3.3"]
    # F3.2非支付动作次数
    # F3.3支付动作次数

    feature = pd.merge(action_num, action_1234, on="userid")
    feature = pd.merge(feature, action_56789, on="userid")

    # 筛选出每个动作并建立新表
    action_1 = action_info[action_info["actionType"].isin([1])]
    action_2 = action_info[action_info["actionType"].isin([2])]
    action_3 = action_info[action_info["actionType"].isin([3])]
    action_4 = action_info[action_info["actionType"].isin([4])]
    action_5 = action_info[action_info["actionType"].isin([5])]
    action_6 = action_info[action_info["actionType"].isin([6])]
    action_7 = action_info[action_info["actionType"].isin([7])]
    action_8 = action_info[action_info["actionType"].isin([8])]
    action_9 = action_info[action_info["actionType"].isin([9])]

    # 分组后取出来某一列进行统计操作，否则是对所有列进行统计操作
    feature["F3.4"] = action_1.groupby("userid")["actionType"].count()  # 动作1次数
    feature["F3.5"] = action_2.groupby("userid")["actionType"].count()  # 动作2次数
    feature["F3.6"] = action_3.groupby("userid")["actionType"].count()  # 动作3次数
    feature["F3.7"] = action_4.groupby("userid")["actionType"].count()  # 动作4次数
    feature["F3.8"] = action_5.groupby("userid")["actionType"].count()  # 动作5次数
    feature["F3.9"] = action_6.groupby("userid")["actionType"].count()  # 动作6次数
    feature["F3.10"] = action_7.groupby(
        "userid")["actionType"].count()  # 动作7次数
    feature["F3.11"] = action_8.groupby(
        "userid")["actionType"].count()  # 动作8次数
    feature["F3.12"] = action_9.groupby(
        "userid")["actionType"].count()  # 动作9次数

    feature["F3.13"] = feature["F3.2"] / feature["F3.1"]  # 非支付动作占比
    feature["F3.14"] = feature["F3.3"] / feature["F3.1"]  # 支付动作占比
    feature["F3.15"] = feature["F3.4"] / feature["F3.1"]  # 动作1占比
    feature["F3.16"] = feature["F3.5"] / feature["F3.1"]  # 动作2占比
    feature["F3.17"] = feature["F3.6"] / feature["F3.1"]  # 动作3占比
    feature["F3.18"] = feature["F3.7"] / feature["F3.1"]  # 动作4占比
    feature["F3.19"] = feature["F3.8"] / feature["F3.1"]  # 动作5占比
    feature["F3.20"] = feature["F3.9"] / feature["F3.1"]  # 动作6占比
    feature["F3.21"] = feature["F3.10"] / feature["F3.1"]  # 动作7占比
    feature["F3.22"] = feature["F3.11"] / feature["F3.1"]  # 动作8占比
    feature["F3.23"] = feature["F3.12"] / feature["F3.1"]  # 动作9占比

    feature = feature.fillna(0)  # 空值填补0

    def time_gap_mean(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()  # diff计算时间间隔，dropna删除空值
        return d.mean()

    actiontime_gap_mean = action_info.groupby(
        "userid").apply(time_gap_mean, args="actionTime")
    actiontime_gap_mean = pd.DataFrame(actiontime_gap_mean)
    actiontime_gap_mean.columns = ["F3.24"]
    # F3.24时间间隔均值
    feature = feature.merge(actiontime_gap_mean, on="userid", how="left")

    def time_gap_var(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()
        return d.var()

    actiontime_gap_var = action_info.groupby(
        "userid").apply(time_gap_var, args="actionTime")
    actiontime_gap_var = pd.DataFrame(actiontime_gap_var)
    actiontime_gap_var.columns = ["F3.25"]
    # F3.25时间间隔方差
    feature = feature.merge(actiontime_gap_var, on="userid", how="left")

    def time_gap_min(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()
        return d.min()

    actiontime_gap_min = action_info.groupby(
        "userid").apply(time_gap_min, args="actionTime")
    actiontime_gap_min = pd.DataFrame(actiontime_gap_min)
    actiontime_gap_min.columns = ["F3.26"]
    # F3.26时间间隔最小值
    feature = feature.merge(actiontime_gap_min, on="userid", how="left")

    def time_gap_max(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()
        return d.max()

    actiontime_gap_max = action_info.groupby(
        "userid").apply(time_gap_max, args="actionTime")
    actiontime_gap_max = pd.DataFrame(actiontime_gap_max)
    actiontime_gap_max.columns = ["F3.27"]
    # F3.27时间间隔最大值
    feature = feature.merge(actiontime_gap_max, on="userid", how="left")

    def last_time_gap(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()
        return d.iloc[-1] if len(d) > 1 else 0

    action_last_time_gap = action_info.groupby(
        "userid").apply(last_time_gap, args="actionTime")
    action_last_time_gap = pd.DataFrame(action_last_time_gap)
    action_last_time_gap.columns = ["F3.28"]
    # 最后一个动作时间间隔

    feature = feature.merge(action_last_time_gap, on="userid", how="left")

    def last2_time_gap(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()
        return d.iloc[-2] if len(d) > 2 else 0

    action_last2_time_gap = action_info.groupby(
        "userid").apply(last2_time_gap, args="actionTime")

    action_last2_time_gap = pd.DataFrame(action_last2_time_gap)
    action_last2_time_gap.columns = ["F3.29"]
    feature = feature.merge(action_last2_time_gap, on="userid", how="left")

    def last3_time_gap(df, args):
        t = df.loc[:, args]
        d = t.diff().dropna()
        return d.iloc[-3] if len(d) > 3 else 0

    action_last3_time_gap = action_info.groupby(
        "userid").apply(last3_time_gap, args="actionTime")

    action_last3_time_gap = pd.DataFrame(action_last3_time_gap)
    action_last3_time_gap.columns = ["F3.30"]
    # 倒数第三个动作时间间隔
    feature = feature.merge(action_last3_time_gap, on="userid", how="left")

    def last_type(df, args):
        t = list(df.loc[:, args])
        return t[-1]

    action_last_type = action_info.groupby(
        "userid").apply(last_type, args="actionType")
    action_last_type = pd.DataFrame(action_last_type)
    action_last_type.columns = ["F3.31"]
    # 最后一个动作行为
    feature = feature.merge(action_last_type, on="userid", how="left")

    # In[70]:

    def last2_type(df, args):
        t = list(df.loc[:, args])
        lt = len(t)
        l2 = 0
        if lt > 1:
            l2 = t[-2]
        return l2

    action_last2_type = action_info.groupby(
        "userid").apply(last2_type, args="actionType")
    action_last2_type = pd.DataFrame(action_last2_type)
    action_last2_type.columns = ["F3.32"]
    # 倒数第二个动作行为

    feature = feature.merge(action_last2_type, on="userid", how="left")

    # In[73]:

    def last3_type(df, args):
        t = list(df.loc[:, args])
        lt = len(t)
        l3 = 0
        if lt > 2:
            l3 = t[-3]
        return l3

    action_last3_type = action_info.groupby(
        "userid").apply(last3_type, args="actionType")
    action_last3_type = pd.DataFrame(action_last3_type)
    action_last3_type.columns = ["F3.33"]
    # 倒数第三个动作行为

    feature = feature.merge(action_last3_type, on="userid", how="left")

    # 倒数三个动作时间间隔均值
    feature["F3.34"] = (feature["F3.28"] +
                        feature["F3.29"] + feature["F3.30"]) / 3
    # 倒数三个动作时间间隔方差
    feature["F3.35"] = ((feature["F3.28"] - feature["F3.34"])**2 + (feature["F3.29"] -
                                                                    feature["F3.34"])**2 + (feature["F3.30"] - feature["F3.34"])**2) / 3

    last = action_info.groupby("userid")["actionTime"].max()
    last1 = action_1.groupby("userid")["actionTime"].max()
    last2 = action_2.groupby("userid")["actionTime"].max()
    last3 = action_3.groupby("userid")["actionTime"].max()
    last4 = action_4.groupby("userid")["actionTime"].max()
    last5 = action_5.groupby("userid")["actionTime"].max()
    last6 = action_6.groupby("userid")["actionTime"].max()
    last7 = action_7.groupby("userid")["actionTime"].max()
    last8 = action_8.groupby("userid")["actionTime"].max()
    last9 = action_9.groupby("userid")["actionTime"].max()

    last = pd.DataFrame(last)
    last1 = pd.DataFrame(last1)
    last2 = pd.DataFrame(last2)
    last3 = pd.DataFrame(last3)
    last4 = pd.DataFrame(last4)
    last5 = pd.DataFrame(last5)
    last6 = pd.DataFrame(last6)
    last7 = pd.DataFrame(last7)
    last8 = pd.DataFrame(last8)
    last9 = pd.DataFrame(last9)
    last.columns = ["time"]
    last1.columns = ["time"]
    last2.columns = ["time"]
    last3.columns = ["time"]
    last4.columns = ["time"]
    last5.columns = ["time"]
    last6.columns = ["time"]
    last7.columns = ["time"]
    last8.columns = ["time"]
    last9.columns = ["time"]

    feature["F3.36"] = last["time"] - last1["time"]
    feature["F3.37"] = last["time"] - last2["time"]
    feature["F3.38"] = last["time"] - last3["time"]
    feature["F3.39"] = last["time"] - last4["time"]
    feature["F3.40"] = last["time"] - last5["time"]
    feature["F3.41"] = last["time"] - last6["time"]
    feature["F3.42"] = last["time"] - last7["time"]
    feature["F3.43"] = last["time"] - last8["time"]
    feature["F3.44"] = last["time"] - last9["time"]

    action_1_mean = action_1.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_1_var = action_1.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_1_min = action_1.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_1_max = action_1.groupby("userid").apply(
        time_gap_max, args="actionTime")

    action_2_mean = action_2.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_2_var = action_2.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_2_min = action_2.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_2_max = action_2.groupby("userid").apply(
        time_gap_max, args="actionTime")

    action_3_mean = action_3.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_3_var = action_3.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_3_min = action_3.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_3_max = action_3.groupby("userid").apply(
        time_gap_max, args="actionTime")

    action_4_mean = action_4.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_4_var = action_4.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_4_min = action_4.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_4_max = action_4.groupby("userid").apply(
        time_gap_max, args="actionTime")

    # In[57]:

    action_5_mean = action_5.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_5_var = action_5.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_5_min = action_5.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_5_max = action_5.groupby("userid").apply(
        time_gap_max, args="actionTime")

    # In[58]:

    action_6_mean = action_6.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_6_var = action_6.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_6_min = action_6.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_6_max = action_6.groupby("userid").apply(
        time_gap_max, args="actionTime")

    # In[59]:

    action_7_mean = action_7.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_7_var = action_7.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_7_min = action_7.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_7_max = action_7.groupby("userid").apply(
        time_gap_max, args="actionTime")

    # In[60]:

    action_8_mean = action_8.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_8_var = action_8.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_8_min = action_8.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_8_max = action_8.groupby("userid").apply(
        time_gap_max, args="actionTime")

    # In[61]:

    action_9_mean = action_9.groupby("userid").apply(
        time_gap_mean, args="actionTime")
    action_9_var = action_9.groupby("userid").apply(
        time_gap_var, args="actionTime")
    action_9_min = action_9.groupby("userid").apply(
        time_gap_min, args="actionTime")
    action_9_max = action_9.groupby("userid").apply(
        time_gap_max, args="actionTime")

    # In[62]:

    df1_1 = pd.DataFrame(action_1_mean)
    df1_2 = pd.DataFrame(action_1_var)
    df1_3 = pd.DataFrame(action_1_min)
    df1_4 = pd.DataFrame(action_1_max)
    df1_1.columns = ["F3.45"]
    df1_2.columns = ["F3.46"]
    df1_3.columns = ["F3.47"]
    df1_4.columns = ["F3.48"]

    df2_1 = pd.DataFrame(action_2_mean)
    df2_2 = pd.DataFrame(action_2_var)
    df2_3 = pd.DataFrame(action_2_min)
    df2_4 = pd.DataFrame(action_2_max)
    df2_1.columns = ["F3.49"]
    df2_2.columns = ["F3.50"]
    df2_3.columns = ["F3.51"]
    df2_4.columns = ["F3.52"]

    df3_1 = pd.DataFrame(action_3_mean)
    df3_2 = pd.DataFrame(action_3_var)
    df3_3 = pd.DataFrame(action_3_min)
    df3_4 = pd.DataFrame(action_3_max)
    df3_1.columns = ["F3.53"]
    df3_2.columns = ["F3.54"]
    df3_3.columns = ["F3.55"]
    df3_4.columns = ["F3.56"]

    df4_1 = pd.DataFrame(action_4_mean)
    df4_2 = pd.DataFrame(action_4_var)
    df4_3 = pd.DataFrame(action_4_min)
    df4_4 = pd.DataFrame(action_4_max)
    df4_1.columns = ["F3.57"]
    df4_2.columns = ["F3.58"]
    df4_3.columns = ["F3.59"]
    df4_4.columns = ["F3.60"]

    df5_1 = pd.DataFrame(action_5_mean)
    df5_2 = pd.DataFrame(action_5_var)
    df5_3 = pd.DataFrame(action_5_min)
    df5_4 = pd.DataFrame(action_5_max)
    df5_1.columns = ["F3.61"]
    df5_2.columns = ["F3.62"]
    df5_3.columns = ["F3.63"]
    df5_4.columns = ["F3.64"]

    df6_1 = pd.DataFrame(action_6_mean)
    df6_2 = pd.DataFrame(action_6_var)
    df6_3 = pd.DataFrame(action_6_min)
    df6_4 = pd.DataFrame(action_6_max)
    df6_1.columns = ["F3.65"]
    df6_2.columns = ["F3.66"]
    df6_3.columns = ["F3.67"]
    df6_4.columns = ["F3.68"]

    # In[68]:

    df7_1 = pd.DataFrame(action_7_mean)
    df7_2 = pd.DataFrame(action_7_var)
    df7_3 = pd.DataFrame(action_7_min)
    df7_4 = pd.DataFrame(action_7_max)
    df7_1.columns = ["F3.69"]
    df7_2.columns = ["F3.70"]
    df7_3.columns = ["F3.71"]
    df7_4.columns = ["F3.72"]

    # In[69]:

    df8_1 = pd.DataFrame(action_8_mean)
    df8_2 = pd.DataFrame(action_8_var)
    df8_3 = pd.DataFrame(action_8_min)
    df8_4 = pd.DataFrame(action_8_max)
    df8_1.columns = ["F3.73"]
    df8_2.columns = ["F3.74"]
    df8_3.columns = ["F3.75"]
    df8_4.columns = ["F3.76"]

    # In[70]:

    df9_1 = pd.DataFrame(action_9_mean)
    df9_2 = pd.DataFrame(action_9_var)
    df9_3 = pd.DataFrame(action_9_min)
    df9_4 = pd.DataFrame(action_9_max)
    df9_1.columns = ["F3.77"]
    df9_2.columns = ["F3.78"]
    df9_3.columns = ["F3.79"]
    df9_4.columns = ["F3.80"]

    # In[71]:

    feature = feature.merge(df1_1, on="userid", how="left")
    feature = feature.merge(df1_2, on="userid", how="left")
    feature = feature.merge(df1_3, on="userid", how="left")
    feature = feature.merge(df1_4, on="userid", how="left")
    feature = feature.merge(df2_1, on="userid", how="left")
    feature = feature.merge(df2_2, on="userid", how="left")
    feature = feature.merge(df2_3, on="userid", how="left")
    feature = feature.merge(df2_4, on="userid", how="left")
    feature = feature.merge(df3_1, on="userid", how="left")
    feature = feature.merge(df3_2, on="userid", how="left")
    feature = feature.merge(df3_3, on="userid", how="left")
    feature = feature.merge(df3_4, on="userid", how="left")
    feature = feature.merge(df4_1, on="userid", how="left")
    feature = feature.merge(df4_2, on="userid", how="left")
    feature = feature.merge(df4_3, on="userid", how="left")
    feature = feature.merge(df4_4, on="userid", how="left")
    feature = feature.merge(df5_1, on="userid", how="left")
    feature = feature.merge(df5_2, on="userid", how="left")
    feature = feature.merge(df5_3, on="userid", how="left")
    feature = feature.merge(df5_4, on="userid", how="left")
    feature = feature.merge(df6_1, on="userid", how="left")
    feature = feature.merge(df6_2, on="userid", how="left")
    feature = feature.merge(df6_3, on="userid", how="left")
    feature = feature.merge(df6_4, on="userid", how="left")
    feature = feature.merge(df7_1, on="userid", how="left")
    feature = feature.merge(df7_2, on="userid", how="left")
    feature = feature.merge(df7_3, on="userid", how="left")
    feature = feature.merge(df7_4, on="userid", how="left")
    feature = feature.merge(df8_1, on="userid", how="left")
    feature = feature.merge(df8_2, on="userid", how="left")
    feature = feature.merge(df8_3, on="userid", how="left")
    feature = feature.merge(df8_4, on="userid", how="left")
    feature = feature.merge(df9_1, on="userid", how="left")
    feature = feature.merge(df9_2, on="userid", how="left")
    feature = feature.merge(df9_3, on="userid", how="left")
    feature = feature.merge(df9_4, on="userid", how="left")

    return feature


feature = fun_action(action_info_test)
feature.to_csv("../data/workeddata/F3ys-fun-test.csv",
               encoding="gb2312", index=True)

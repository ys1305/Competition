
# coding: utf-8


import numpy as np
import pandas as pd
# 全部行都能输出

table_target = pd.read_csv(
    "../data/data/trainingset/orderFuture_train.csv", sep=",", encoding="UTF-8")  # 读取目标属性表
# user_info = pd.read_csv("../data/data/trainingset/userProfile_train.csv",
#                         sep=",", encoding="UTF-8")  # 读取用户基本属性表
# order_info = pd.read_csv(
#     "../data/data/trainingset/orderHistory_train.csv", sep=",", encoding="UTF-8")  # 读取历史订单数据表
# 没有使用action_info
# user_com = pd.read_csv("../data/data/trainingset/userComment_train.csv",sep=',',encoding="UTF-8")


user_info = pd.read_csv("../data/data/test/userProfile_test.csv",
                        sep=",", encoding="UTF-8")  # 读取用户基本属性表
order_info = pd.read_csv(
    "../data/data/test/orderHistory_test.csv", sep=",", encoding="UTF-8")  # 读取历史订单数据表


def fun_user(user_info, order_info):
    order_info["orderTime"] = pd.to_datetime(order_info["orderTime"], unit="s")

    # ## 一、数据探索
    # ### 1. 用户基本属性表
    # #### 1.1 数据转换
    # 原始数据需要经过转换才能符合建模需求，例如：生成新字段、重新分类、变换哑变量、去除重复数据、过滤极端\异常值、填补缺失值、变量聚类、离散化等

    # In[47]:

    user_info = user_info.fillna("未知")  # 空值填补

    # #### 1.2 数据探索
    # 除了describe计算基本统计量外，还需要探索目标变量和自变量以及自变量之间的关系，以决定哪些变量入选模型，这种数据探索往往结合图形来说明。

    # 常用的绘图包包括matplotlib、seaborn等

    user_orderType = user_info.merge(table_target, on="userid", how="left")
    # 合并

    gender_orderType = user_orderType.groupby("gender", as_index=False)[
        "orderType"].agg({"总数": np.size, "精品订单数": np.sum})
    gender_orderType["精品订单率"] = gender_orderType["精品订单数"] / \
        gender_orderType["总数"]
    gender_orderType.head()

    # In[62]:

    province_orderType = user_orderType.groupby("province", as_index=False)[
        "orderType"].agg({"总数": np.size, "精品订单数": np.sum})
    province_orderType["精品订单率"] = province_orderType["精品订单数"] / \
        province_orderType["总数"]
    province_orderType.sort_values(by="精品订单数", ascending=False)

    # In[63]:

    age_orderType = user_orderType.groupby("age", as_index=False)[
        "orderType"].agg({"总数": np.size, "精品订单数": np.sum})
    age_orderType["精品订单率"] = age_orderType["精品订单数"] / age_orderType["总数"]
    age_orderType.sort_values(by="精品订单率", ascending=False)

    # ## 筛选变量
    # 筛选变量时可以应用专业知识，选取与目标字段相关性较高的字段用于建模，也可通过分析现有数据，用统计量辅助选择
    # 为了增强模型稳定性，自变量之间最好相互独立，可运用统计方法选择要排除的变量或进行变量聚类

    def type_count(df, type, args):
        count = len(df[df.loc[:, args] == type])
        return count

    type1_amount = order_info.groupby("userid").apply(
        type_count, 1, args="orderType")  # 精品订单个数
    type0_amount = order_info.groupby("userid").apply(
        type_count, 0, args="orderType")  # 普通订单个数

    # In[68]:

    type1_amount = pd.DataFrame(type1_amount)
    type0_amount = pd.DataFrame(type0_amount)
    type1_amount.columns = ["F2.3"]
    type0_amount.columns = ["F2.0"]
    # 列名

    # 订单总数
    df1 = type1_amount.merge(type0_amount, on="userid", how="left")
    df1["F2.1"] = df1["F2.3"] + df1["F2.0"]

    # 删除无用的列
    df1 = df1.drop("F2.0", 1)
    df1 = df1.drop("F2.3", 1)
    # 是否有精品订单

    def type_count(df, type, args):
        count = len(df[df.loc[:, args] == type])
        return count

    type1_exist = order_info.groupby("userid").apply(
        type_count, 1, args="orderType")
    type1_exist[type1_exist > 0] = 1
    type1_exist = pd.DataFrame(type1_exist)
    type1_exist.columns = ["F2.2"]

    feature = df1.merge(type1_exist, on="userid", how="left")
    feature = feature.merge(type1_amount, on="userid", how="left")

    # 精品订单占比
    feature["F2.4"] = feature["F2.3"] / feature["F2.1"]

    # 订单城市最大次数
    df3 = order_info.groupby(["userid", "city"])["orderid"].count()

    df3 = pd.DataFrame(df3)
    df3.columns = ["Freq"]
    feature["F2.5"] = df3.groupby("userid")["Freq"].max()
    # F2.5 订单城市最大次数

    feature["F2.6"] = df3.groupby("userid").count()
    # F2.6 订单城市个数

    # #### 其后关于国家/地区、大洲的操作方法与之相同。

    # 订单国家最大次数
    df4 = order_info.groupby(["userid", "country"])["orderid"].count()

    df4 = pd.DataFrame(df4)
    df4.columns = ["Freq"]

    # In[95]:

    feature["F2.7"] = df4.groupby("userid")["Freq"].max()
    # F2.7 订单国家最大次数

    feature["F2.8"] = df4.groupby("userid").count()

    # F2.8 订单国家个数

    # In[97]:

    # 订单大洲最大次数
    df5 = order_info.groupby(["userid", "continent"])["orderid"].count()
    df5 = pd.DataFrame(df5)
    df5.columns = ["Freq"]
    feature["F2.9"] = df5.groupby("userid")["Freq"].max()
    # F2.9 订单大洲最大次数
    feature["F2.10"] = df5.groupby("userid").count()

    # F2.10 订单大洲个数
    # 筛选出精品订单
    # 利用Pandas cookbook输出值为1的行
    temp1 = order_info[order_info["orderType"].isin([1])]
    df6 = temp1.groupby(["userid", "city"])["orderid"].count()
    df6 = pd.DataFrame(df6)
    df6.columns = ["Freq"]
    feature["F2.11"] = df6.groupby("userid")["Freq"].max()
    # F2.11精品订单城市最大次数
    feature["F2.12"] = df6.groupby("userid").count()
    # F2.12精品订单城市个数

    # In[109]:

    df7 = temp1.groupby(["userid", "country"])["orderid"].count()
    df7 = pd.DataFrame(df7)
    df7.columns = ["Freq"]
    feature["F2.13"] = df7.groupby("userid")["Freq"].max()
    # F2.13精品订单国家最大次数
    feature["F2.14"] = df7.groupby("userid").count()
    # F2.14精品订单国家个数

    df8 = temp1.groupby(["userid", "continent"])["orderid"].count()
    df8 = pd.DataFrame(df8)
    df8.columns = ["Freq"]
    feature["F2.15"] = df8.groupby("userid")["Freq"].max()
    # F2.15精品订单大洲最大次数
    feature["F2.16"] = df8.groupby("userid").count()
    # F2.16精品订单大洲个数

    # 计算时间间隔
    time_gap_max = order_info.groupby("userid")["orderTime"].max()
    time_gap_min = order_info.groupby("userid")["orderTime"].min()
    time_gap_max = pd.DataFrame(time_gap_max)
    time_gap_min = pd.DataFrame(time_gap_min)
    time_gap = time_gap_max.merge(time_gap_min, on="userid", how="left")

    # feature["F2.17"] = (time_gap["orderTime_x"] -
    #                     time_gap["orderTime_y"]) / feature["F2.1"]

    f217 = (time_gap["orderTime_x"] -
            time_gap["orderTime_y"]) / feature["F2.1"]

    f217 = pd.DataFrame(f217)

    bf217 = f217.dtypes != object

    if isinstance(bf217, bool):
        if bf217:
            f217[0] = f217[0].values / np.timedelta64(1, 'ns')
    else:
        if bf217.bool():
            f217[0] = f217[0].values / np.timedelta64(1, 'ns')

    feature["F2.17"] = f217

    # F2.17订单平均时间间隔
    time_gap_max_1 = temp1.groupby("userid")["orderTime"].max()
    time_gap_min_1 = temp1.groupby("userid")["orderTime"].min()
    time_gap_max_1 = pd.DataFrame(time_gap_max_1)
    time_gap_min_1 = pd.DataFrame(time_gap_min_1)
    time_gap_1 = time_gap_max_1.merge(time_gap_min_1, on="userid", how="left")

    f23 = pd.DataFrame(feature.loc[time_gap_1.index]["F2.3"])

    # feature["F2.18"] = (time_gap_1["orderTime_x"] -
    #                     time_gap_1["orderTime_y"]) / f23['F2.3']

    f218 = (time_gap_1["orderTime_x"] -
            time_gap_1["orderTime_y"]) / f23['F2.3']
    f218 = pd.DataFrame(f218)
    bf218 = f218.dtypes != object
    if isinstance(bf218, bool):
        if bf218:
            f218[0] = f218[0].values / np.timedelta64(1, 'ns')
    else:
        if bf218.bool():
            f218[0] = f218[0].values / np.timedelta64(1, 'ns')
    feature["F2.18"] = f218

    temp2 = order_info[order_info["city"].isin(
        ["新加坡", "东京", "纽约", "台北", "吉隆坡", "悉尼", "香港", "大阪", "墨尔本", "曼谷"])]
    df9 = temp2.groupby(["userid", "city"])["orderid"].count()
    df9 = pd.DataFrame(df9)
    df9.columns = ["Freq"]
    feature["F2.19"] = df9.groupby("userid").sum()  # 订单热门城市访问次数
    df9[df9["Freq"] > 0] = 1
    feature["F2.20"] = df9.groupby("userid").max()  # 订单热门城市是否访问
    feature["F2.21"] = df9.groupby("userid").count()  # 订单热门城市的数量
    temp3 = order_info[order_info["country"].isin(
        ["日本", "美国", "澳大利亚", "新加坡", "泰国", "马来西亚", "中国台湾", "中国香港", "法国", "英国"])]
    df10 = temp3.groupby(["userid", "country"])["orderid"].count()
    df10 = pd.DataFrame(df10)
    df10.columns = ["Freq"]
    feature["F2.22"] = df10.groupby("userid").sum()  # 订单热门国家访问次数
    df10[df10["Freq"] > 0] = 1
    feature["F2.23"] = df10.groupby("userid").max()  # 订单热门国家是否访问
    feature["F2.24"] = df10.groupby("userid").count()  # 订单热门国家的数量
    temp4 = order_info[order_info["continent"].isin(["亚洲", "北美洲"])]

    df11 = temp4.groupby(["userid", "continent"])["orderid"].count()
    df11 = pd.DataFrame(df11)
    df11.columns = ["Freq"]
    feature["F2.25"] = df11.groupby("userid").sum()  # 订单热门大洲访问次数

    df11[df11["Freq"] > 0] = 1
    feature["F2.26"] = df11.groupby("userid").max()  # 订单热门大洲是否访问
    feature["F2.27"] = df11.groupby("userid").count()  # 订单热门大洲的数量

    temp2_1 = temp1[temp1["city"].isin(
        ["东京", "大阪", "台北", "京都", "曼谷", "巴厘岛", "墨尔本", "巴黎", "吉隆坡", "伦敦"])]
    df9_1 = temp2_1.groupby(["userid", "city"])["orderid"].count()
    df9_1 = pd.DataFrame(df9_1)
    df9_1.columns = ["Freq"]
    feature["F2.28"] = df9_1.groupby("userid").sum()  # 精品订单热门城市访问次数

    df9_1[df9_1["Freq"] > 0] = 1
    feature["F2.29"] = df9_1.groupby("userid").max()  # 精品订单热门城市是否访问
    feature["F2.30"] = df9_1.groupby("userid").count()  # 精品订单热门城市的数量

    temp3_1 = temp1[temp1["country"].isin(
        ["日本", "美国", "中国台湾", "澳大利亚", "泰国", "法国", "印度尼西亚", "英国", "马来西亚", "韩国"])]
    df10_1 = temp3_1.groupby(["userid", "country"])["orderid"].count()
    df10_1 = pd.DataFrame(df10_1)
    df10_1.columns = ["Freq"]
    feature["F2.31"] = df10_1.groupby("userid").sum()  # 精品订单热门国家访问次数

    # In[258]:

    df10_1[df10_1["Freq"] > 0] = 1
    feature["F2.32"] = df10_1.groupby("userid").max()  # 精品订单热门国家是否访问
    feature["F2.33"] = df10_1.groupby("userid").count()  # 精品订单热门国家的数量

    temp4_1 = temp1[temp1["continent"].isin(["亚洲", "欧洲"])]
    df11_1 = temp4_1.groupby(["userid", "continent"])["orderid"].count()
    df11_1 = pd.DataFrame(df11_1)
    df11_1.columns = ["Freq"]

    # In[262]:

    feature["F2.34"] = df11_1.groupby("userid").sum()  # 订单热门大洲访问次数
    df11_1[df11_1["Freq"] > 0] = 1
    feature["F2.35"] = df11_1.groupby("userid").max()  # 订单热门大洲是否访问
    feature["F2.36"] = df11_1.groupby("userid").count()  # 订单热门大洲的数量
    feature = feature.fillna(0)  # 空值填补0

    return feature


feature = fun_user(user_info, order_info)


feature.to_csv("../data/workeddata/F2-fun-test-2.csv",
               encoding="gb2312", index=True)

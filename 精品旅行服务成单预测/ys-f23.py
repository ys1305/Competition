import numpy as np
import pandas as pd

# F2 = pd.read_csv("../data/workeddata/F2-fun.csv", sep=",", encoding="UTF-8")
# F3 = pd.read_csv("../data/workeddata/F3ys-fun.csv", sep=",", encoding="UTF-8")

F2 = pd.read_csv("../data/workeddata/F2-fun-test.csv",
                 sep=",", encoding="UTF-8")
F3 = pd.read_csv("../data/workeddata/F3ys-fun-test.csv",
                 sep=",", encoding="UTF-8")

F2_data = F2[["userid", "F2.1", "F2.3"]]
F3_data = F3.iloc[:, 0:13]

F2_3 = F2_data.merge(F3_data, on="userid", how="left")

feature = F2_3[["userid"]]

feature["F2.3.1"] = F2_3["F3.1"] / F2_3["F2.1"]
feature["F2.3.2"] = F2_3["F3.2"] / F2_3["F2.1"]
feature["F2.3.3"] = F2_3["F3.3"] / F2_3["F2.1"]
feature["F2.3.4"] = F2_3["F3.4"] / F2_3["F2.1"]
feature["F2.3.5"] = F2_3["F3.5"] / F2_3["F2.1"]
feature["F2.3.6"] = F2_3["F3.6"] / F2_3["F2.1"]
feature["F2.3.7"] = F2_3["F3.7"] / F2_3["F2.1"]
feature["F2.3.8"] = F2_3["F3.8"] / F2_3["F2.1"]
feature["F2.3.9"] = F2_3["F3.9"] / F2_3["F2.1"]
feature["F2.3.10"] = F2_3["F3.10"] / F2_3["F2.1"]
feature["F2.3.11"] = F2_3["F3.11"] / F2_3["F2.1"]
feature["F2.3.12"] = F2_3["F3.12"] / F2_3["F2.1"]
feature["F2.3.13"] = F2_3["F3.1"] / F2_3["F2.3"]
feature["F2.3.14"] = F2_3["F3.2"] / F2_3["F2.3"]
feature["F2.3.15"] = F2_3["F3.3"] / F2_3["F2.3"]
feature["F2.3.16"] = F2_3["F3.4"] / F2_3["F2.3"]
feature["F2.3.17"] = F2_3["F3.5"] / F2_3["F2.3"]
feature["F2.3.18"] = F2_3["F3.6"] / F2_3["F2.3"]
feature["F2.3.19"] = F2_3["F3.7"] / F2_3["F2.3"]
feature["F2.3.20"] = F2_3["F3.8"] / F2_3["F2.3"]
feature["F2.3.21"] = F2_3["F3.9"] / F2_3["F2.3"]
feature["F2.3.22"] = F2_3["F3.10"] / F2_3["F2.3"]
feature["F2.3.23"] = F2_3["F3.11"] / F2_3["F2.3"]
feature["F2.3.24"] = F2_3["F3.12"] / F2_3["F2.3"]

feature = feature.fillna(0)  # 空值填充

inf = np.isinf(feature)
feature[inf] = float("NaN")  # 极值填充

feature.to_csv("../data/workeddata/F2_3-fun-test.csv",
               encoding="gb2312", index=True)

import numpy as np
import pandas as pd
from tqdm import tqdm
# import distance
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
# from datetime import datetime
from imblearn.over_sampling import SMOTE


# 获取数据
def load_and_process_data(filename):
    data1 = pd.read_excel(filename, header=0)
    data_DataFrame = data1.loc[:, [
        'longitude',
        'latitude',
        'speed',
        'dir',
        'height',
        'tags',
        'time',
    ]]

    # 数据清理
    data_DataFrame.drop_duplicates(inplace=True)
    data_DataFrame.dropna(inplace=True)
    data_DataFrame = data_DataFrame[data_DataFrame['speed'] >= 0]
    data_DataFrame = data_DataFrame[data_DataFrame['height'] >= 0]

    return data_DataFrame.to_numpy()


# def calculate_speed(time_col, distances):
#     time1 = datetime.strptime(time_col[0], '%Y/%m/%d %H:%M:%S')
#     time2 = datetime.strptime(time_col[1], '%Y/%m/%d %H:%M:%S')
#     delta_seconds = (time2 - time1).total_seconds()
#     time_deltas = [delta_seconds]
#     for i in range(1, len(time_col) - 1):
#         time1 = datetime.strptime(time_col[i - 1], '%Y/%m/%d %H:%M:%S')
#         time2 = datetime.strptime(time_col[i + 1], '%Y/%m/%d %H:%M:%S')
#         delta_seconds = ((time2 - time1) / 2).total_seconds()
#         time_deltas.append(delta_seconds)
#     time1 = datetime.strptime(time_col[-2], '%Y/%m/%d %H:%M:%S')
#     time2 = datetime.strptime(time_col[-1], '%Y/%m/%d %H:%M:%S')
#     delta_seconds = (time2 - time1).total_seconds()
#     time_deltas.append(delta_seconds)  # 使用最后一个时间差来填充最后一个元素

#     speeds = np.divide(distances, time_deltas)
#     return speeds

print('loading data...\n')

data_list = []

# 循环获取数据
for i in tqdm(range(20, 21), desc="Loading data"):
    filename = "C:/Users/LMY/Desktop/agricultural machinery/tractor/tractor_{}.xlsx".format(
        i)
    data = load_and_process_data(filename)
    data_list.append(data)

# 将所有数据堆叠在一起
data = np.vstack(data_list)

# 加载数据集
X, y, time_col = data[:, :-2], data[:, -2], data[:, -1]
y = y.astype(int)

# # 计算距离
# distances = [distance.haversine_distance(X[0, 1], X[0, 0], X[1, 1],
#                                          X[1, 0])]  # 第一个点的距离设置为与后一个点的距离
# for i in range(1, len(X) - 1):
#     lat1, lon1 = X[i - 1, 1], X[i - 1, 0]
#     lat2, lon2 = X[i, 1], X[i, 0]
#     lat3, lon3 = X[i + 1, 1], X[i + 1, 0]

#     distance_prev = distance.haversine_distance(lat1, lon1, lat2, lon2)
#     distance_next = distance.haversine_distance(lat2, lon2, lat3, lon3)
#     avg_distance = (distance_prev + distance_next) / 2

#     distances.append(avg_distance)
# distances.append(
#     distance.haversine_distance(X[-2, 1], X[-2, 0], X[-1, 1],
#                                 X[-1, 0]))  # 最后一个点的距离设置为与前一个点的距离

# # 计算速度
# speeds = calculate_speed(time_col, distances)

# # 将速度替换原始速度列
# X[:, 2] = speeds

# # 添加到X中
# X = np.column_stack((X, distances))

# 对X进行归一化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 处理类别不平衡
smote = SMOTE()
X, y = smote.fit_resample(X_normalized, y)

# 对y进行One-Hot编码
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

if __name__ == '__main__':
    print(X)
    print(y)
    print(y_one_hot)

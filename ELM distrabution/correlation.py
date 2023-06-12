import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import loaddata as lddata

# 将X_normalized和y合并到一个数组中
data = np.column_stack((lddata.y, lddata.X))

# 使用Pandas DataFrame计算相关性
data_df = pd.DataFrame(
    data, columns=['LNG', 'LAT', 'SPEED', 'DIRECTION', 'HIGHT', 'y'])

# 将数据转换为浮点数
data_df = data_df.astype(float)

# 检查数据是否包含空值或无穷值
print("Data contains NaN values:", data_df.isnull().values.any())
print("Data contains Inf values:", np.isinf(data_df.to_numpy()).any())

# 删除空值和无穷值
data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
data_df.dropna(inplace=True)

# 计算相关性
corr = data_df.corr()

# 使用seaborn绘制热力图
sns.set(style="white")
plt.figure(figsize=(12, 10))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm',
            square=True,
            cbar_kws={"shrink": .75})
plt.show()

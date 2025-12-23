import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

# 读取处理好的数据
df = pd.read_csv("/home/deep/TimeSeries/Zhendong/data3/processed_motor_data.csv")

print("前几行数据：")
print(df.head())

# 转换成 Chronos 格式
ts_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
print("✅ 转换成功, 总共时间序列:", len(ts_data.item_ids))

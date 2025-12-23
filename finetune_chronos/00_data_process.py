import os
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

# 原始数据目录
DATA_DIR = "/home/deep/TimeSeries/Zhendong/data3"
# 处理后的输出文件
OUTPUT_FILE = "/home/deep/TimeSeries/Zhendong/data3/processed_motor_data.csv"

def load_motor_data(data_dir):
    rows = []

    # 遍历测点（ShengYing / ZhenDong）
    for point in ["ShengYing", "ZhenDong"]:
        point_dir = os.path.join(data_dir, point)
        if not os.path.exists(point_dir):
            continue

        # 遍历状态（normal / spark / vibrate）
        for label in ["normal", "spark", "vibrate"]:
            label_dir = os.path.join(point_dir, label)
            if not os.path.exists(label_dir):
                continue

            # 遍历该目录下的文件
            for fname in os.listdir(label_dir):
                if not fname.endswith(".csv"):
                    continue

                path = os.path.join(label_dir, fname)

                # 读取一条时间序列 (65536 点)
                values = pd.read_csv(path, header=None).squeeze("columns")

                # 时间戳用采样点索引
                timestamps = range(len(values))

                # 唯一 item_id (point_label_filename 去掉扩展名)
                item_id = f"{point}_{label}_{fname.replace('.csv','')}"

                # 组织成长表格式
                df = pd.DataFrame({
                    "item_id": item_id,
                    "timestamp": timestamps,
                    "target": values,
                    "label": label  # 标签方便后续分类/分析
                })
                rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ============ 1. 数据处理 ============
print("🔄 正在处理原始 CSV 数据...")
data = load_motor_data(DATA_DIR)
print(f"✅ 已处理数据: {len(data)} 条记录, {data['item_id'].nunique()} 条时间序列")

# ============ 2. 保存处理结果 ============
# 保存为 CSV（便于直接查看）
data.to_csv(OUTPUT_FILE, index=False)
print(f"💾 已保存处理后的数据到: {OUTPUT_FILE}")

# 也可以保存成 Parquet，读取速度更快
# data.to_parquet(OUTPUT_FILE.replace(".csv", ".parquet"))

# ============ 3. 转换为 Chronos 格式 ============
ts_data = TimeSeriesDataFrame.from_data_frame(
    data,
    id_column="item_id",
    timestamp_column="timestamp"
)
print("✅ 已转换为 TimeSeriesDataFrame，可直接用于 Chronos")

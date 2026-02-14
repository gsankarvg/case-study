import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # create time features
    df["Month"] = df["Datetime"].dt.month
    df["Day_of_week"] = df["Datetime"].dt.dayofweek + 1   # 1–7
    df["Hour"] = df["Datetime"].dt.hour

    return df

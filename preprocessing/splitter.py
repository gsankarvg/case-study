from config import TRAIN_HOURS, VAL_HOURS

def split_data(df):
    train = df.iloc[:TRAIN_HOURS]
    val = df.iloc[TRAIN_HOURS:TRAIN_HOURS + VAL_HOURS]
    test = df.iloc[TRAIN_HOURS + VAL_HOURS:]
    return train, val, test

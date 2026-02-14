from sklearn.preprocessing import MinMaxScaler

def scale_data(train, val, test):
    scaler = MinMaxScaler()
    
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    return train_scaled, val_scaled, test_scaled, scaler

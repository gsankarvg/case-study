from tensorflow.keras.callbacks import EarlyStopping
from config import EPOCHS

def train_model(model, X_train, y_train, X_val, y_val, model_type):

    if model_type == "lstm":
        batch_size = 168
        patience = 15
    elif model_type == "cnn":
        batch_size = 128
        patience = 30
    else:
        batch_size = 168
        patience = 15

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return history

import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, save_path, day_index=0):

    plt.figure(figsize=(10,4))

    plt.plot(y_true[day_index], label="Actual")
    plt.plot(y_pred[day_index], label="Predicted")

    plt.xlabel("Hour")
    plt.ylabel("Electricity")
    plt.title("Actual vs Predicted (24-hour forecast)")
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def plot_loss(history, save_path):
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(save_path)
    plt.close()

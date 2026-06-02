import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


def train(model, pipeline, optimizer=Adam(learning_rate=3e-4), epochs=50):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    history = model.fit(
        pipeline.train_ds,
        epochs=epochs,
        validation_data=pipeline.val_ds,
    )

    return history


def plot_learning_curves(hist):
    epochs = len(hist.history["loss"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(range(1, epochs+1), hist.history['accuracy'], 'b-', label='Train accuracy')
    ax[0].plot(range(1, epochs+1), hist.history['val_accuracy'], 'r-', label='Validation accuracy')
    ax[0].set(xlabel='Epoch', ylabel='Accuracy')
    ax[0].legend()
    
    ax[1].plot(range(1, epochs+1), hist.history['loss'], 'b-', label='Train loss')
    ax[1].plot(range(1, epochs+1), hist.history['val_loss'], 'r-', label='Validation loss')
    ax[1].set(xlabel='Epoch', ylabel='Loss')
    ax[1].legend()
    
    plt.show()

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

epochs = 10

def make_model():
    model = keras.Sequential()

    # Pre-processing
    model.add(keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(keras.layers.experimental.preprocessing.Rescaling(1.0 / 255))

    # First layer pair (convolution and pooling)
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the matrices into an array 
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Deep neural network step
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def main():
    # Fetch dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Create model
    model = make_model()

    # Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # Train model
    history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_test, y_test))

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia do modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_model(
    input_shape=(224, 224, None),  # Accept dynamic channels (grayscale or RGB)
    num_classes=1,
    activation='sigmoid',
    optimizer=Adam(learning_rate=0.001),
    dropout_rate=0.5,
    filters=[32, 64, 128],
    kernel_size=(3, 3),
    pool_size=(2, 2),
    dense_units=[128, 64]
):
    """
    Builds a customizable CNN model that supports both grayscale and color images.

    Args:
        input_shape (tuple): Shape of the input images, e.g., (224, 224, None) for dynamic channels.
        num_classes (int): Number of output classes (1 for binary classification, >1 for multi-class).
        activation (str): Activation function for the final layer ('sigmoid' for binary, 'softmax' for multi-class).
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for model compilation.
        dropout_rate (float): Dropout rate to prevent overfitting.
        filters (list): List of filter sizes for Conv2D layers.
        kernel_size (tuple): Kernel size for Conv2D layers.
        pool_size (tuple): Pool size for MaxPooling2D layers.
        dense_units (list): List of units for Dense layers.

    Returns:
        keras.Model: A compiled Keras model.
    """
    model = Sequential([
        Dense(10, activation='relu', input_shape=(20,)),
        Dense(1, activation='sigmoid')
]   )

    # Add convolutional layers dynamically based on the `filters` list
    for i, filter_size in enumerate(filters):
        if i == 0:
            # First layer with input shape
            model.add(Input(shape=input_shape))  # Dynamic channel input
            model.add(Conv2D(filter_size, kernel_size, activation='relu'))
        else:
            model.add(Conv2D(filter_size, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_rate))  # Dropout after each pooling layer

    # Flatten the output
    model.add(Flatten())

    # Add dense layers dynamically based on the `dense_units` list
    for units in dense_units:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))  # Dropout for fully connected layers

    # Add the output layer
    model.add(Dense(num_classes, activation=activation))

    # Compile the model
    loss = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
